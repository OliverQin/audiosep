#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip install torch torchaudio soundfile numpy

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@torch.no_grad()
def fft_convolve_same(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    逐通道 FFT 卷积并裁剪成和 h 同长度（'same'）。
    x: [2, Lx], h: [2, Lh]  ->  y: [2, Lh]
    """
    assert x.ndim == 2 and h.ndim == 2 and x.size(0) == h.size(0) == 2
    Lx, Lh = x.size(1), h.size(1)
    out_len_full = Lx + Lh - 1
    nfft = next_pow2(out_len_full)

    y_full = []
    for ch in range(2):
        Xf = torch.fft.rfft(x[ch], n=nfft)
        Hf = torch.fft.rfft(h[ch], n=nfft)
        Yf = Xf * Hf
        y = torch.fft.irfft(Yf, n=nfft)[:out_len_full]  # [Lx+Lh-1]
        y_full.append(y)
    y_full = torch.stack(y_full, dim=0)                 # [2, Lx+Lh-1]

    # SAME 裁剪：取从 0 开始的前 Lh 样本（因我们已把 gunshot 旋转到从 T 对齐到 0）
    y_same = y_full[:, :Lh].contiguous()                # [2, Lh]
    return y_same


def build_envelope(L: int, T: int, N: int, M: int, cutoff: float,
                   halo_sec: float, sr: int, dtype=torch.float32) -> torch.Tensor:
    """
    构造满足你规则的包络：
      - [T-N, T] cutoff->1 线性上升；[T, T+M] 1->cutoff 线性下降；
      - 之外：在 T- halo_sec 与 T+ halo_sec 处分段，从 cutoff 线性降到 0；
      - 其他区间= 0
    返回: [L] 的 1D 张量
    """
    env = torch.zeros(L, dtype=dtype)

    # 关键边界
    l_sec = int(round(halo_sec * sr))
    left_taper_start  = max(0, T - l_sec)
    right_taper_start = min(L, T + l_sec)
    left_plateau_end  = max(0, T - N)      # plateau 左端终点（不含）
    right_plateau_beg = min(L, T + M)      # plateau 右端起点（含/后续处理注意）

    # 左侧 0 -> cutoff 线性上升（到 T - halo_sec）
    if left_taper_start > 0:
        env[:left_taper_start] = 0.0

    # 左侧 plateau: [T - halo_sec, T - N) = cutoff
    a = left_taper_start
    b = left_plateau_end
    if b > a:
        env[a:b] = torch.linspace(0.0, cutoff, steps=(b - a), dtype=dtype)

    # ramp up: [T-N, T) cutoff -> 1
    up_a = max(0, T - N)
    up_b = max(0, T)
    if up_b > up_a:
        env[up_a:up_b] = torch.linspace(cutoff, 1.0, steps=(up_b - up_a), dtype=dtype)

    # center sample T = 1.0
    if 0 <= T < L:
        env[T] = 1.0

    # ramp down: (T, T+M] 1 -> cutoff
    down_a = T
    down_b = right_plateau_beg
    if down_b > down_a:
        env[down_a:down_b] = torch.linspace(1.0, cutoff, steps=(down_b - down_a), dtype=dtype)

    # 右侧 plateau: [T+M+1, T + halo_sec) = cutoff
    plat_a = right_plateau_beg
    plat_b = right_taper_start
    if plat_b > plat_a:
        env[plat_a:plat_b] = torch.linspace(cutoff, 0.0, steps=(plat_b - plat_a), dtype=dtype)

    # 右侧 cutoff -> 0 线性下降： [T+halo_sec, L)
    if right_taper_start < L:
        env[right_taper_start:] = 0.0

    return env


def main():
    ap = argparse.ArgumentParser(description="Convolve a 'shaped' gunshot with a stereo Float32 RIR (same length output).")
    ap.add_argument("--rir", required=True, type=Path, help="Stereo Float32 WAV RIR (no resample allowed)")
    ap.add_argument("--gunshot", required=True, type=Path, help="Gunshot WAV (any sr/ch)")
    ap.add_argument("--sr", type=int, default=44100, help="Processing sample rate (RIR must already be this sr)")
    ap.add_argument("--cutoff", type=float, default=0.05, help="Envelope floor (default 0.05)")
    ap.add_argument("--halo_sec", type=float, default=0.2, help="侧翼线性降到 0 的时间（秒），默认 0.2s")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--out", type=Path, default=Path("simulated_rir.wav"), help="Output WAV filename (float32)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    target_sr = int(args.sr)

    # 1) 读取并验证 RIR（必须 WAV + 双声道 + Float32 + 采样率匹配）
    info = sf.info(str(args.rir))
    if info.format != "WAV":
        raise ValueError(f"RIR must be WAV, got {info.format}")
    if info.channels != 2:
        raise ValueError(f"RIR must be stereo (2ch), got {info.channels}")
    if info.subtype != "FLOAT":
        raise ValueError(f"RIR must be Float32 WAV (subtype='FLOAT'), got {info.subtype}")
    if info.samplerate != target_sr:
        raise ValueError(f"RIR sr mismatch: expected {target_sr}, got {info.samplerate}")

    rir_np, rir_sr = sf.read(str(args.rir), dtype="float32")  # [L, 2]
    rir = torch.from_numpy(rir_np.T.copy())                   # -> [2, L]

    # 2) 读取 gunshot（任意 sr/ch）
    g_wave, g_sr = torchaudio.load(str(args.gunshot))         # [C, T]

    # 3) 若单声道：右声道 = 左声道“提前”1~3 个 sample（按原 sr 的 frame）
    if g_wave.size(0) == 1:
        print('Warning: Gunshot is mono, duplicating and shifting right channel.', file=sys.stderr)
        shift = random.randint(1, 3)
        left = g_wave[0]
        right = torch.cat([left[shift:], torch.zeros(shift, dtype=left.dtype)])
        g_wave = torch.stack([left, right], dim=0)
    elif g_wave.size(0) >= 2:
        g_wave = g_wave[:2, :]
    else:
        raise ValueError("Gunshot has zero channels")

    # 4) 重采样 gunshot → target_sr（RIR 不重采样）
    if g_sr != target_sr:
        g_wave = torchaudio.functional.resample(g_wave, orig_freq=g_sr, new_freq=target_sr)

    # 5) 找到 T（左右联合的最大绝对值点）
    abs_mix = g_wave.abs().max(dim=0).values
    T = int(torch.argmax(abs_mix).item())

    # 6) 采样 N, M
    N = random.randrange(3, 8)      # [3, 8)
    M = random.randrange(30, 100)  # [30, 100)
    cutoff = float(args.cutoff)

    # 7) 构造包络并施加
    Lg = g_wave.size(1)
    env = build_envelope(L=Lg, T=T, N=N, M=M, cutoff=cutoff,
                         halo_sec=float(args.halo_sec), sr=target_sr, dtype=g_wave.dtype)

    sf.write('env_debug.wav', env.cpu().numpy().astype(np.float32), 44100, subtype="FLOAT")
    g_shaped = g_wave * env.unsqueeze(0)

    sf.write('g_shaped_debug.wav', g_shaped.T.cpu().numpy().astype(np.float32), 44100, subtype="FLOAT")

    # 8) 旋转 gunshot（从 T 开始复制到 0；长度对齐 RIR）
    Lrir = rir.size(1)
    g_rot = torch.cat([g_shaped[:, T:], g_shaped[:, :T]], dim=1)  # [2, Lg]
    if g_rot.size(1) >= Lrir:
        g_rot = g_rot[:, :Lrir]
    else:
        pad = torch.zeros((2, Lrir - g_rot.size(1)), dtype=g_rot.dtype)
        g_rot = torch.cat([g_rot, pad], dim=1)  # [2, Lrir]

    sf.write('g_rot_debug.wav', g_rot.T.cpu().numpy().astype(np.float32), 44100, subtype="FLOAT")

    # 9) FFT 卷积 → SAME 长度，并归一化到 < 0.95
    y = fft_convolve_same(g_rot, rir)  # [2, Lrir]
    peak = float(y.abs().max())
    if peak > 0:
        y = y * ((0.95 - 1e-6) / peak)

    # 输出 WAV（float32），并给个简单反馈
    y_np = y.T.cpu().numpy().astype(np.float32)  # [L, 2]
    sf.write(str(args.out), y_np, target_sr, subtype="FLOAT")
    print(f"✅ Done. Wrote {args.out} | shape={tuple(y.shape)} sr={target_sr}  N={N} M={M} halo={args.halo_sec}s")

    # 如果你在脚本中当库使用，也可以在这里 return y
    return y


if __name__ == "__main__":
    main()

    # out = build_envelope(44100 * 10, 44100 * 5, 5, 100, 0.05, 0.5, 44100, dtype=torch.float32)
    # sf.write('envelop.wav', out.cpu().numpy().astype(np.float32), 44100, subtype="FLOAT")
