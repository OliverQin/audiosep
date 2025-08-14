#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

@torch.no_grad()
def rir_spectro_densify(
    x: torch.Tensor,           # [C, T], float32, on device
    sr: int,
    n_fft: int = 2048,
    hop: int = 512,
    win: torch.Tensor = None,
    t_split_ms: float = 40.0,  # 早期/后期分界（只对后期做致密化）
    blend_ms: float = 80.0,    # 从原始->致密化的线性过渡时长
    kF: int = 21,              # 频率维局部均值核大小（奇数）
    kT: int = 21,              # 时间维局部均值核大小（奇数）
    stereo_lock: bool = True,  # 立体声用同一随机场，避免左右不一致
    energy_match: bool = True, # 每帧能量匹配，保持原始包络
    eps: float = 1e-8
) -> torch.Tensor:
    """
    返回致密化后的 RIR，形状与 x 相同（[C, T]）。
    思路：后期混响部分的 STFT 复系数 ~ 局部方差匹配的复高斯，幅度分布更接近 Rayleigh。
    """

    C, T = x.shape
    device = x.device
    if win is None:
        win = torch.hann_window(n_fft, periodic=False, device=device, dtype=x.dtype)
    # STFT: [C, F, Tfr]
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True)
    Fbins, Tfr = X.shape[1], X.shape[2]

    # 计算“早-晚”分界与过渡帧数
    split = int(round((t_split_ms / 1000.0) * sr / hop))
    blend = int(round((blend_ms   / 1000.0) * sr / hop))
    split = max(0, min(split, Tfr))
    blend = max(1, blend)

    # 局部能量估计（|X|^2 的 2D 盒状平滑，保留频谱着色 + 时间包络）
    mag2 = (X.real**2 + X.imag**2)  # [C, F, Tfr]
    # 用 conv2d 做均值平滑
    padF, padT = kF//2, kT//2
    kernel = torch.ones((1, 1, kF, kT), device=device, dtype=mag2.dtype) / (kF*kT)
    mag2_ = mag2.unsqueeze(1)  # [C,1,F,T]

    mag2_pad = F.pad(mag2_, (padT, padT, padF, padF), mode='reflect')
    pow_local = F.conv2d(mag2_pad, kernel)  # [C,F,T]

    mag2_pad = F.pad(mag2_, (padT, padT, padF, padF), mode='reflect')
    pow_local = F.conv2d(mag2_pad, kernel)  # [C,F,T]

    mag2_pad = F.pad(mag2_, (padT, padT, padF, padF), mode='reflect')
    pow_local = F.conv2d(mag2_pad, kernel)  # [C,F,T]

    mag2_pad = F.pad(pow_local, (padT, padT, padF, padF), mode='reflect')
    pow_local = F.conv2d(mag2_pad, kernel).squeeze(1)  # [C,F,T]

    sigma = torch.sqrt(torch.clamp(pow_local, min=0.0) + eps)  # 局部幅度标准差

    # 目标：致密化后的复系数 ~ 复高斯(0, sigma^2)
    # 生成复高斯随机场 G：CN(0,1)，再按 sigma 缩放
    if stereo_lock and C == 2:
        # 共享同一随机相位场，左右仅按各自 sigma 放缩
        gr = torch.randn(1, Fbins, Tfr, device=device, dtype=X.real.dtype)
        gi = torch.randn(1, Fbins, Tfr, device=device, dtype=X.real.dtype)
        G = (gr + 1j*gi) / np.sqrt(2.0)  # unit power
        Xdiff = G.repeat(2, 1, 1) * sigma  # [2,F,T]
    else:
        gr = torch.randn(C, Fbins, Tfr, device=device, dtype=X.real.dtype)
        gi = torch.randn(C, Fbins, Tfr, device=device, dtype=X.real.dtype)
        G = (gr + 1j*gi) / np.sqrt(2.0)
        Xdiff = G * sigma

    # 能量包络匹配（每一帧匹配频率平均能量）
    if energy_match:
        E_orig = torch.sqrt(torch.clamp(mag2.mean(dim=1), min=0.0) + eps)   # [C, T]
        E_new  = torch.sqrt(torch.clamp((Xdiff.real**2 + Xdiff.imag**2).mean(dim=1), min=0.0) + eps)  # [C, T]
        scale = (E_orig / torch.clamp(E_new, min=eps)).unsqueeze(1)  # [C,1,T]
        Xdiff = Xdiff * scale

    # 仅对后段应用致密化；前段保留；中间做线性过渡
    beta = torch.zeros((Tfr,), device=device, dtype=X.real.dtype)
    if split < Tfr:
        end = min(Tfr, split + blend)
        if end > split:
            ramp = torch.linspace(0, 1, end - split, device=device, dtype=beta.dtype)
            beta[split:end] = ramp
        if end < Tfr:
            beta[end:] = 1.0
    beta = beta.view(1, 1, Tfr)  # [1,1,T]
    X_out = (1.0 - beta) * X + beta * Xdiff

    # iSTFT 回去
    y = torch.istft(X_out, n_fft=n_fft, hop_length=hop, window=win, center=True, length=T)
    return y

def main():
    ap = argparse.ArgumentParser(description="Densify/equalize STFT texture of image-method RIR to mimic real late reverb.")
    ap.add_argument("input_wav", type=str)
    ap.add_argument("output_wav", type=str)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--t_split_ms", type=float, default=40.0, help="early/late split; only late part is diffused")
    ap.add_argument("--blend_ms", type=float, default=80.0, help="crossfade length from original to diffused")
    ap.add_argument("--kF", type=int, default=7, help="freq smoothing kernel")
    ap.add_argument("--kT", type=int, default=21, help="time smoothing kernel")
    ap.add_argument("--no_stereo_lock", action="store_true", help="disable shared random field for L/R")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    x_np, sr = sf.read(args.input_wav, always_2d=True, dtype="float32")  # [N, C]
    x_np = x_np.T  # [C, T]
    C, T = x_np.shape
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    x = torch.from_numpy(x_np).to(device)
    y = rir_spectro_densify(
        x, sr,
        n_fft=args.n_fft, hop=args.hop,
        t_split_ms=args.t_split_ms, blend_ms=args.blend_ms,
        kF=args.kF, kT=args.kT,
        stereo_lock=(not args.no_stereo_lock)
    ).clamp_(-1.0, 1.0)

    y_np = y.detach().cpu().numpy().T  # [T, C]
    sf.write(args.output_wav, y_np, sr, subtype="FLOAT")
    print(f"Done. Wrote {args.output_wav}  shape={y_np.shape}  sr={sr}")

if __name__ == "__main__":
    main()