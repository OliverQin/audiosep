#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import random
import sys

def list_wavs(in_dir: Path):
    files = sorted(list(in_dir.glob("*.wav")) + list(in_dir.glob("*.WAV")))
    return [p for p in files if p.is_file()]

def ensure_stereo(x: np.ndarray) -> np.ndarray:
    """x: (frames, channels) float32 -> (frames, 2) float32"""
    if x.ndim != 2:
        raise ValueError(f"audio array must be 2D, got shape {x.shape}")
    c = x.shape[1]
    if c == 1:
        x = np.repeat(x, 2, axis=1)
    elif c >= 2:
        x = x[:, :2]
    return x.astype(np.float32, copy=False)

def resample_linear(x: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """Simple linear-resample to tgt_sr using np.interp, channel-wise."""
    if src_sr == tgt_sr or x.shape[0] <= 1:
        return x
    r = float(tgt_sr) / float(src_sr)
    n_src = x.shape[0]
    n_tgt = int(round(n_src * r))
    old_idx = np.arange(n_src, dtype=np.float64)
    new_idx = np.linspace(0, n_src - 1, n_tgt, dtype=np.float64)
    # channel-wise interpolate
    if x.shape[1] == 1:
        y = np.interp(new_idx, old_idx, x[:, 0]).astype(np.float32)[:, None]
    else:
        chans = []
        for ch in range(x.shape[1]):
            chans.append(np.interp(new_idx, old_idx, x[:, ch]).astype(np.float32))
        y = np.stack(chans, axis=1)
    return y

def pad_to_length(x: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad x (frames, 2) to 'length' frames."""
    if x.shape[0] == length:
        return x
    pad = np.zeros((length - x.shape[0], x.shape[1]), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)

def mix_three(a, b, c, w):
    """All inputs are (frames, 2) float32; w is length-3 weights summing to 1."""
    return w[0] * a + w[1] * b + w[2] * c

def main():
    parser = argparse.ArgumentParser(
        description="Randomly blend 3 WAVs into float32 stereo, repeated N times."
    )
    parser.add_argument("--in_dir", required=True, type=Path, help="Input folder with WAV files")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output folder")
    parser.add_argument("--num", type=int, default=50, help="How many blended files to generate")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for file naming")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    wavs = list_wavs(args.in_dir)
    if len(wavs) < 3:
        print("ERROR: Need at least 3 WAV files in input directory.", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num):
        # Randomly pick 3 distinct files
        picks = random.sample(wavs, 3)

        datas = []
        srs = []
        for p in picks:
            # Read as float32; always_2d keeps (frames, channels)
            x, sr = sf.read(p, dtype="float32", always_2d=True)
            x = ensure_stereo(x)
            datas.append(x)
            srs.append(sr)

        # Resample to a common samplerate (use the max to preserve bandwidth)
        target_sr = max(srs)
        datas = [resample_linear(x, sr, target_sr) for x, sr in zip(datas, srs)]

        # Pad to the longest
        max_len = max(x.shape[0] for x in datas)
        datas = [pad_to_length(x, max_len) for x in datas]

        # Random weights in [0,1], sum to 1 (Dirichlet)
        w = np.random.dirichlet(np.ones(3)).astype(np.float32)

        # Mix
        y = mix_three(datas[0], datas[1], datas[2], w).astype(np.float32)

        out_path = args.out_dir / f"blended_{args.start_index + i:06d}.wav"
        # Write as float32 stereo WAV
        sf.write(out_path, y, samplerate=target_sr, subtype="FLOAT")
        print(f"Wrote {out_path.name}  |  sr={target_sr}  |  weights={w.tolist()}  |  picks={[p.name for p in picks]}")

if __name__ == "__main__":
    main()

