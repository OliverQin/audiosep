#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time

import numpy as np
import soundfile as sf
import tqdm


def random_long_tail(
    sr=44100,
    attack_duration_ms=0.8,  # Attack duration in milliseconds
    rt60=1.0,                # Reverberation time in seconds
) -> np.ndarray:
    """Generate a random long-tail sound with exponential decay."""
    N = int(sr * rt60)

    ln1000 = np.log(1000)
    sound = np.zeros(N, dtype=np.float64)

    t = np.linspace(0, -ln1000, N, endpoint=False)
    env = (np.exp(t) + 0.45 * np.exp(t * 4.5) + 0.15 * np.exp(t * 17.0)) * (1.0 / 1.6)

    M = int(sr * attack_duration_ms / 1000.0)
    assert M < N, "Attack duration must be less than total duration"
    t2 = np.linspace(0, -ln1000, M, endpoint=False)
    attack = 1.0 - np.exp(t2)
    env[:M] *= attack

    rng = np.random.default_rng(seed=int(time.time() * 1e6))
    for _i in range(3):
        sound += rng.uniform(-1.0, 1.0, size=N)
    sound *= env / 3.0
    sound[env.argmax()] = 1.0  # Ensure peak is at 1.0

    # Time-varying low-pass filter
    fc0, fc1 = 19000.0, 1000.0
    fc = fc1 + (fc0 - fc1) * np.exp(t)
    coeff = np.exp(-2.0 * np.pi * fc / sr)
    for i in range(1, N):
        sound[i] = (1 - coeff[i]) * sound[i] + coeff[i] * sound[i - 1]

    sound[env.argmax()] = 1.0  # Ensure peak is at 1.0 after filtering
    sound /= np.sqrt((sound ** 2).sum()) # Normalize to unit energy

    return sound.astype(np.float32)

def convolve_same_fft(x: np.ndarray, krn: np.ndarray) -> np.ndarray:
    krn = np.pad(krn, (0, len(x) - len(krn)), mode="constant", constant_values=0)

    idx = krn.argmax()
    if idx > 0:
        krn = np.roll(krn, -idx)
    
    x_fft = np.fft.rfft(x)
    k_fft = np.fft.rfft(krn)
    y_fft = x_fft * k_fft
    y = np.fft.irfft(y_fft)

    if len(y) > len(x):
        y = y[:len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)), mode="constant", constant_values=0)

    return y

def augment_rir_with_long_tail(x, modules=16, sr=44100, rt60=1.0, attack_duration_ms=0.8):
    y = np.empty_like(x, dtype=np.float32)
    length, channels = x.shape
    rng = np.random.default_rng(seed=int(time.time() * 1e6))
    idx = rng.choice(modules, size=length)
    
    for i in tqdm.tqdm(range(modules)):
        krn = random_long_tail(sr=sr, attack_duration_ms=attack_duration_ms, rt60=rt60).astype(np.float64)
        x_selected = np.zeros_like(x)
        x_selected[idx == i] = x[idx == i]

        for c in range(channels):
            y[:, c] += convolve_same_fft(x_selected[:, c].astype(np.float64), krn).astype(np.float32)

    return y

def main():
    ap = argparse.ArgumentParser(description="Convolve input WAV with a random long tail (float32 output).")
    ap.add_argument("input_wav", type=str, help="Path to input WAV")
    ap.add_argument("output_wav", type=str, nargs="?", help="Path to output WAV (optional)")
    ap.add_argument("--attack_ms", type=float, default=0.8, help="Attack duration in ms for the long tail (default: 0.8)")
    args = ap.parse_args()

    # 读取并转换为 float32（soundfile 会把 16/24-bit 自动转换为 [-1,1] 的 float32）
    x, sr = sf.read(args.input_wav, always_2d=True, dtype="float32")  # shape: [N, C]
    N, C = x.shape
    duration_sec = N / float(sr)
    rt60 = 0.5 * duration_sec  # 文件总长度的一半

    y = augment_rir_with_long_tail(
        x,
        modules=16,  # 模块数量
        sr=sr,
        rt60=rt60,  # 使用文件总长度的一半作为 RT60
        attack_duration_ms=args.attack_ms
    )

    # 写出为 float32 的 WAV（IEEE float）
    out_path = args.output_wav or os.path.splitext(args.input_wav)[0] + "_longtail.wav"
    sf.write(out_path, y, sr, subtype="FLOAT")
    print(f"Wrote: {out_path}\nSample rate: {sr}\nShape: {y.shape}\nDtype: float32")


if __name__ == "__main__":
    main()
