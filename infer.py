#!/usr/bin/env python3

"""
Infer script: load model ckpt, process a FLAC in chunks with 1s crossfade,
write out two WAVs: *_music.wav and *_noise.wav
"""

import os
import argparse
import pathlib
import soundfile
import soxr
import ffmpeg
import numpy as np
import torch

from scipy.signal import medfilt

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from pathlib import Path
from htdemucs import *

from tqdm import tqdm

def load_audio_ffmpeg(path, target_sr=44100):
    out, _ = (
        ffmpeg
        .input(path)
        .output('pipe:', format='f32le', acodec='pcm_f32le', ac=2, ar=target_sr)
        .run(capture_stdout=True, capture_stderr=True)
    )
    audio = np.copy(np.frombuffer(out, np.float32).reshape(-1, 2))
    return audio, target_sr

def load_audio(path, target_sr=44100):
    audio, sr = soundfile.read(path, always_2d=True)  # (T,C)
    if sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr, quality="VHQ")
    return audio.astype(np.float32), target_sr  # (T,C)

def save_audio(path, data, sr, subtype="FLOAT"):
    soundfile.write(path, data, sr, subtype=subtype)

def overlap_add_sliding(x, chunk_len, hop, process_fn):
    """
    x: np.ndarray (T, C), float32
    chunk_len: window length (samples)
    hop: step size (<= chunk_len)
    process_fn: fn(chunk_np) -> (music_np, noise_np), shape (chunk_len, C)
    """
    T, C = x.shape
    n_chunks = 1 + int(np.ceil((T - chunk_len) / hop)) if T > chunk_len else 1

    # Hann window：window L
    win = np.hanning(chunk_len).astype(np.float32) + 1e-8  # [0..1..0]
    win = win[:, None]  # (L,1) broadcast to LC

    out_music = np.zeros((T, C), dtype=np.float32)
    out_noise = np.zeros((T, C), dtype=np.float32)
    weight    = np.zeros((T, C), dtype=np.float32)

    for i in range(n_chunks):
        start = i * hop
        rnd_shift = np.random.randint(-hop // 2, hop // 2)

        start += rnd_shift
        if start < 0:
            start = 0
        
        end = start + chunk_len
        if end > T:
            start -= (end - T)
            end = T

        chunk = x[start:end]

        # infer
        m_chunk, n_chunk = process_fn(chunk)  # (chunk_len, C)

        # pad
        valid = min(chunk_len, T - start)
        w = win[:valid]

        out_music[start:start+valid] += m_chunk[:valid] * w
        out_noise[start:start+valid] += n_chunk[:valid] * w
        weight[start:start+valid]    += w

    weight[weight == 0] = 1.0
    out_music /= weight
    out_noise /= weight
    return out_music, out_noise

def overlap_add_process(x, chunk_len, fade_len, process_fn):
    """
    x: np.ndarray (T, C) float32
    chunk_len: samples per chunk
    fade_len: samples for crossfade (>=0)
    process_fn: function(np_chunk)->(np_music,np_noise)  each (chunk_len,C)

    Returns music_out, noise_out (T,C)
    """
    T, C = x.shape
    if fade_len > chunk_len // 2:
        raise ValueError("fade_len too large relative to chunk_len")

    hop = chunk_len - fade_len  # stride
    n_chunks = int(np.ceil((T - fade_len) / hop))
    out_music = np.zeros((T, C), dtype=np.float32)
    out_noise = np.zeros((T, C), dtype=np.float32)
    weight    = np.zeros((T, C), dtype=np.float32)

    # crossfade windows
    fade_in  = np.linspace(0, 1, fade_len, dtype=np.float32)
    fade_out = 1.0 - fade_in

    for i in range(n_chunks):
        start = i * hop
        end   = min(start + chunk_len, T)
        chunk = np.zeros((chunk_len, C), dtype=np.float32)
        chunk[:end-start] = x[start:end]

        print(chunk.shape, 'shape')

        m_chunk, n_chunk = process_fn(chunk)  # (chunk_len,C)
        valid = end - start

        # apply fades on edges (only where we overlap)
        if i > 0:  # has overlap on the left
            m_chunk[:fade_len] *= fade_in[:, None]
            n_chunk[:fade_len] *= fade_in[:, None]
        if i < n_chunks - 1:  # will overlap to the right
            m_chunk[valid - fade_len:valid] *= fade_out[-fade_len:, None]
            n_chunk[valid - fade_len:valid] *= fade_out[-fade_len:, None]

        out_music[start:end] += m_chunk[:valid]
        out_noise[start:end] += n_chunk[:valid]
        weight[start:end]    += 1.0  # simple weight, since we already shaped with fades

    # avoid div by zero (shouldn't happen)
    weight[weight == 0] = 1.0
    out_music /= weight
    out_noise /= weight
    return out_music, out_noise

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=pathlib.Path)
    ap.add_argument("flac", type=pathlib.Path)
    ap.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("infer_out"))
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--chunk_len", type=int, default=2**18, help="samples per chunk expected by model")
    ap.add_argument("--fade_sec", type=float, default=1.0, help="crossfade seconds")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- load audio ---
    audio_np, sr = load_audio_ffmpeg(args.flac, args.sr)  # (T,C)
    audio_np *= 0.9 / np.abs(audio_np).max()
    T, C = audio_np.shape

    # --- load model ---
    model = HTDemucs(['music', 'noise'], t_layers=7)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("model", ckpt.get("model_state", ckpt))  # try several keys
    model.load_state_dict(state_dict)
    model.use_train_segment = False
    model.to(device)
    model.eval()

    fade_len = int(args.fade_sec * sr)

    @torch.no_grad()
    def process_fn(chunk_np):
        # chunk_np: (chunk_len, C)
        # to tensor: (1,C,T)
        x = torch.from_numpy(chunk_np.T).unsqueeze(0).to(device)  # (1,C,T)
        out = model(x)  # (B=1, 2, C, T)
        print('outshape', out.shape)
        music = out[:, 0].squeeze(0).cpu().numpy().T  # (T,C)
        noise = out[:, 1].squeeze(0).cpu().numpy().T
        return music, noise

    # music_np, noise_np = overlap_add_process(audio_np, args.chunk_len, fade_len, process_fn)
    music_np, noise_np = overlap_add_sliding(audio_np, args.chunk_len, args.chunk_len // 32, process_fn)
    # music_np[np.abs(music_np) < 0.01] = 0.0
    # noise_np[np.abs(noise_np) < 0.01] = 0.0
    # music_np = medfilt(music_np, kernel_size=[11, 1])
    # noise_np = medfilt(noise_np, kernel_size=[11, 1])

    stem = args.flac.stem
    save_audio(args.out_dir / f"{stem}_music.wav", music_np, sr)
    save_audio(args.out_dir / f"{stem}_noise.wav", noise_np, sr)
    print("Done:", args.out_dir)

if __name__ == "__main__":
    main()
