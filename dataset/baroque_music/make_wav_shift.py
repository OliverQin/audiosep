#!/usr/bin/env python3
import argparse
import pathlib
import random
import soundfile as sf
import soxr
import numpy as np
import pyrubberband as prb
from tqdm import tqdm

def stratified_two_steps(low=-2.0, high=2.0, rnd=None):
    """Return two pitch shift steps: one from [low,0) and one from (0,high]."""
    rnd = rnd or random
    neg = rnd.uniform(low, 0.0) if low < 0 else 0.0
    pos = rnd.uniform(0.0, high) if high > 0 else 0.0
    if abs(neg) < 1e-6: neg = -1e-4
    if abs(pos) < 1e-6: pos =  1e-4
    return neg, pos

def pitch_shift_rb(wav_np: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """
    wav_np: shape (T, C), float32/64
    Rubber Band expects mono arrays, so process per channel then stack.
    """
    if abs(n_steps) < 1e-6:
        return wav_np
    outs = []
    for ch in range(wav_np.shape[1]):
        outs.append(prb.pitch_shift(wav_np[:, ch], sr, n_steps))
    return np.stack(outs, axis=1)

index = 0

def process_file(flac_path, out_dir, sr=44100, bitdepth="FLOAT", rnd=None):
    global index
    rnd = rnd or random
    data, in_sr = sf.read(flac_path, always_2d=True)  # (T, C)
    if in_sr != sr:
        data = soxr.resample(data, in_sr, sr, quality="VHQ")

    neg_step, pos_step = stratified_two_steps(-2.0, 2.0, rnd=rnd)

    # Pitch shift
    data_neg = pitch_shift_rb(data, sr, neg_step)
    data_pos = pitch_shift_rb(data, sr, pos_step)

    stem = flac_path.stem

    sf.write(out_dir / f"{index:06d}.wav",     data, sr, subtype=bitdepth)
    index += 1
    sf.write(out_dir / f"{index:06d}.wav", data_neg, sr, subtype=bitdepth)
    index += 1
    sf.write(out_dir / f"{index:06d}.wav", data_pos, sr, subtype=bitdepth)
    index += 1

def main():
    ap = argparse.ArgumentParser(description="Each FLAC -> 3 WAVs (orig + two stratified pitch shifts) using pyrubberband.")
    ap.add_argument("input_dir", type=pathlib.Path)
    ap.add_argument("output_dir", type=pathlib.Path)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--bitdepth", default="FLOAT",
                    choices=["PCM_16","PCM_24","PCM_32","FLOAT","DOUBLE"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flacs = sorted(args.input_dir.glob("*.flac"))
    if not flacs:
        print("No FLAC files found.")
        return

    for f in tqdm(flacs, desc="Processing", unit="file"):
        process_file(f, args.output_dir, sr=args.sr, bitdepth=args.bitdepth, rnd=rnd)

    print("Done.")

if __name__ == '__main__':
    main()