#!/usr/bin/env python3

import os
import wave
import math
import random
import struct
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

# ----------------------
# Helpers to parse WAV header & memmap PCM
# ----------------------
def _find_data_chunk(fp) -> Tuple[int, int]:
    """
    Return (data_offset, data_size_bytes) of the 'data' chunk in a WAV file.

    Minimal RIFF parser (little-endian).
    """
    fp.seek(0)
    header = fp.read(12)
    if len(header) < 12 or header[0:4] != b'RIFF' or header[8:12] != b'WAVE':
        raise ValueError("Not a valid RIFF/WAVE file")
    # Iterate chunks
    while True:
        chunk = fp.read(8)
        if len(chunk) < 8:
            raise ValueError("Reached EOF without finding data chunk")
        chunk_id, chunk_size = struct.unpack('<4sI', chunk)
        if chunk_id == b'data':
            data_offset = fp.tell()
            return data_offset, chunk_size
        # Skip this chunk
        fp.seek(chunk_size, 1)

def _check_and_memmap_wav(path: Path,
                          expect_sr: int = 44100,
                          expect_channels: int = 2,
                          expect_width: int = 2) -> Tuple[np.memmap, int, int]:
    """
    Validate WAV format and return memmap + num_samples + num_channels.
    Assumes PCM S16LE.
    """
    with wave.open(str(path), 'rb') as wf:
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        comptype = wf.getcomptype()

    if fr != expect_sr:
        raise ValueError(f"{path} sample rate {fr} != {expect_sr}")
    if nch != expect_channels:
        raise ValueError(f"{path} channels {nch} != {expect_channels}")
    if sampwidth != expect_width:
        raise ValueError(f"{path} sample width {sampwidth} != {expect_width} bytes")
    if comptype != 'NONE':
        raise ValueError(f"{path} is not PCM (compression type {comptype})")

    # locate data chunk for memmap
    with open(path, 'rb') as fp:
        data_offset, data_size = _find_data_chunk(fp)

    # total frames double-check
    bytes_per_frame = expect_channels * expect_width
    assert data_size == nframes * bytes_per_frame, "Data size/frame mismatch!"

    # memmap stereo int16
    mm = np.memmap(path, mode='r', dtype='<i2', offset=data_offset, shape=(nframes * expect_channels,))
    # reshape to (frames, channels)
    mm = mm.reshape(-1, expect_channels)
    return mm, nframes, nch

def _gather_wavs(dirs: List[Path]) -> List[Path]:
    files = []
    for d in dirs:
        files.extend(sorted(d.glob("*.wav")))
    return files


# ----------------------
# Simple Augmenter
# ----------------------
class Augmenter:
    """
    Lightweight real-time augmentations.
    You can extend/disable by config flags.
    """
    def __init__(self,
                 gain_db_range: Tuple[float, float] = (-3.0, 3.0),
                 swap_channels_prob: float = 0.1,
                 eq_prob: float = 0.2,
                 eq_gain_db: float = 2.0,
                 pitch_semitone_range: Tuple[float, float] = None,
                 rnd_gen: Optional[random.Random] = None):
        self.gain_db_range = gain_db_range
        self.swap_channels_prob = swap_channels_prob
        self.eq_prob = eq_prob
        self.eq_gain_db = eq_gain_db
        self.rnd = rnd_gen or random.Random()
        self.pitch_semitone_range = pitch_semitone_range

    def seed(self, x):
        self.rnd.seed(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, T) float32
        # Gain
        if self.gain_db_range is not None:
            db = self.rnd.uniform(*self.gain_db_range)
            x = x * (10.0 ** (db / 20.0))

        # Swap channels
        if x.shape[0] == 2 and self.rnd.random() < self.swap_channels_prob:
            x = x[[1, 0], :]

        # Pitch shift
        if self.pitch_semitone_range is not None:
            n_steps = self.rnd.uniform(*self.pitch_semitone_range)
            if abs(n_steps) > 1e-2:  
                x = torchaudio.functional.pitch_shift(
                    waveform=x,
                    sample_rate=44100,
                    n_steps=n_steps
                )

        # Very light EQ: random first-order high-shelf or low-shelf (cheap approximation)
        if self.rnd.random() < self.eq_prob:
            # choose high or low shelf
            is_high = self.rnd.random() < 0.5
            gain = 10 ** (self.eq_gain_db / 20.0)
            # naive freq domain slope via linear fade
            T = x.shape[1]
            freqs = torch.linspace(0, 1.0, T, device=x.device)
            mask = freqs if is_high else (1.0 - freqs)
            mask = 1.0 + (gain - 1.0) * mask  # linear tilt
            x = x * mask.unsqueeze(0)

        return x


# ----------------------
# Dataset
# ----------------------
class BaroqueNoiseDataset(Dataset):
    def __init__(self,
                 music_dirs: List[str],
                 noise_dirs: List[str],
                 rirs_path: str or None,
                 seg_len: int,
                 mode: str = 'train',
                 seed: Optional[int] = None,
                 sr: int = 44100,
                 channels: int = 2,
                 sample_width_bytes: int = 2,
                 music_augment: Optional[Augmenter] = None,
                 noise_augment: Optional[Augmenter] = None):
        """
        mode: 'train' or 'eval'
        """
        assert mode in ('train', 'eval')
        self.mode = mode

        if seed is None:
            seed = 42 if mode == 'eval' else random.randrange(1 << 31)
        self.seed = seed
        self.rnd = random.Random(seed)

        self.sr = sr
        self.seg_len = seg_len
        self.channels = channels

        self.rirs_path = rirs_path
        self._load_rirs()

        # collect files
        self.music_paths = _gather_wavs([Path(p) for p in music_dirs])
        self.noise_paths = _gather_wavs([Path(p) for p in noise_dirs])
        if len(self.music_paths) == 0:
            raise ValueError("No music wav found")
        if len(self.noise_paths) == 0:
            raise ValueError("No noise wav found")

        # memmap + lengths
        self.music_db = []
        self.noise_db = []

        total_music_frames = 0
        for p in self.music_paths:
            mm, nframes, nch = _check_and_memmap_wav(p, sr, channels, sample_width_bytes)
            self.music_db.append({"path": p, "mm": mm, "frames": nframes})
            total_music_frames += max(0, nframes - self.seg_len)

        total_noise_frames = 0
        for p in self.noise_paths:
            mm, nframes, nch = _check_and_memmap_wav(p, sr, channels, sample_width_bytes)
            self.noise_db.append({"path": p, "mm": mm, "frames": nframes})
            total_noise_frames += max(0, nframes - self.seg_len)

        if total_music_frames <= 0:
            raise ValueError("Music files too short for given segment length")
        if total_noise_frames <= 0:
            raise ValueError("Noise files too short for given segment length")

        self.total_music_frames = total_music_frames
        self.total_music_seconds = total_music_frames / sr
        # length of dataset (epoch definition)
        self._length = int(math.floor(self.total_music_seconds / (self.seg_len / sr)))

        # build cumulative weights for picking file proportional to length
        self.music_weights = np.array([max(0, d["frames"] - self.seg_len) for d in self.music_db], dtype=np.float64)
        self.music_weights /= self.music_weights.sum()

        self.noise_weights = np.array([max(0, d["frames"] - self.seg_len) for d in self.noise_db], dtype=np.float64)
        self.noise_weights /= self.noise_weights.sum()

        self.music_cumsum_weights = np.cumsum(self.music_weights)
        self.noise_cumsum_weights = np.cumsum(self.noise_weights)

        # augmenters (can be None)
        self.music_aug = music_augment
        self.noise_aug = noise_augment
    
    def _load_rirs(self):
        self.rirs = []
        if not self.rirs_path:
            return

        wav_paths = sorted(Path(self.rirs_path).glob("*.wav"))
        for path in wav_paths:
            data, _ = sf.read(path)
            arr = data.T.astype(np.float32)
            if arr.shape[1] >= self.seg_len:
                arr = arr[:, :self.seg_len]
            else:
                arr = np.pad(rir, [[0, 0], [0, self.seg_len - arr.shape[1]]])
            assert tuple(arr.shape) == (2, self.seg_len)

            self.rirs.append([np.fft.fft(arr[0]), np.fft.fft(arr[1])])
        return

    def __len__(self):
        if self.mode == 'eval':
            return self._length // 16
        return self._length

    def _pick_segment(self, db_list, cumsum_weights) -> torch.Tensor:
        # pick file index by weights
        idx = self._weighted_choice(cumsum_weights)
        entry = db_list[idx]
        frames = entry["frames"]
        if frames <= self.seg_len:
            start = 0
        else:
            start = self.rnd.randint(0, frames - self.seg_len - 1)
        end = start + self.seg_len
        # slice memmap -> numpy -> torch float
        seg = entry["mm"][start:end, :]  # shape (T, C)
        seg = seg.astype(np.float32) * (1.0 / 32768.0)
        seg = seg.T

        # Random RIRS
        if self.rirs:
            idx = self.rnd.randrange(0, len(self.rirs))

            for ch in range(2):
                maxv = np.abs(seg[ch]).max()
                this_chn = np.real(np.fft.ifft(np.fft.fft(seg[ch]) * self.rirs[idx][ch]))
                after_maxv = np.abs(this_chn).max()
                seg[ch] = this_chn * (maxv / after_maxv)

        # to torch
        seg = torch.from_numpy(seg)
        return seg

    def _weighted_choice(self, cumsum_weights: np.ndarray) -> int:
        # random.Random has random() in [0,1)
        x = self.rnd.random()
        return int(np.searchsorted(cumsum_weights, x, side='right'))

    def reset_seed(self):
        if self.mode == 'eval':
            self.seed = 42
            self.rnd = random.Random(self.seed)
        else:
            self.seed = random.SystemRandom().randrange(2**31)
            self.rnd = random.Random(self.seed)

    def __getitem__(self, index):
        self.rnd.seed(str([self.seed, index]))

        # music
        music = self._pick_segment(self.music_db, self.music_cumsum_weights)
        if self.music_aug is not None:
            self.music_aug.seed(str([self.seed, index, 'music']))
            music = self.music_aug(music)

        # noise
        noise = self._pick_segment(self.noise_db, self.noise_cumsum_weights)
        if self.noise_aug is not None:
            self.noise_aug.seed(str([self.seed, index, 'noise']))
            noise = self.noise_aug(noise)

        # You can also return mix here if you want:
        # Combine with random SNR each call
        snr_db = self.rnd.uniform(-10, 10)  # example range
        # scale noise to achieve SNR
        music_power = music.pow(2).mean()
        noise_power = noise.pow(2).mean() + 1e-12
        target_noise_power = music_power / (10 ** (snr_db / 10))
        scale = (target_noise_power / noise_power).sqrt()
        noise_scaled = noise * scale
        mix = music + noise_scaled

        return {
            "mix": mix,               # (C,T)
            "music": music,           # (C,T)
            "noise": noise_scaled,    # scaled version
            "snr_db": torch.tensor(snr_db, dtype=torch.float32)
        }
