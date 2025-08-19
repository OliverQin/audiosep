#!/usr/bin/env python3

import os
import wave
import math
import random
import struct
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import scipy.signal as sps
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
# Dataset
# ----------------------
class BaroqueNoiseDataset(Dataset):
    def __init__(self,
                 music_dirs: List[str],
                 noise_dirs: List[str],
                 rirs_path: str or None,
                 seg_len: int,
                 mouth_noise_dirs: List[str] = [],
                 mouth_noise_prob: float = 0.15,
                 mode: str = 'train',
                 sr: int = 44100,
                 channels: int = 2,
                 swap_channels_prob: float = 0.5,
                 speed_change_range: Tuple[float, float] or None = (0.9, 1.1),
                 sample_width_bytes: int = 2):
        """
        mode: 'train' or 'eval'
        """
        assert mode in ('train', 'eval')
        self.mode = mode

        if mode == 'eval':
            self.salt = 42
        else:
            self.salt = random.SystemRandom().randrange(1 << 63)
        self.rnd = np.random.Generator(np.random.PCG64(self.salt))

        self.sr = sr
        self.seg_len = seg_len
        self.channels = channels
        self.valid_freqs = []

        self.rirs_path = rirs_path
        self._load_rirs()

        # collect files
        self.music_paths = _gather_wavs([Path(p) for p in music_dirs])
        self.noise_paths = _gather_wavs([Path(p) for p in noise_dirs])
        if len(self.music_paths) == 0:
            raise ValueError("No music wav found")
        if len(self.noise_paths) == 0:
            raise ValueError("No noise wav found")

        # mouth noise files
        self.mouth_noise_paths = _gather_wavs([Path(p) for p in mouth_noise_dirs])
        self.mouth_noise_prob = mouth_noise_prob

        # memmap + lengths
        self.music_db = []
        self.noise_db = []

        total_music_frames = 0
        for p in self.music_paths:
            mm, nframes, _nch = _check_and_memmap_wav(p, sr, channels, sample_width_bytes)
            self.music_db.append({"path": p, "mm": mm, "frames": nframes})
            total_music_frames += max(0, nframes - self.seg_len)

        total_noise_frames = 0
        for p in self.noise_paths:
            mm, nframes, _nch = _check_and_memmap_wav(p, sr, channels, sample_width_bytes)
            self.noise_db.append({"path": p, "mm": mm, "frames": nframes})
            total_noise_frames += max(0, nframes - self.seg_len)

        if total_music_frames <= 0:
            raise ValueError("Music files too short for given segment length")
        if total_noise_frames <= 0:
            raise ValueError("Noise files too short for given segment length")

        self.mouth_noise_db = []
        for p in self.mouth_noise_paths:
            mm, nframes, _nch = _check_and_memmap_wav(p, sr, channels, sample_width_bytes)
            self.mouth_noise_db.append({"path": p, "mm": mm, "frames": nframes})

        self.total_music_frames = total_music_frames
        self.total_music_seconds = total_music_frames / sr

        # length of dataset (epoch definition)
        self._length = int(math.floor(self.total_music_seconds / (self.seg_len / sr)))

        # build cumulative weights for picking file proportional to length
        self.music_weights = np.array([max(0, d["frames"] - self.seg_len) for d in self.music_db], dtype=np.float64)
        self.music_weights /= self.music_weights.sum()
        self.music_cumsum_weights = np.cumsum(self.music_weights)

        self.noise_weights = np.array([max(0, d["frames"] - self.seg_len) for d in self.noise_db], dtype=np.float64)
        self.noise_weights /= self.noise_weights.sum()
        self.noise_cumsum_weights = np.cumsum(self.noise_weights)

        # augmentation parameters
        self.swap_channels_prob = swap_channels_prob
        if speed_change_range:
            self._init_speed_change_freqs(speed_change_range, min_gcd=100)
    
    def _init_speed_change_freqs(self, speed_change_range, min_gcd):
        """
        Initialize valid frequencies for speed changes.
        Args:
            speed_change_range: Tuple of (min_speed, max_speed)
            min_gcd: Minimum required GCD with sample rate
        """
        min_r, max_r = speed_change_range
        min_freq = int(self.sr * min_r)
        max_freq = int(self.sr * max_r)
        
        # Find all valid frequencies where gcd with sr >= min_gcd
        self.valid_freqs = []
        for freq in range(min_freq, max_freq + 1):
            if math.gcd(freq, self.sr) >= min_gcd:
                self.valid_freqs.append(freq)
        
        if not self.valid_freqs:
            raise ValueError(f"No valid frequencies found for speed change range {speed_change_range} "
                             f"with sr={self.sr} and min_gcd={min_gcd}")

    def _load_rirs(self):
        self.rirs = []
        if not self.rirs_path:
            return

        wav_paths = sorted(Path(self.rirs_path).glob("*.wav"))
        for path in wav_paths:
            data, _ = sf.read(path)
            arr = data.T.astype(np.float32)

            krn_len = self.seg_len * 2
            if arr.shape[1] >= krn_len:
                arr = arr[:, :krn_len]
            else:
                arr = np.pad(arr, [[0, 0], [0, krn_len - arr.shape[1]]])
            assert tuple(arr.shape) == (2, self.seg_len * 2)

            # Align RIR to the maximum energy position
            idx = arr.max(axis=0).argmax()
            if idx > 0:
                arr = np.roll(arr, -idx, axis=1)

            lhs = np.fft.fft(arr[0]).astype(np.complex64)
            rhs = np.fft.fft(arr[1]).astype(np.complex64)
            lhs, rhs = torch.from_numpy(lhs), torch.from_numpy(rhs)
            self.rirs.append([lhs, rhs])

    def _pick_rir(self):
        if not self.rirs:
            return None
            
        # Randomly select two RIRs (can be the same)
        idx1 = self.rnd.integers(len(self.rirs))
        idx2 = self.rnd.integers(0, len(self.rirs))
        
        # Random blend factor between 0 and 1
        blend = self.rnd.random()
        
        # Blend the RIRs
        lhs = self.rirs[idx1][0] * blend + self.rirs[idx2][0] * (1 - blend)
        rhs = self.rirs[idx1][1] * blend + self.rirs[idx2][1] * (1 - blend)
        
        return [lhs, rhs]

    def __len__(self):
        if self.mode == 'eval':
            return self._length // 16
        return self._length

    def _pick_mouth_noise(self, length: int) -> np.ndarray or None:
        if not self.mouth_noise_db:
            return None

        # Pick a random mouth noise file
        idx = self.rnd.integers(len(self.mouth_noise_db))
        entry = self.mouth_noise_db[idx]
        frames = entry["frames"]

        # Randomly select many segments of [50, 200) ms
        # And concat them together, with 20 ms crossfade
        # Using hann window for crossfade
        min_len = int(self.sr * 0.05)  # 50 ms
        max_len = int(self.sr * 0.2)   # 200 ms
        head_len = int(self.sr * 0.02)  # 20 ms
        
        ret = np.zeros((length, self.channels), dtype=np.float32)
        weight = np.zeros((length, self.channels), dtype=np.float32)

        noise_length = 0
        while noise_length < length:
            seg_len = self.rnd.integers(min_len, max_len)
            if noise_length - head_len + seg_len > length:
                seg_len = length - noise_length + head_len
            
            start = self.rnd.integers(frames - seg_len)
            
            piece = entry["mm"][start:start + seg_len, :].astype(np.float32) * (1.0 / 32768.0)
            win = sps.windows.hann(seg_len, sym=True).astype(np.float32) + 1e-6

            if noise_length == 0:
                # First segment, no crossfade
                ret[0:seg_len, :] = piece * win[:, None]
                weight[0:seg_len, :] = win[:, None]

                noise_length += seg_len
            else:
                # Crossfade with previous segment
                prev_end = noise_length - head_len
                ret[prev_end:prev_end+seg_len, :] += piece * win[:, None]
                weight[prev_end:prev_end+seg_len, :] += win[:, None]

                noise_length = prev_end + seg_len
        
        return ret / weight

    @torch.no_grad()
    def _pick_segment(self, db_list, cumsum_weights, rir, with_mouth_noise=False) -> torch.Tensor:
        # Pick file index by weights
        idx = self._weighted_choice(cumsum_weights)
        entry = db_list[idx]
        frames = entry["frames"]

        # Calculate required frames for resampling
        orig_freq = self.sr
        if self.valid_freqs and len(self.valid_freqs) > 0:
            orig_freq = self.rnd.choice(self.valid_freqs)
            orig_len = math.ceil(2 * self.seg_len * orig_freq / self.sr)
        else:
            orig_len = 2 * self.seg_len

        # Calculate start position
        if frames <= self.seg_len:
            start = 0
        else:
            start = self.rnd.integers(frames - self.seg_len)
        head = start - orig_len // 2
        tail = head + orig_len

        lpad = 0 if head >= 0 else -head
        rpad = 0 if tail <= frames else tail - frames

        if head < 0:
            head = 0
        if tail > frames:
            tail = frames

        # Slice memmap -> numpy -> torch float
        seg = entry["mm"][head:tail, :]  # shape (T, C)
        seg = seg.astype(np.float32) * (1.0 / 32768.0)
        seg = seg.T
        if lpad > 0 or rpad > 0:
            seg = np.pad(seg, ((0, 0), (lpad, rpad)), mode='constant', constant_values=0)

        seg = torch.from_numpy(seg)  # (C, T)

        # Mouth noise augmentation
        if with_mouth_noise and self.mouth_noise_db:
            noise = self._pick_mouth_noise(seg.shape[1])
            seg_pow_avg = float((seg ** 2).mean().item())
            noise_pow_avg = (noise ** 2).mean() + 1e-12

            target_noise_pow = seg_pow_avg * (10 ** (self.rnd.uniform(-20, -10) / 10))
            scale = (target_noise_pow / noise_pow_avg) ** 0.5
            noise *= scale

            seg += torch.from_numpy(noise).T.to(seg.device)

        # Resample if needed
        if orig_freq != self.sr:
            seg = torchaudio.functional.resample(seg, orig_freq, self.sr)

        # Swap channels if needed
        if self.swap_channels_prob > 0 and self.rnd.random() < self.swap_channels_prob:
            seg = seg.flip(0)

        # Minor rounding to ensure length
        if seg.shape[1] > 2 * self.seg_len:
            seg = seg[:, :2 * self.seg_len]

        if rir:
            maxv = np.abs(seg).max()
            fft_seg = torch.fft.fft(seg, dim=-1)
            fft_seg[0, :] *= rir[0]
            fft_seg[1, :] *= rir[1]
            seg = torch.fft.ifft(fft_seg, dim=-1).real

            after_maxv = np.abs(seg).max() + 1e-7
            seg = seg * (maxv / after_maxv)

        seg = seg[:, -self.seg_len:]
        return seg

    def _weighted_choice(self, cumsum_weights: np.ndarray) -> int:
        # random.Random has random() in [0,1)
        x = self.rnd.random()
        return int(np.searchsorted(cumsum_weights, x, side='right'))

    def reset_seed(self):
        if self.mode == 'eval':
            self.salt = 42
        else:
            self.salt = random.SystemRandom().randrange(1 << 63)
        self.rnd = np.random.Generator(np.random.PCG64(self.salt))

    def __getitem__(self, index):
        salt_item = hash((self.salt, index)) & 0x7fffffff
        self.rnd = np.random.Generator(np.random.PCG64(salt_item))

        # Random RIRS
        rir = self._pick_rir()
        if self.swap_channels_prob > 0 and self.rnd.random() < self.swap_channels_prob:
            rir = [rir[1], rir[0]]

        # Pick segments
        wmn = self.mouth_noise_db and self.rnd.random() < self.mouth_noise_prob
        music = self._pick_segment(self.music_db, self.music_cumsum_weights, rir, with_mouth_noise=wmn)
        noise = self._pick_segment(self.noise_db, self.noise_cumsum_weights, rir)

        # Noise normalization
        snr_db = self.rnd.uniform(-10, 10)  # example range

        music_power = music.pow(2).mean()
        noise_power = noise.pow(2).mean() + 1e-12
        target_noise_power = music_power / (10 ** (snr_db / 10))
        scale = (target_noise_power / noise_power).sqrt()
        noise_scaled = noise * scale
        mix = music + noise_scaled

        # Mix peak normalization
        target_peak = self.rnd.uniform(0.15, 0.95)
        mix_peak = mix.abs().max()
        coeff = target_peak / (mix_peak + 1e-6)

        music *= coeff
        noise_scaled *= coeff
        mix *= coeff

        return {
            "mix": mix,               # (C,T)
            "music": music,           # (C,T)
            "noise": noise_scaled,    # scaled version
            "snr_db": torch.tensor(snr_db, dtype=torch.float32)
        }


if __name__ == "__main__":
    # Example usage
    dataset = BaroqueNoiseDataset(
        music_dirs=["./dataset/flute_recorder_sopranino/wavs"],
        noise_dirs=["./dataset/classical_nowoodwind/wavs"],
        mouth_noise_dirs=["./dataset/mouth_noise"],
        rirs_path='./rirs',
        seg_len=2**19,
        mode='train',
        sr=44100,
        channels=2,
        swap_channels_prob=0.5,
        speed_change_range=(0.9, 1.1),
    )

    assert len(dataset.valid_freqs) == 441

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, drop_last=True)

    import time
    time_start = time.time()
    n_batches = 0
    for batch in loader:
        print(batch['mix'].shape, batch['music'].shape, batch['noise'].shape, batch['snr_db'].shape)
        n_batches += 1
    time_finish = time.time()

    print('Time taken:', time_finish - time_start)
    print('Number of batches:', n_batches)
    print('Average batch time:', (time_finish - time_start) / n_batches)

    
