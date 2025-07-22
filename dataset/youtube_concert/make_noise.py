#!/usr/bin/env python3

import os
import numpy as np
import tqdm

from numpy.lib.stride_tricks import sliding_window_view
from pydub import AudioSegment

def parse_time_string_to_ms(t: str) -> int:
    """
    Parse a time string into milliseconds.
    - "SS"       -> seconds
    - "MM:SS"    -> minutes:seconds
    - "HH:MM:SS" -> hours:minutes:seconds
    """
    parts = t.strip().split(':')
    if len(parts) == 1:
        seconds = int(parts[0])
        return seconds * 1000
    elif len(parts) == 2:
        minutes, seconds = parts
        return (int(minutes) * 60 + int(seconds)) * 1000
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000
    else:
        raise ValueError(f"Invalid time format: {t}")

def fft_convolve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Perform 1D circular convolution of x and y using FFT, without zero-padding.
    Treats signals as periodic (wrap-around at boundaries).

    If x and y have different lengths, both are wrapped/truncated to length n = max(len(x), len(y)).
    Returns real-valued result of length n.

    Parameters:
        x (np.ndarray): First input array.
        y (np.ndarray): Second input array.

    Returns:
        np.ndarray: Circular convolution result, real-valued of length n.
    """
    # Determine circular length
    n = max(x.shape[0], y.shape[0])

    # Wrap or truncate x
    if x.shape[0] < n:
        x = np.pad(x, (0, n - x.shape[0]), mode='constant', constant_values=0)

    # Wrap or truncate y
    if y.shape[0] < n:
        y = np.pad(y, (0, n - y.shape[0]), mode='constant', constant_values=0)

    # FFT of both signals
    X = np.fft.fft(x)
    Y = np.fft.fft(y)

    # Pointwise multiply in frequency domain and inverse FFT
    result = np.fft.ifft(X * Y)

    # Discard negligible imaginary parts and return real result
    return np.real(result)

def smooth_normalize(audio: AudioSegment, window_ms: int = 2000, target_level: float = 0.5, noise_level: float = 0.1) -> AudioSegment:
    """
    Smoothly normalize audio using envelope smoothing:
    1. Compute absolute-value envelope of signal
    2. Convolve with a rectangular window of length `window_ms`
    3. Scale each sample so that the envelope maps to `target_level`
    """
    # Extract samples as numpy array
    samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        # reshape to (channels, n_samples)
        samples = samples.reshape((-1, audio.channels)).T
    else:
        samples = samples[np.newaxis, :]

    # Compute mono envelope
    env = np.mean(np.abs(samples), axis=0)
    
    # Don't scale too much: when signal is below 60 (for 16-bit data) it is not reliable
    BLACK_LEVEL = 60
    env = np.clip(env, BLACK_LEVEL, None)

    # Convert window length from ms to samples
    window_samples = int(window_ms * audio.frame_rate / 1000)
    if window_samples < 1:
        window_samples = 1

    # Rectangular window
    window = np.ones(window_samples) / window_samples
    
    # Smooth envelope
    conv_out = fft_convolve((env * 1e-5) ** 8, window)
    conv_out = np.clip(conv_out, 0.0, None)
    smooth_env = conv_out ** (1.0 / 8) * 1e5
    smooth_env = np.clip(smooth_env, BLACK_LEVEL, env.max())

    # Determine target peak in absolute units
    max_possible = float(2 ** (8 * audio.sample_width - 1) - 1)
    target_peak = max_possible * target_level

    # Apply gain to each channel
    processed = np.empty_like(samples, dtype=smooth_env.dtype)
    for ch in range(samples.shape[0]):
        processed[ch] = samples[ch] / smooth_env

    # Scale to target
    processed *= (target_peak / np.abs(processed).max())

    # Sampling noise
    noise = np.random.randn(*processed.shape)
    noise *= np.abs(processed) * (noise_level)
    processed += noise

    # Heat noise
    noise2 = np.random.randn(*processed.shape) * (noise_level * BLACK_LEVEL)
    processed += noise2

    # Reconstruct interleaved array
    if audio.channels > 1:
        interleaved = processed.T.flatten()
    else:
        interleaved = processed.flatten()

    # Clip to valid range and convert to int
    interleaved = np.clip(interleaved, -max_possible, max_possible).astype(np.int16)

    # Create new AudioSegment
    return AudioSegment(
        interleaved.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )


def process_video_ranges(input_video: str, ranges_str: str, output_flac: str):
    """
    1. Load audio from MP4
    2. Extract segments by comma-separated ranges
    3. Crossfade adjacent segments by 1 second
    4. Smoothly normalize with sliding-window envelope
    5. Export as 16-bit FLAC
    """
    # Load audio
    audio = AudioSegment.from_file(input_video, format="mp4")

    # Extract segments
    segments = []
    for part in ranges_str.split(','):
        start_str, end_str = part.split('-', 1)
        start_ms = parse_time_string_to_ms(start_str)
        end_ms = parse_time_string_to_ms(end_str)
        assert end_ms > start_ms
        segments.append(audio[start_ms:end_ms])

    # Combine with 1s crossfade
    crossfade_ms = 1000
    combined = segments[0]
    for seg in segments[1:]:
        combined = combined.append(seg, crossfade=crossfade_ms)

    # Smooth normalization
    combined = smooth_normalize(combined, window_ms=2000, target_level=0.5)

    # Ensure 16-bit sample width
    combined = combined.set_sample_width(2)

    # Export FLAC
    combined.export(output_flac, format="flac")


if __name__ == "__main__":
    filelist = [
        ("Vivaldis Flautino Concerto in C Major RV 443 Lucie Horsch.mp4", "8-21,4:5-4:19,8:14-8:19,11:02-12:54,15:30-16:19"),
        ("Vivaldi_ The Four Seasons (Orquesta Reino de Aragón).mp4", "0-3,3:27-3:45,6:15-6:26,10:12-10:29,15:56-16:17,21:00-21:30,26:32-26:50,29:31-29:38,32:55-33:15,36:41-37:02,39:12-39:17,42:48-43:20"),
        ("Vivaldi_ Four Seasons_Quattro Stagioni - Janine Jansen - Internationaal Kamermuziek Festival.mp4", "0-5,3:19-3:30,5:51-6:3,9:55-10:23,20:27-21:02,25:33-25:36,25:57-26:11,28:31-28:42,31:47-32:06,35:10-35:22,36:55-37:00,40:07-43:54"),
        ("Vivaldi Winter.mp4","0-5,3:38-3:41,5:52-5:55,9:17-9:26"),
        ("Vivaldi - Gloria (RV 589).mp4","0-5,2:31-2:37,5:38-5:50,8:18-8:28,9:49-10:03,13:35-13:43,16:04-16:08,16:15-16:20,20:08-20:14,21:17-21:23,23:40-23:45,24:33-24:36"),
        ("Symphony No. 2 _ Sergei Rachmaninoff _ Vasily Petrenko _ Oslo Philharmonic.mp4","0-25,19:39-20:09,30:11-30:42,59:03-60:00"),
        ("Schubert_ Symphony in C major .mp4", "0-41,13:51-14:11,23:06-23:16,28:44-29:00,44:20-44:30,56:33-59:28"),
        ("MAHLER Symphony No. 2 Resurrection Lan Shui.mp4", "0-56,23:16-23:59,33:45-34:05,1:27:31-1:38:02"),
        ("Handel's Messiah Live from the Sydney Opera House.mp4", "0-1:32,21:36-21:39,30:04-30:06,38:20-38:24,39:06-39:08,45:41-1:06:52,1:07:32-1:08:10,2:30:11-2:32:40"),
    ]
    pbar = enumerate(tqdm.tqdm(filelist))
    for idx, (fn, ranges) in pbar:
        output_file = f"normalized/{idx:06d}.flac"
        process_video_ranges(os.path.join('raw', fn), ranges, output_file)
        # print(f"Exported {output_file}")
