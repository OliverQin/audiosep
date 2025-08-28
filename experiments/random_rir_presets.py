#!/usr/bin/env python3

import math

import soundfile as sf
import numpy as np
import pyroomacoustics as pra

from pyroomacoustics.directivities.analytic import Cardioid
from pyroomacoustics.directivities.direction import DirectionVector

def sample_log_uniform(low, high):
    return math.exp(np.random.uniform(math.log(low), math.log(high)))

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector encountered when computing orientation.")
    return v / n

def _vec_to_azimuth_colatitude(v):
    v = _normalize(v)
    x, y, z = v
    az = np.arctan2(y, x)
    col = np.arccos(z)
    return az, col

def _rodrigues_rotate(vec, axis, angle_rad):
    """Rotate vec around axis by angle_rad using Rodrigues' rotation formula."""
    v = np.asarray(vec, dtype=float)
    k = _normalize(axis)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return v * c + np.cross(k, v) * s + k * (np.dot(k, v)) * (1 - c)

def generate_random_rir(
    fs=44100,
    rir_length=2**21,
    room_xy_range=(3, 60),
    room_h_range=(2.8, 3.2),
    speaker_wall_distance=0.5,
    speaker_height_range=(1.0, 2.5),
    absorption_range=(0.05, 0.99),
    scattering_range=(0.02, 0.2),
    mic_xy_angle_range=(85, 95),
    mic_speaker_distance_range=(0.1, 1.5),
):
    # 1. Random geometrics of the room (x, y, h)
    room_x = sample_log_uniform(*room_xy_range)
    room_y = sample_log_uniform(*room_xy_range)
    room_height = sample_log_uniform(*room_h_range)
    room_dims = [
        room_x,
        room_y,
        room_height
    ]
    
    # 2. Random absorption rate
    abs_coef = sample_log_uniform(*absorption_range)
    sctr_coef = sample_log_uniform(*scattering_range)
    materials = pra.Material(energy_absorption=abs_coef, scattering=sctr_coef)
    
    # 3. ShoeBox room
    room = pra.ShoeBox(
        p=room_dims,
        fs=fs,
        materials=materials,
        max_order=180,            # Max times of reflection
        air_absorption=True,
        ray_tracing=False
    )
    
    # 4. Source should be >=speaker_wall_distance to the wall
    src = np.array([
        np.random.uniform(speaker_wall_distance, room_dims[0] - speaker_wall_distance),
        np.random.uniform(speaker_wall_distance, room_dims[1] - speaker_wall_distance),
        np.random.uniform(*speaker_height_range),
    ])
    room.add_source(src.tolist())
    
    # 5. Random microphone pos
    mic_center = np.array([0., 0., 0.])
    while ((mic_center - np.array([0, 0, 0])) <= 0).any() or ((mic_center - np.array(room_dims)) >= 0).any():
        # Random direction
        mic_dir = np.random.normal(size=(3,))
        mic_dir /= np.linalg.norm(mic_dir)
        mic_dir *= np.random.uniform(*mic_speaker_distance_range)
        
        mic_center = mic_dir + src
        
    # XY Microphone: the same position, pointing to two directions
    mic_angle_half = np.random.uniform(*mic_xy_angle_range) / 180. * np.pi / 2.

    up0 = np.array([0.0, 1.0, 0.0])
    fwd = _normalize(src - mic_center)
    right = _normalize(np.cross(fwd, up0))
    up = _normalize(np.cross(right, fwd))

    dir1 = _rodrigues_rotate(fwd, up, +mic_angle_half)
    dir2 = _rodrigues_rotate(fwd, up, -mic_angle_half)
    az1, col1 = _vec_to_azimuth_colatitude(dir1)
    az2, col2 = _vec_to_azimuth_colatitude(dir2)

    mic1 = Cardioid(
        orientation=DirectionVector(azimuth=az1, colatitude=col1, degrees=False),
    )
    mic2 = Cardioid(
        orientation=DirectionVector(azimuth=az2, colatitude=col2, degrees=False),
    )

    mics = np.tile(mic_center.reshape(3, 1), (1, 2))
    
    mic_array = pra.MicrophoneArray(mics, fs, directivity=[mic1, mic2])
    room.add_microphone_array(mic_array)
    
    # import matplotlib
    # import matplotlib.pyplot
    # fig, ax = room.plot()
    # matplotlib.pyplot.show()

    # 6. Compute RIR
    room.compute_rir()

    # room.rir -> list of list: shape (n_mics, n_sources)
    rirs = []
    for mic_idx in range(len(room.rir)):
        rir = np.array(room.rir[mic_idx][0])
        rirs.append(rir[:rir_length])

    max_len = max(len(r) for r in rirs)
    rirs = [np.pad(r, (0, max_len - len(r))) for r in rirs]
    
    # normalize
    output = np.vstack(rirs)
    
    return output 

if __name__ == '__main__':
    np.random.seed(31415926)

    for i in range(20):
        a = generate_random_rir().T
        sf.write(f"newrirs/{i:06d}.wav", a, 44100, subtype="FLOAT")

