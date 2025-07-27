#!/usr/bin/env python3

import math

import soundfile as sf
import numpy as np
import pyroomacoustics as pra

def sample_log_uniform(low, high):
    return math.exp(np.random.uniform(math.log(low), math.log(high)))

def generate_random_rir(
    fs=44100,
    rir_length=2**19,
    room_x_range=(3, 60),
    room_y_range=(2, 40),
    room_h_range=(3, 6),
    absorption_range=(0.2, 0.9),
    mic_xy_angle_range=(50, 130),
    mic_distance_range=(0, 0.02),
):
    # 1. Random geometrics of the room (x, y, h)
    room_dims = [
        sample_log_uniform(*room_x_range),
        sample_log_uniform(*room_y_range),
        sample_log_uniform(*room_h_range)
    ]
    
    # 2. Random absorption rate
    abs_coef = np.random.uniform(*absorption_range)
    materials = pra.Material(energy_absorption=abs_coef)
    
    # 3. ShoeBox room
    room = pra.ShoeBox(
        p=room_dims,
        fs=fs,
        materials=materials,
        max_order=15,            # Max times of reflection
        air_absorption=True
    )
    
    # 4. Source should be >=0.5m to the wall
    src = np.array([
        np.random.uniform(0.5, room_dims[0] - 0.5),
        np.random.uniform(0.5, room_dims[1] - 0.5),
        np.random.uniform(0.5, room_dims[2] - 0.5)
    ])
    room.add_source(src.tolist())
    
    # 5. Random microphone pos
    mic_center = np.array([
        np.random.uniform(0.5, room_dims[0] - 0.5),
        np.random.uniform(0.5, room_dims[1] - 0.5),
        np.random.uniform(0.5, room_dims[2] - 0.5)
    ])
    
    # XY Microphone: almost the same position, pointing to two directions
    mic_angle_half = np.random.uniform(*mic_xy_angle_range) / 180. * np.pi / 2.

    vec = src - mic_center
    az = math.atan2(vec[1], vec[0])

    mic_distance = np.random.uniform(*mic_distance_range)
    offset1 = mic_distance * np.array([math.cos(az + mic_angle_half), math.sin(az + mic_angle_half), 0.])
    offset2 = mic_distance * np.array([math.cos(az - mic_angle_half), math.sin(az - mic_angle_half), 0.])
    mics = np.c_[ (mic_center + offset1), (mic_center + offset2) ]
    
    mic_array = pra.MicrophoneArray(mics, fs)
    room.add_microphone_array(mic_array)
    
    # 6. Compute RIR
    room.compute_rir()

    # room.rir -> list of list: shape (n_mics, n_sources)
    rirs = []
    for mic_idx in range(len(room.rir)):
        rir = np.array(room.rir[mic_idx][0])
        if len(rir) < rir_length:
            rir = np.pad(rir, (0, rir_length - len(rir)))
        else:
            rir = rir[:rir_length]
        rirs.append(rir)
    
    # normalize
    output = np.vstack(rirs)

    for ch in range(2):
        v_sum = output[ch].sum()
        output[ch] /= np.abs(v_sum)

    return output 

if __name__ == '__main__':
    np.random.seed(31415926)

    for i in range(128):
        a = generate_random_rir().T
        sf.write(f"rirs/{i:06d}.wav", a, 44100, subtype="FLOAT")

