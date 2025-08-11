# pip install pyroomacoustics soundfile numpy
import numpy as np
import soundfile as sf
import pyroomacoustics as pra

fs = 44100

# —— 先建一个“中殿”shoebox（80×30×30 m），RT60 先设 6.5 s 做起点 ——
room_dim = np.array([80.0, 30.0, 30.0])  # [x(西→东), y(南→北), z]
rt60 = 8.29     # at 250 Hz

absorption, max_order = pra.inverse_sabine(rt60, room_dim)
# absorption, max_order = 0.1, 150

room = pra.ShoeBox(
    room_dim,
    fs=fs,
    materials=pra.Material(absorption),
    max_order=max_order,
    air_absorption=True,
)

# —— 讲坛与听众位置（可按你需要再微调）——
pulpit   = np.array([65.0,  5.0, 1.6])   # 讲坛靠东端、南侧，口部高度
listener = np.array([40.0, 15.0, 1.6])   # 观众席中部

room.add_source(pulpit)

# —— 计算“面向讲坛”的方位角，并做 XY（±45°）——
forward = pulpit - listener
forward[2] = 0.0
az = np.arctan2(forward[1], forward[0])       # 弧度
left_az, right_az = az + np.deg2rad(45), az - np.deg2rad(45)

# —— 用 DirectionVector + Cardioid 正确初始化指向性（在 XY 平面：colatitude=90°）——
from pyroomacoustics.directivities import Cardioid, DirectionVector

dir_L = Cardioid(DirectionVector(azimuth=left_az,  colatitude=0, degrees=False), gain=1.0)
dir_R = Cardioid(DirectionVector(azimuth=right_az, colatitude=0, degrees=False), gain=1.0)

# —— 把两只麦放在同一点（经典 XY），也可以在 x 方向分开 1~2cm 做“接近-XY”——
mic_locs = np.c_[listener, listener]  # shape (3, 2)
room.add_microphone_array(mic_locs, directivity=[dir_L, dir_R])

# —— 计算并保存 RIR（float32 WAV；顺带导出立体声版本）——
room.compute_rir()
rir_L = np.asarray(room.rir[0][0], dtype=np.float32)
rir_R = np.asarray(room.rir[1][0], dtype=np.float32)

# 轻微归一化，防止极端峰值
peak = max(np.max(np.abs(rir_L)), np.max(np.abs(rir_R)))
if peak > 0:
    scale = 0.99 / peak
    rir_L *= scale; rir_R *= scale

sf.write("york_minster_nave_XY_L.wav", rir_L, fs, subtype="FLOAT")
sf.write("york_minster_nave_XY_R.wav", rir_R, fs, subtype="FLOAT")

N = max(len(rir_L), len(rir_R))
stereo = np.stack([np.pad(rir_L, (0, N - len(rir_L))),
                   np.pad(rir_R, (0, N - len(rir_R)))], axis=1).astype(np.float32)
sf.write("york_minster_nave_XY_stereo.wav", stereo, fs, subtype="FLOAT")

print(f"Absorption={absorption:.4f}, max_order={max_order}, RIR length ≈ {N/fs:.2f}s")
