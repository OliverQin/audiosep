import numpy as np
import math
import time


def random_long_tail(
    sr = 44100,
    attack_duration_ms = 0.8, # Attack duration in milliseconds
    rt60 = 1.0, # Reverberation time in seconds
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

    return sound.astype(np.float32)

def synth_balloon_pop(
    sr=44100,
    duration=0.7,
    seed=None,
    brightness=0.9,      # 0..1  起爆瞬间的高频亮度
    boom=0.35,           # 0..1  低频“气浪/咚”的量感
    crackle_density=900, # 前80ms每秒的点击期望次数
    stereo=False
):
    """Return mono (N,) or stereo (N,2) float32 waveform in [-1,1]."""
    rng = np.random.default_rng(seed)
    N = int(duration * sr)
    t = np.arange(N, dtype=np.float64) / sr

    # 1) 宽带噪声 + 多级指数衰减 + 极快起音
    attack = 1.0 - np.exp(-t / 0.0008)  # ~0.8ms
    env = (np.exp(-t/0.020) + 0.45*np.exp(-t/0.090) + 0.15*np.exp(-t/0.350)) * attack
    # burst = rng.standard_normal(N) * env
    burst = rng.uniform(-1.0, 1.0, size=N)
    for _i in range(2):
        burst += rng.uniform(-1.0, 1.0, size=N)
    burst = (burst / 3) * env
    burst[env.argmax()] = 1.0

    # 2) 时间变化低通：很亮开头 -> 很快变暗
    fc0 = 18000.0 * (0.6 + 0.4*brightness)   # 起始截止
    fc1 = 1200.0 + (1.0 - brightness)*800.0  # 结束截止
    print(fc0, fc1)
    fc = fc1 + (fc0 - fc1) * np.exp(-t / 0.060)
    print(fc[:10], fc[-10:])
    y = np.zeros(N, dtype=np.float64)
    aa = np.exp(-2.0*np.pi*fc/sr) 
    print(aa[:5], aa[-5:])
    for n in range(N):
        a = math.exp(-2.0*math.pi*fc[n]/sr)         # 一阶LP系数
        y[n] = (a*y[n-1] if n else 0.0) + (1-a)*burst[n]
        # y[n] = env[n]
        # y[n] = burst[n]

    # 3) 橡胶裂纹：前80ms稀疏点击
    # crack_win = (t < 0.080)
    # p = crackle_density / sr
    # impulses = (rng.random(N) < p) & crack_win
    # if impulses.any():
    #     imp = np.zeros(N, dtype=np.float64)
    #     idx = np.nonzero(impulses)[0]
    #     amp = rng.uniform(0.4, 1.0, size=idx.size) * (1.0 - (idx/sr)/0.08)
    #     imp[idx] = amp
    #     k = np.array([1.0,-0.85,0.65,-0.50,0.38,-0.28,0.20], dtype=np.float64)
    #     clicks = np.convolve(imp, k, mode="same") * np.exp(-t/0.015)
    #     y += clicks

    # 4) 气浪“咚”和若干房间短共振
    # f_boom = rng.uniform(80.0, 140.0)
    # y += boom*0.35*np.sin(2*np.pi*f_boom*t + rng.uniform(0,2*np.pi)) * np.exp(-t/0.090)
    # for _ in range(3 + rng.integers(0, 3)):              # 3~5个模态
    #     f   = 200.0 * 2.0**rng.uniform(0.5, 3.8)         # ~283Hz..3.5kHz
    #     tau = rng.uniform(0.020, 0.120)
    #     amp = rng.uniform(0.04, 0.12)
    #     ph  = rng.uniform(0, 2*np.pi)
    #     y  += amp*np.sin(2*np.pi*f*t + ph) * np.exp(-t/tau)

    # 5) DC阻塞（1阶高通）
    # R = math.exp(-2.0*math.pi*20.0/sr)
    # y_hp = np.zeros_like(y)
    # x_prev = 0.0
    # for n in range(N):
    #     xn = y[n]
    #     y_hp[n] = xn - x_prev + R*(y_hp[n-1] if n else 0.0)
    #     x_prev = xn
    # y = y_hp

    # 6) 软削波 + 归一化
    # y = np.tanh(2.2*y)
    y *= 0.98 / (np.max(np.abs(y)) + 1e-12)

    if stereo:
        # 细微左右差异：0.7ms 延迟 + 很小早期反射
        d = int(0.0007*sr)
        r = np.zeros_like(y); r[d:] = y[:-d]
        refl = int(0.004*sr)
        if refl < N: r[refl:] += 0.15*y[:-refl]*np.exp(-t[:N-refl]/0.020)
        y = np.stack([y, r], axis=-1)

    return y.astype(np.float32)

sr = 44100
# y = synth_balloon_pop(sr=sr, duration=1.0, seed=42, stereo=False)
y = random_long_tail(sr=sr, attack_duration_ms=0.8, rt60=5.0)

# 保存（需要 soundfile）
# import soundfile as sf; sf.write("balloon_pop.wav", y, sr)

# 或用标准库写 16-bit PCM：
import wave, numpy as np
data = (np.clip(y, -1, 1)*32767).astype(np.int16)
with wave.open("balloon_pop_lo.wav","wb") as wf:
    wf.setnchannels(2 if y.ndim==2 else 1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(data.tobytes())
