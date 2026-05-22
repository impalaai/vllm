"""
CVR CAM spectrogram -> playable audio sonification.

This is NOT recovered cockpit audio. It is a synthesized reconstruction built
from a visual reading of the Figure 4 spectrogram screenshot: steady tones,
broadband transient bursts and a low-frequency rumble bed are placed at the
frequencies and timestamps visible in the image. Speech is represented only as
band-limited noise bursts (it cannot be recovered from a magnitude spectrogram).

Window: 12:03:43 -> 12:04:08.5  (t=0 corresponds to 12:03:43)
Output: 16 kHz mono 16-bit PCM WAV.
"""
import wave
import math
import random

SR = 16000
DUR = 25.5
N = int(SR * DUR)
random.seed(42)

buf = [0.0] * N


def add_tone(freq, amp, t0, t1, fade=0.15, vib=0.0):
    """Steady sine over [t0, t1] with raised-cosine fades."""
    i0 = max(int(t0 * SR), 0)
    i1 = min(int(t1 * SR), N)
    fa = max(int(fade * SR), 1)
    for i in range(i0, i1):
        a = amp
        k = i - i0
        if k < fa:
            a *= 0.5 - 0.5 * math.cos(math.pi * k / fa)
        if i1 - i < fa:
            a *= 0.5 - 0.5 * math.cos(math.pi * (i1 - i) / fa)
        t = i / SR
        f = freq * (1.0 + vib * math.sin(2 * math.pi * 5.0 * t))
        buf[i] += a * math.sin(2 * math.pi * f * t)


def _bandpass(x, hp, lp):
    """One-pole low-pass then one-pole high-pass, in place."""
    k = 2 * math.pi * lp / SR
    alp = k / (1 + k)
    y = 0.0
    for i in range(len(x)):
        y += alp * (x[i] - y)
        x[i] = y
    k = 2 * math.pi * hp / SR
    a = 1.0 / (1 + k)
    yp = 0.0
    xp = 0.0
    for i in range(len(x)):
        cur = x[i]
        yp = a * (yp + cur - xp)
        xp = cur
        x[i] = yp
    return x


def add_burst(t0, dur, amp, hp=150, lp=3500, attack=0.015, decay=0.35):
    """Broadband transient: band-limited noise, fast attack, exp decay."""
    i0 = int(t0 * SR)
    n = int(dur * SR)
    x = _bandpass([random.uniform(-1, 1) for _ in range(n)], hp, lp)
    ai = max(int(attack * SR), 1)
    for i in range(n):
        e = amp
        if i < ai:
            e *= i / ai
        else:
            e *= math.exp(-(i - ai) / (decay * SR))
        idx = i0 + i
        if 0 <= idx < N:
            buf[idx] += e * x[i]


def add_callout(t0, amp=0.16):
    """Speech-like callout: a few syllabic band-noise bursts (200-2600 Hz)."""
    syl = [(0.00, 0.18), (0.20, 0.16), (0.40, 0.22)]
    for off, d in syl:
        add_burst(t0 + off, d, amp, hp=200, lp=2600, attack=0.02, decay=0.10)


def add_buzz(t0, dur, f0, amp, nharm=9):
    """Harmonic 'stack' (voiced speech / warbling alert) with vibrato."""
    i0 = int(t0 * SR)
    n = int(dur * SR)
    ai = max(int(0.03 * SR), 1)
    for i in range(n):
        t = i / SR
        f = f0 * (1.0 + 0.03 * math.sin(2 * math.pi * 6.0 * t))
        ph = 2 * math.pi * f * t
        s = sum((1.0 / h) * math.sin(h * ph) for h in range(1, nharm + 1))
        e = amp
        if i < ai:
            e *= i / ai
        else:
            e *= math.exp(-(i - ai) / (0.30 * SR))
        idx = i0 + i
        if 0 <= idx < N:
            buf[idx] += 0.5 * e * s


# --- steady tones --------------------------------------------------------
add_tone(4200, 0.045, 0.0, DUR, fade=0.05)        # constant ~4.2 kHz tone
add_tone(6300, 0.075, 4.6, DUR, fade=0.30)        # "ringing began" ~12:03:47.6

# --- low-frequency rumble bed, growing toward the end --------------------
rumble = _bandpass([random.uniform(-1, 1) for _ in range(N)], hp=35, lp=260)
rumble = _bandpass(rumble, hp=35, lp=260)
for i in range(N):
    t = i / SR
    g = 0.015 + 0.11 * (t / DUR) ** 1.6
    if t > 22.0:
        g += 0.10 * ((t - 22.0) / 3.5)
    buf[i] += g * rumble[i]

# --- crew callouts -------------------------------------------------------
add_callout(2.0)    # "Rotate call"  ~12:03:45
add_callout(5.0)    # "V2 call"      ~12:03:48

# --- harmonic stacks (voiced speech / alert), 12:03:54 -> 12:04:06 -------
for t0, f0, a in [(11.2, 190, 0.13), (12.6, 210, 0.12), (15.5, 175, 0.13),
                   (16.3, 230, 0.11), (20.8, 200, 0.13), (21.7, 185, 0.12),
                   (22.5, 220, 0.12)]:
    add_buzz(t0, 0.55, f0, a)

# --- broadband transient bursts -----------------------------------------
add_burst(11.5, 0.9, 0.22, hp=120, lp=4000, decay=0.30)   # ~12:03:54-55
add_burst(16.0, 0.8, 0.20, hp=120, lp=4000, decay=0.28)   # ~12:03:59
add_burst(21.3, 0.9, 0.22, hp=120, lp=4500, decay=0.32)   # ~12:04:04-05
add_burst(24.0, 1.4, 0.34, hp=60,  lp=5500, decay=0.55)   # ~12:04:07-08 (loudest)

# --- faint broadband hiss floor -----------------------------------------
for i in range(N):
    buf[i] += 0.006 * random.uniform(-1, 1)

# --- normalize + soft clip ----------------------------------------------
peak = max(abs(v) for v in buf) or 1.0
scale = 0.9 / peak
frames = bytearray()
for v in buf:
    s = math.tanh(v * scale * 1.1)
    iv = int(max(-1.0, min(1.0, s)) * 32767)
    frames += iv.to_bytes(2, "little", signed=True)

OUT = "/home/user/vllm/cvr_spectrogram_sonification.wav"
with wave.open(OUT, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(bytes(frames))

print(f"wrote {OUT}  ({DUR:.1f}s, {SR} Hz mono, peak-normalized)")
