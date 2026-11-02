"""Software instruments and drum kits. Every sound is built from math with numpy, no samples."""

import numpy as np

SR = 44100
TAU = 2 * np.pi


def _t(n):
    return np.arange(n) / SR


def adsr(n, n_on, attack, decay, sustain, release):
    """Envelope: rises over `attack`, settles to `sustain`, fades out over `release` after note-off."""
    t = _t(n)
    attack = max(attack, 1e-3)
    held = np.where(t < attack, t / attack,
                    sustain + (1 - sustain) * np.exp(-np.maximum(t - attack, 0) / max(decay, 1e-3)))
    n_on = min(n_on, n - 1)
    env = held.copy()
    tail = _t(n - n_on)
    env[n_on:] = held[n_on] * np.exp(-tail * 5 / max(release, 1e-3))
    return env


def _length(dur, release):
    n_on = max(1, int(dur * SR))
    return n_on, n_on + int(release * SR)


def _saw(phase, freq, max_harm=12, rolloff=1.0):
    k_max = int(max(1, min(max_harm, 16000 // max(freq, 1))))
    return sum(np.sin(k * phase) / k ** rolloff for k in range(1, k_max + 1)) * 0.6


def _fade(x, samples=64):
    if len(x) > samples:
        x[-samples:] *= np.linspace(1, 0, samples)
    return x


# ---------------------------------------------------------------- pitched instruments

def piano(freq, dur, vel, rng):
    n_on, n = _length(dur, 0.6)
    t = _t(n)
    out = np.zeros(n)
    decay = 0.7 + freq / 500
    for k in range(1, 11):
        fk = k * freq * np.sqrt(1 + 0.0002 * k * k)
        if fk > 15000:
            break
        amp = (1 / k ** 1.3) * vel ** (0.4 * (k - 1))  # harder = brighter
        out += amp * np.sin(TAU * fk * t + rng.random()) * np.exp(-t * decay * (1 + 0.4 * (k - 1)))
    out += rng.normal(0, 1, n) * np.exp(-t * 400) * 0.03  # hammer
    return _fade(out * adsr(n, n_on, 0.002, 10, 1, 0.6) * 0.5)


def epiano(freq, dur, vel, rng):
    n_on, n = _length(dur, 0.35)
    t = _t(n)
    index = (0.5 + 1.6 * vel) * np.exp(-t * 3)
    out = np.sin(TAU * freq * t + index * np.sin(TAU * freq * t)) * np.exp(-t * (0.6 + freq / 1500))
    out += 0.12 * vel * np.sin(TAU * freq * 7 * t) * np.exp(-t * 14)  # bell-like tine
    out *= 1 + 0.12 * np.sin(TAU * 4.5 * t)  # gentle tremolo
    return _fade(out * adsr(n, n_on, 0.003, 10, 1, 0.35) * 0.5)


def pluck(freq, dur, vel, rng):
    """Karplus-Strong plucked string, resampled so it's exactly in tune."""
    n_on, n = _length(dur, 0.2)
    period = int(SR / freq)
    actual = SR / (period + 0.5)
    ratio = freq / actual
    m = int(n * ratio) + period + 2
    y = np.zeros(m)
    burst = rng.uniform(-1, 1, period)
    for _ in range(1 if vel > 0.7 else 2):  # softer = darker
        burst = 0.5 * (burst + np.roll(burst, 1))
    y[:period] = burst
    for start in range(period, m, period):
        end = min(start + period, m)
        a = y[start - period:end - period]
        b = y[start - period - 1:end - period - 1] if start > period else np.concatenate(([0.0], y[:end - period - 1]))
        y[start:end] = 0.996 * 0.5 * (a + b)
    out = np.interp(np.arange(n) * ratio, np.arange(m), y)
    return _fade(out * adsr(n, n_on, 0.001, 10, 1, 0.2) * 0.8)


def lead(freq, dur, vel, rng):
    n_on, n = _length(dur, 0.15)
    t = _t(n)
    vib = 1 + 0.006 * np.sin(TAU * 5.5 * t) * np.clip((t - 0.15) / 0.2, 0, 1)
    phase = TAU * np.cumsum(freq * vib) / SR
    out = _saw(phase, freq, 12) + 0.5 * _saw(phase * 1.004, freq, 8)
    return _fade(out * adsr(n, n_on, 0.01, 0.15, 0.7, 0.15) * 0.45)


def stab(freq, dur, vel, rng):
    n_on, n = _length(dur, 0.08)
    phase = TAU * freq * _t(n)
    out = _saw(phase, freq, 10) + _saw(phase * 1.007, freq, 10)
    return _fade(out * adsr(n, n_on, 0.002, 0.12, 0.1, 0.08) * 0.35)


def pad(freq, dur, vel, rng):
    n_on, n = _length(dur, 1.8)
    t = _t(n)
    out = np.zeros(n)
    for cents in (-9, 0, 9):
        f = freq * 2 ** (cents / 1200)
        out += _saw(TAU * f * t + rng.random() * TAU, f, 10, rolloff=1.6)
    out *= 1 + 0.1 * np.sin(TAU * 0.3 * t + rng.random() * TAU)  # slow movement
    return _fade(out * adsr(n, n_on, 0.7, 1.0, 0.85, 1.8) * 0.25)


def saw_bass(freq, dur, vel, rng):
    n_on, n = _length(dur, 0.06)
    phase = TAU * freq * _t(n)
    out = _saw(phase, freq, 8, rolloff=1.3) + 0.7 * np.sin(phase)
    return _fade(np.tanh(out * 1.5) * adsr(n, n_on, 0.004, 0.2, 0.6, 0.06) * 0.6)


def sub_bass(freq, dur, vel, rng):
    n_on, n = _length(dur, 0.15)
    phase = TAU * freq * _t(n)
    out = np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)
    return _fade(out * adsr(n, n_on, 0.01, 0.5, 0.75, 0.15) * 0.7)


def bell(freq, dur, vel, rng):
    n_on, n = _length(max(dur, 1.5), 1.5)
    t = _t(n)
    index = 2.0 * vel * np.exp(-t * 1.5)
    out = np.sin(TAU * freq * t + index * np.sin(TAU * freq * 3.5 * t)) * np.exp(-t * 0.9)
    return _fade(out * adsr(n, n_on, 0.002, 10, 1, 1.5) * 0.4)


def square(freq, dur, vel, rng):
    n_on, n = _length(dur, 0.04)
    out = np.sign(np.sin(TAU * freq * _t(n))) * 0.3
    return _fade(out * adsr(n, n_on, 0.002, 0.1, 0.6, 0.04))


def triangle(freq, dur, vel, rng):
    n_on, n = _length(dur, 0.03)
    out = (2 / np.pi) * np.arcsin(np.sin(TAU * freq * _t(n)))
    return _fade(out * adsr(n, n_on, 0.002, 0.2, 0.9, 0.03) * 0.7)


# name -> (sound function, General MIDI program number for the .mid file)
INSTRUMENTS = {
    "piano": (piano, 0), "epiano": (epiano, 4), "pluck": (pluck, 25), "lead": (lead, 81),
    "stab": (stab, 81), "pad": (pad, 89), "saw_bass": (saw_bass, 38), "sub_bass": (sub_bass, 33),
    "bell": (bell, 14), "square": (square, 80), "triangle": (triangle, 38),
}


# ---------------------------------------------------------------- drums

def _kick(vel, rng, punch=1.0, length=0.45):
    n = int(length * SR)
    t = _t(n)
    phase = TAU * np.cumsum(45 + 110 * np.exp(-t * 30)) / SR
    out = np.sin(phase) * np.exp(-t * 7) + rng.normal(0, 1, n) * np.exp(-t * 600) * 0.25 * punch
    return np.tanh(out * (1 + punch)) * 0.9


def _snare(vel, rng, tone=185, snappy=1.0):
    n = int(0.3 * SR)
    t = _t(n)
    noise = np.diff(rng.normal(0, 1, n + 1))
    out = 0.5 * np.sin(TAU * tone * t) * np.exp(-t * 25) + 0.35 * snappy * noise * np.exp(-t * 18)
    return out


def _clap(vel, rng):
    n = int(0.3 * SR)
    t = _t(n)
    noise = np.diff(rng.normal(0, 1, n + 1))
    env = sum(np.where(t >= d, np.exp(-(t - d) * 120), 0) for d in (0, 0.01, 0.02))
    env += np.where(t >= 0.03, np.exp(-(t - 0.03) * 16), 0) * 0.6
    return noise * env * 0.35


def _hat(vel, rng, decay=55, bright=2):
    n = int(0.12 * SR)
    noise = rng.normal(0, 1, n + bright)
    for _ in range(bright):
        noise = np.diff(noise)
    return noise * np.exp(-_t(n) * decay) * 0.12


def _chip_noise(vel, rng, decay, length):
    n = int(length * SR)
    hold = 4 if decay < 50 else 1  # chunkier noise for the snare
    noise = np.repeat(rng.choice([-1.0, 1.0], n // hold + 1), hold)[:n]
    return noise * np.exp(-_t(n) * decay) * 0.35


def _chip_kick(vel, rng):
    n = int(0.15 * SR)
    t = _t(n)
    phase = TAU * np.cumsum(50 + 180 * np.exp(-t * 40)) / SR
    return np.sign(np.sin(phase)) * np.exp(-t * 25) * 0.5


KITS = {
    "lofi": {36: lambda v, r: _kick(v, r, punch=0.3, length=0.4),
             38: lambda v, r: _snare(v, r, tone=170, snappy=0.7),
             42: lambda v, r: _hat(v, r, decay=70, bright=1)},
    "electronic": {36: lambda v, r: _kick(v, r, punch=1.2, length=0.5),
                   38: _clap,
                   42: lambda v, r: _hat(v, r, decay=35, bright=2)},
    "chip": {36: _chip_kick,
             38: lambda v, r: _chip_noise(v, r, decay=25, length=0.15),
             42: lambda v, r: _chip_noise(v, r, decay=90, length=0.05)},
}


def render_note(instrument, freq, dur, vel, rng):
    return INSTRUMENTS[instrument][0](freq, dur, vel, rng)


def render_drum(kit, note, vel, rng):
    return KITS[kit][note](vel, rng)
