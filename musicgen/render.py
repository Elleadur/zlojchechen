"""Mixes a Song into stereo audio and saves WAV / MP3 / MIDI files."""

import shutil
import struct
import subprocess
import wave

import numpy as np

from .composer import BEATS_PER_BAR
from .styles import STYLES
from .synth import INSTRUMENTS, SR, render_drum, render_note
from .theory import midi_to_freq


def _fft_convolve(x, ir):
    n = len(x) + len(ir) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]


def lowpass(x, cutoff, order=2):
    """Smooth low-pass filter applied in the frequency domain."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    spec /= np.sqrt(1 + (f / cutoff) ** (2 * order))
    return np.fft.irfft(spec, len(x))


def _reverb_ir(room, rng):
    n = int(room * SR)
    t = np.arange(n) / SR
    pre = int(0.015 * SR)
    irs = []
    for _ in range(2):  # different noise left and right = wide stereo
        ir = rng.normal(0, 1, n) * np.exp(-t * 6.9 / room)
        ir = lowpass(ir, 5000)
        ir = np.concatenate((np.zeros(pre), ir))
        irs.append(ir / np.sqrt(np.sum(ir ** 2)))
    return irs


def render(song):
    st = STYLES[song.style]
    rng = np.random.default_rng(song.seed + 1)
    spb = 60 / song.bpm
    length = int((song.bars * BEATS_PER_BAR * spb + st["room"] + 2) * SR)
    humanize = st.get("humanize", 0.0)

    kicks = [s for tr in song.tracks if tr.role == "drums" for s, _, n, _ in tr.events if n == 36]
    duck_env = np.ones(length)
    if kicks:
        tail = np.arange(int(0.25 * SR)) / SR
        shape = np.exp(-tail / 0.08)
        for s in kicks:
            i0 = int(s * spb * SR)
            seg = duck_env[i0:i0 + len(shape)]
            seg[:] = np.minimum(seg, 1 - shape[: len(seg)])

    left, right, send = np.zeros(length), np.zeros(length), np.zeros(length)
    for tr in song.tracks:
        stem = np.zeros(length)
        for start, dur, note, vel in tr.events:
            t0 = start * spb + (rng.normal(0, humanize) if humanize else 0)
            i0 = max(0, int(t0 * SR))
            v = vel / 127
            if tr.role == "drums":
                sig = render_drum(tr.instrument, note, v, rng)
            else:
                sig = render_note(tr.instrument, midi_to_freq(note), dur * spb, v, rng)
            end = min(i0 + len(sig), length)
            stem[i0:end] += sig[: end - i0] * (0.25 + 0.75 * v)
        if tr.duck:
            stem *= 1 - tr.duck * (1 - duck_env)
        angle = (tr.pan + 1) * np.pi / 4
        left += stem * tr.gain * np.cos(angle) * np.sqrt(2)
        right += stem * tr.gain * np.sin(angle) * np.sqrt(2)
        send += stem * tr.gain * tr.send

    if st["reverb"] > 0:
        ir_l, ir_r = _reverb_ir(st["room"], rng)
        left += _fft_convolve(send, ir_l) * st["reverb"]
        right += _fft_convolve(send, ir_r) * st["reverb"]

    audio = np.stack([left, right], axis=1)
    m = st.get("master", {})
    if m.get("lowpass"):
        audio = np.stack([lowpass(audio[:, c], m["lowpass"]) for c in range(2)], axis=1)
    if m.get("crackle"):  # vinyl: soft hiss plus random pops
        hiss = lowpass(rng.normal(0, 0.004, length), 3000)
        pops = (rng.random(length) < 3 / SR) * rng.normal(0, 0.25, length)
        noise = (hiss + lowpass(pops, 2500)) * m["crackle"]
        audio += noise[:, None]
    peak = np.max(np.abs(audio)) or 1.0
    audio = audio / peak * 0.9
    drive = m.get("drive", 1.0)
    if drive > 1:
        audio = np.tanh(audio * drive) / np.tanh(drive)

    # trim the silent tail and fade out
    loud = np.where(np.max(np.abs(audio), axis=1) > 1e-3)[0]
    end = min(length, (loud[-1] if len(loud) else length) + int(0.2 * SR))
    audio = audio[:end]
    fade = int(0.2 * SR)
    audio[-fade:] *= np.linspace(1, 0, fade)[:, None]
    return audio / (np.max(np.abs(audio)) or 1.0) * 0.89


def save_wav(audio, path):
    data = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def save_mp3(wav_path, mp3_path):
    if not shutil.which("ffmpeg"):
        return False
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
                    "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)], check=True)
    return True


# ---------------------------------------------------------------- MIDI

def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _chunk(kind, data):
    return kind + struct.pack(">I", len(data)) + data


def save_midi(song, path, tpb=480):
    tracks = []
    tempo = int(60_000_000 / song.bpm)
    meta = b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big") + b"\x00\xff\x58\x04\x04\x02\x18\x08"
    tracks.append(_chunk(b"MTrk", meta + b"\x00\xff\x2f\x00"))

    channel = 0
    for tr in song.tracks:
        if tr.role == "drums":
            ch, program = 9, 0
        else:
            ch, program = channel, INSTRUMENTS[tr.instrument][1]
            channel += 1
            if channel == 9:
                channel = 10
        msgs = []
        for start, dur, note, vel in tr.events:
            on = int(round(start * tpb))
            off = on + max(1, int(round(dur * tpb)))
            msgs.append((on, 1, bytes([0x90 | ch, note, max(1, min(127, vel))])))
            msgs.append((off, 0, bytes([0x80 | ch, note, 0])))
        msgs.sort(key=lambda m: (m[0], m[1]))  # note-offs first at the same tick

        name = tr.name.encode("ascii", "replace")
        data = b"\x00\xff\x03" + _vlq(len(name)) + name
        if ch != 9:
            data += b"\x00" + bytes([0xC0 | ch, program])
        now = 0
        for tick, _, msg in msgs:
            data += _vlq(tick - now) + msg
            now = tick
        data += b"\x00\xff\x2f\x00"
        tracks.append(_chunk(b"MTrk", data))

    header = _chunk(b"MThd", struct.pack(">HHH", 1, len(tracks), tpb))
    with open(path, "wb") as f:
        f.write(header + b"".join(tracks))
