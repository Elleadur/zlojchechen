"""Turns a style preset into a song: chords, bass line, melody and drums as note events."""

from dataclasses import dataclass, field

import numpy as np

from .styles import STYLES
from .theory import Key, SCALES, chord_name, voice_lead

DRUM_NOTES = {"kick": 36, "snare": 38, "hat": 42}
BEATS_PER_BAR = 4


@dataclass
class Track:
    name: str
    role: str
    instrument: str
    gain: float = 0.5
    pan: float = 0.0
    send: float = 1.0
    duck: float = 0.0
    events: list = field(default_factory=list)  # (start_beat, length_beats, midi_note, velocity)


@dataclass
class Song:
    style: str
    key: Key
    bpm: int
    bars: int
    seed: int
    chord_names: list
    tracks: list

    @property
    def seconds(self):
        return self.bars * BEATS_PER_BAR * 60 / self.bpm


# ---------------------------------------------------------------- melody

# Two-beat rhythm cells. Negative lengths are rests. Second value = how busy the cell is (0..1).
CELLS = [
    ([2.0], 0.1), ([1.0, 1.0], 0.4), ([1.5, 0.5], 0.5), ([0.5, 0.5, 1.0], 0.65),
    ([1.0, 0.5, 0.5], 0.65), ([0.5, 1.0, 0.5], 0.6), ([0.5, 0.5, 0.5, 0.5], 0.85),
    ([-1.0, 1.0], 0.25), ([-0.5, 0.5, 1.0], 0.5), ([1.0, -1.0], 0.15),
    ([0.25, 0.25, 0.5, 1.0], 0.95), ([-2.0], 0.0),
]
# Four-beat phrase endings that settle on a long note.
ENDINGS = [[1, 1, 2], [0.5, 0.5, 1, 2], [2, 2], [1, 0.5, 0.5, 2], [-1, 1, 2], [1.5, 0.5, 2]]
STEPS = [-3, -2, -1, 0, 1, 2, 3]
STEP_WEIGHTS = [0.05, 0.15, 0.3, 0.1, 0.25, 0.1, 0.05]


def _rhythm(rng, density, beats):
    weights = np.array([np.exp(-((busy - density) ** 2) / 0.06) for _, busy in CELLS])
    weights /= weights.sum()
    out, t = [], 0.0
    while t < beats:
        for d in CELLS[rng.choice(len(CELLS), p=weights)][0]:
            out.append((t, d))
            t += abs(d)
    return out


def _with_steps(rng, rhythm, offset=0.0):
    return [(t + offset, d, int(rng.choice(STEPS, p=STEP_WEIGHTS))) for t, d in rhythm]


def make_motif(rng, density):
    """A two-bar idea: a rhythm plus up/down movement, reused across chord changes."""
    for _ in range(20):
        motif = _with_steps(rng, _rhythm(rng, density, 8))
        if sum(1 for _, d, _ in motif if d > 0) >= 3:
            return motif
    return motif


def make_answer(rng, motif):
    """Same start as the motif, but its second bar becomes a phrase ending."""
    ending = ENDINGS[rng.integers(len(ENDINGS))]
    rhythm, t = [], 0.0
    for d in ending:
        rhythm.append((t, d))
        t += abs(d)
    return [e for e in motif if e[0] < 4] + _with_steps(rng, rhythm, offset=4.0)


def _nearest(pool, idx, pcs):
    for dist in range(len(pool)):
        for i in (idx - dist, idx + dist):
            if 0 <= i < len(pool) and pool[i] % 12 in pcs:
                return i
    return idx


def realize(events, start, pool, pcs_at, idx, rng, final_pcs=None):
    """Turn a motif into notes over the actual chords, snapping strong beats to chord tones."""
    notes = []
    last = max((j for j, e in enumerate(events) if e[1] > 0), default=-1)
    for j, (off, dur, step) in enumerate(events):
        if dur <= 0:
            continue
        if rng.random() < 0.12:
            step += int(rng.choice([-1, 1]))
        # Keep the line in its range by turning around near the edges.
        if (idx > 0.85 * len(pool) and step > 0) or (idx < 0.15 * len(pool) and step < 0):
            step = -step
        idx = int(np.clip(idx + step, 0, len(pool) - 1))
        t = start + off
        if j == last and final_pcs:
            idx = _nearest(pool, idx, final_pcs)
        elif off % 2 == 0 or dur >= 1.5:
            idx = _nearest(pool, idx, pcs_at(t))
        vel = 82 + (10 if off % 1 == 0 else 0) + int(rng.integers(-6, 7))
        notes.append((t, dur * 0.92, pool[idx], vel))
    return notes, idx


# ---------------------------------------------------------------- song

def compose(style="lofi", bars=32, key=None, scale=None, bpm=None, seed=None):
    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from: {', '.join(STYLES)}")
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 1_000_000))
    rng = np.random.default_rng(seed)
    st = STYLES[style]
    bars = max(4, int(bars))

    scale = scale or str(rng.choice(st["scales"]))
    if scale not in SCALES:
        raise ValueError(f"Unknown scale '{scale}'. Choose from: {', '.join(SCALES)}")
    root_pc = key if key is not None else int(rng.integers(12))
    k = Key(root_pc, scale)
    bpm = int(bpm or rng.integers(st["bpm"][0], st["bpm"][1] + 1))
    prog = st["progressions"][rng.integers(len(st["progressions"]))]

    intro = 2 if bars >= 8 else 0
    degrees = [prog[b % len(prog)] for b in range(bars)]
    degrees[-1] = 0  # end on the home chord
    # Diminished chords sound harsh in these styles: use the chord a third below instead
    # (it shares two notes, e.g. vii° -> V).
    for i, d in enumerate(degrees):
        root, third, fifth = k.chord(d, 4, 3)
        if fifth - root == 6:
            degrees[i] = d - 2

    voicings, prev = [], None
    for d in degrees:
        prev = voice_lead(k.chord(d, 4, st["chord_size"]), prev)
        voicings.append(prev)
    roots = [k.degree_to_midi(d, 0) % 12 for d in degrees]
    pcs = [{n % 12 for n in v} for v in voicings]

    def pcs_at(beat):
        return pcs[min(int(beat // BEATS_PER_BAR), bars - 1)]

    tracks = []
    for layer in st["layers"]:
        tr = Track(layer["name"], layer["role"], layer["instrument"], layer.get("gain", 0.5),
                   layer.get("pan", 0.0), layer.get("send", 1.0), layer.get("duck", 0.0))
        role = layer["role"]
        if role == "chords":
            _chords(tr, layer, voicings, rng)
        elif role == "bass":
            _bass(tr, layer, roots)
        elif role == "drums":
            _drums(tr, layer, bars, intro, rng)
        elif role == "melody":
            _melody(tr, layer, k, bars, intro, pcs_at, rng)
        tracks.append(tr)

    if st["swing"]:
        for tr in tracks:
            tr.events = [(_swing(s, st["swing"]), d, n, v) for s, d, n, v in tr.events]

    names = [chord_name(roots[b], voicings[b]) for b in range(min(len(prog), bars))]
    return Song(style, k, bpm, bars, seed, names, tracks)


def _swing(beat, amount):
    frac = beat % 1
    if abs(frac - 0.5) < 1e-6:
        return beat + amount * 0.5
    if abs(frac - 0.25) < 1e-6 or abs(frac - 0.75) < 1e-6:
        return beat + amount * 0.25
    return beat


def _chords(tr, layer, voicings, rng):
    base_vel = layer.get("velocity", 70)
    rhythm = layer["rhythm"]
    last = len(voicings) - 1
    for b, v in enumerate(voicings):
        bar = b * BEATS_PER_BAR
        if b == last:  # final bar: one long chord
            for n in v:
                tr.events.append((bar, 4.0, n, base_vel))
            continue
        if isinstance(rhythm, str):  # arpeggio
            step = 0.5 if rhythm == "arp8" else 0.25
            seq = list(v) + [v[0] + 12]
            cycle = seq + seq[-2:0:-1]
            ring = 1.0 if rhythm == "arp8" else step * 0.9
            for i in range(int(BEATS_PER_BAR / step)):
                vel = base_vel + (8 if i % 4 == 0 else 0) + int(rng.integers(-5, 6))
                tr.events.append((bar + i * step, ring, cycle[i % len(cycle)], vel))
        else:
            for start, length in rhythm:
                vel = base_vel + int(rng.integers(-6, 7))
                for n in v:
                    tr.events.append((bar + start, length, n, vel))


def _bass(tr, layer, roots):
    octave = layer.get("octave", 2)
    last = len(roots) - 1
    for b, pc in enumerate(roots):
        root = 12 * (octave + 1) + pc
        if pc > 7:
            root -= 12  # keep the bass low
        bar = b * BEATS_PER_BAR
        if b == last:
            tr.events.append((bar, 4.0, root, 95))
            continue
        for start, length, which in layer["pattern"]:
            note = root + {"root": 0, "fifth": 7, "octave": 12}[which]
            tr.events.append((bar + start, length, note, 92 if start % 1 == 0 else 84))


def _drums(tr, layer, bars, intro, rng):
    pattern = layer["pattern"]
    for b in range(bars):
        bar = b * BEATS_PER_BAR
        if b == bars - 1:
            tr.events.append((bar, 1.0, DRUM_NOTES["kick"], 110))
            continue
        if b < intro:
            if b == intro - 1:  # build-up: hats only
                for s in pattern.get("hat", []):
                    tr.events.append((bar + s / 4, 0.25, DRUM_NOTES["hat"], 60))
            continue
        for part, steps in pattern.items():
            for s in steps:
                base = {"kick": 108, "snare": 100, "hat": 74 if s % 4 == 0 else 58}[part]
                tr.events.append((bar + s / 4, 0.25, DRUM_NOTES[part], base + int(rng.integers(-8, 9))))
        if (b - intro) % 8 == 7:  # little fill every 8 bars
            for s, vel in ((13, 70), (14, 82), (15, 95)):
                tr.events.append((bar + s / 4, 0.25, DRUM_NOTES["snare"], vel))


def _melody(tr, layer, key, bars, intro, pcs_at, rng):
    lo, hi = layer["range"]
    pool = key.scale_notes(lo, hi)
    density = layer.get("density", 0.5)
    motif_a = make_motif(rng, density)
    motif_b = make_motif(rng, min(1.0, density * 1.25))
    half = intro + (bars - 1 - intro) // 2
    idx = len(pool) // 2
    tonic = {key.root_pc}

    bar = intro
    while bar < bars - 1:
        remaining = bars - 1 - bar
        motif = motif_a if bar < half else motif_b
        if bar == half:
            idx = min(len(pool) - 1, idx + 2)  # second half sits a little higher
        start = bar * BEATS_PER_BAR
        if remaining >= 4:
            notes, idx = realize(motif, start, pool, pcs_at, idx, rng)
            tr.events += notes
            final = tonic if remaining == 4 else None
            notes, idx = realize(make_answer(rng, motif), start + 8, pool, pcs_at, idx, rng, final)
            tr.events += notes
            bar += 4
        elif remaining >= 2:
            notes, idx = realize(make_answer(rng, motif), start, pool, pcs_at, idx, rng, tonic)
            tr.events += notes
            bar += 2
        else:
            idx = _nearest(pool, idx, pcs_at(start))
            tr.events.append((start, 3.6, pool[idx], 80))
            bar += 1
    if bars - 1 > intro:
        idx = _nearest(pool, idx, tonic)
        tr.events.append(((bars - 1) * BEATS_PER_BAR, 3.8, pool[idx], 78))
