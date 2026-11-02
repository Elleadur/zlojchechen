"""Music theory helpers: note names, scales, keys, chords and voice leading."""

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
}


def midi_to_freq(note):
    return 440.0 * 2 ** ((note - 69) / 12)


def note_name(note):
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


def parse_pitch_class(name):
    """'C', 'c#', 'Bb' -> 0..11"""
    name = name.strip()
    name = name[:1].upper() + name[1:].lower()
    name = FLATS.get(name, name)
    if name not in NOTE_NAMES:
        raise ValueError(f"Unknown key '{name}'. Use something like C, F#, Bb.")
    return NOTE_NAMES.index(name)


class Key:
    def __init__(self, root_pc, scale):
        if scale not in SCALES:
            raise ValueError(f"Unknown scale '{scale}'. Choose from: {', '.join(SCALES)}")
        self.root_pc = root_pc
        self.scale = scale
        self.intervals = SCALES[scale]

    def __str__(self):
        return f"{NOTE_NAMES[self.root_pc]} {self.scale.replace('_', ' ')}"

    def degree_to_midi(self, degree, octave):
        """Scale degree (0 = tonic, may go past 7) in a given octave -> MIDI note."""
        o, d = divmod(degree, len(self.intervals))
        return 12 * (octave + 1 + o) + self.root_pc + self.intervals[d]

    def chord(self, degree, octave=4, size=3):
        """Stack thirds inside the scale: size 3 = triad, 4 = seventh chord."""
        return [self.degree_to_midi(degree + 2 * i, octave) for i in range(size)]

    def scale_notes(self, low, high):
        pcs = {(self.root_pc + i) % 12 for i in self.intervals}
        return [m for m in range(low, high + 1) if m % 12 in pcs]


def voice_lead(chord, previous, low=52, high=77):
    """Choose the inversion of `chord` that moves least from the previous chord."""
    notes = sorted(chord)
    candidates = []
    for inv in range(len(notes)):
        voicing = notes[inv:] + [n + 12 for n in notes[:inv]]
        for shift in (-24, -12, 0, 12):
            c = [n + shift for n in voicing]
            if c[0] >= low and c[-1] <= high:
                candidates.append(c)
    if not candidates:
        return notes
    if previous is None or len(previous) != len(notes):
        center = (low + high) / 2
        return min(candidates, key=lambda c: abs(sum(c) / len(c) - center))
    prev = sorted(previous)
    return min(candidates, key=lambda c: sum(abs(a - b) for a, b in zip(c, prev)))


_QUALITIES = {
    (4, 7): "", (3, 7): "m", (3, 6): "dim", (4, 8): "aug",
    (4, 7, 11): "maj7", (4, 7, 10): "7", (3, 7, 10): "m7",
    (3, 6, 10): "m7b5", (3, 6, 9): "dim7", (3, 7, 11): "mMaj7", (4, 8, 11): "maj7#5",
}


def chord_name(root_pc, notes):
    intervals = tuple(sorted({(n - root_pc) % 12 for n in notes} - {0}))
    return NOTE_NAMES[root_pc] + _QUALITIES.get(intervals, "")
