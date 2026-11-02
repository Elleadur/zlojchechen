# zlojchechen

A music generator written in Python. It composes original songs (chords, bass line, melody and drums), plays them on built-in software instruments, and saves:

- **`.wav` / `.mp3`**: the finished song, ready to listen to
- **`.mid`**: the notes, so you can open the song in GarageBand, FL Studio, Ableton, MuseScore and others to change instruments or edit it

No samples, no AI service and no account are needed. Every sound is generated with math.

## Listen to the examples

The [`examples/`](examples) folder has one song in each style.

## Styles

| Style | Tempo | Sound |
|---|---|---|
| `lofi` | 70–86 BPM | Chill lo-fi hip hop: jazzy 7th chords, lazy swung drums, vinyl crackle |
| `piano` | 78–96 BPM | Solo piano ballad: flowing arpeggios and a singing melody |
| `electronic` | 118–128 BPM | Four-on-the-floor dance track: pads, offbeat stabs and bass, saw lead |
| `ambient` | 60–72 BPM | Slow, spacious ambient: long pads, sparse bells, huge reverb |
| `chiptune` | 130–150 BPM | 8-bit video game music: square-wave lead, fast arpeggios, triangle bass |

## How to use

You need Python 3.9+ and numpy. MP3 export also needs [ffmpeg](https://ffmpeg.org/); without it you still get WAV and MIDI files.

```bash
pip install -r requirements.txt
```

```bash
python generate.py
```

This makes a random lo-fi song in the `output/` folder. More examples:

```bash
python generate.py --style piano --key D --scale minor
python generate.py --style electronic --bpm 124 --bars 48
python generate.py --style all --seed 7
python generate.py --list
```

| Option | What it does |
|---|---|
| `--style` | `lofi`, `piano`, `electronic`, `ambient`, `chiptune` or `all` |
| `--bars` | Length in bars, 4 beats each (default 32) |
| `--key` | Root note such as `C`, `F#`, `Bb` (random if not set) |
| `--scale` | `major`, `minor`, `dorian`, `mixolydian`, `lydian`, `harmonic_minor` |
| `--bpm` | Tempo (random within the style's range if not set) |
| `--seed` | The same seed and options always give the same song. Every run prints its seed, so you can recreate a song you like. |
| `--out` / `--name` | Output folder and file name |
| `--no-mp3` | Skip the MP3 copy |

## How it works

| File | Job |
|---|---|
| `musicgen/theory.py` | Notes, scales, chords and smooth voice leading between chords |
| `musicgen/styles.py` | Style presets: instruments, rhythms, tempo, mix. **Add a new style here.** |
| `musicgen/composer.py` | Writes the song: picks a key and chord progression, builds a two-bar melody idea and develops it over the chords, adds bass and drums with an intro, fills and an ending |
| `musicgen/synth.py` | The instruments: piano, electric piano, plucked guitar, pads, leads, basses, bells, 8-bit waves and three drum kits |
| `musicgen/render.py` | Mixes everything with panning, reverb, sidechain pumping and lo-fi effects, then saves WAV, MP3 and MIDI |
