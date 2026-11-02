#!/usr/bin/env python3
"""Generate a song.

Examples:
    python generate.py                       # random lo-fi track
    python generate.py --style piano --bars 24 --key D --scale minor
    python generate.py --style electronic --seed 42 --bpm 124
    python generate.py --style all --seed 7  # one song in every style
    python generate.py --list
"""

import argparse
import time
from pathlib import Path

from musicgen import STYLES, compose, render, save_midi, save_mp3, save_wav
from musicgen.theory import SCALES, parse_pitch_class


def make_song(args, style, out_dir):
    started = time.time()
    key = parse_pitch_class(args.key) if args.key else None
    song = compose(style, bars=args.bars, key=key, scale=args.scale, bpm=args.bpm, seed=args.seed)
    name = args.name if args.name and args.style != "all" else f"{style}_{song.seed}"
    base = out_dir / name

    audio = render(song)
    save_wav(audio, base.with_suffix(".wav"))
    save_midi(song, base.with_suffix(".mid"))
    files = [base.with_suffix(".wav"), base.with_suffix(".mid")]
    if not args.no_mp3 and save_mp3(base.with_suffix(".wav"), base.with_suffix(".mp3")):
        files.append(base.with_suffix(".mp3"))

    mins, secs = divmod(round(len(audio) / 44100), 60)
    print(f"\n♪ {style}  |  {song.key}  |  {song.bpm} BPM  |  {mins}:{secs:02d}  |  seed {song.seed}")
    print(f"  chords: {' - '.join(song.chord_names)}")
    print(f"  parts:  {', '.join(t.name for t in song.tracks)}")
    for f in files:
        print(f"  saved:  {f}")
    print(f"  ({time.time() - started:.1f}s)")


def main():
    p = argparse.ArgumentParser(description="Generate music from code.")
    p.add_argument("--style", default="lofi", choices=list(STYLES) + ["all"])
    p.add_argument("--bars", type=int, default=32, help="song length in bars (4 beats each)")
    p.add_argument("--key", help="root note, e.g. C, F#, Bb (random if not set)")
    p.add_argument("--scale", choices=list(SCALES), help="random from the style if not set")
    p.add_argument("--bpm", type=int, help="tempo (random within the style's range if not set)")
    p.add_argument("--seed", type=int, help="same seed + same options = same song")
    p.add_argument("--out", default="output", help="folder to save into")
    p.add_argument("--name", help="file name without extension")
    p.add_argument("--no-mp3", action="store_true", help="skip the MP3 copy")
    p.add_argument("--list", action="store_true", help="show the available styles")
    args = p.parse_args()

    if args.list:
        for name, st in STYLES.items():
            print(f"{name:11} {st['bpm'][0]}-{st['bpm'][1]} BPM  {st['description']}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for style in (STYLES if args.style == "all" else [args.style]):
        make_song(args, style, out_dir)


if __name__ == "__main__":
    main()
