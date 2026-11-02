"""zlojchechen music generator: composes and renders songs from code."""

from .composer import compose
from .render import render, save_midi, save_mp3, save_wav
from .styles import STYLES

__all__ = ["compose", "render", "save_wav", "save_mp3", "save_midi", "STYLES"]
