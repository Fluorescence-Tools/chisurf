"""Self-contained sound manager for Breakout game.

Generates WAV data programmatically — no external sound files needed.
"""

from __future__ import annotations

import math
import struct
import tempfile
import wave
from pathlib import Path

from qtpy.QtCore import QUrl
from qtpy.QtMultimedia import QSoundEffect


def _generate_wav(
    path: Path,
    frequency: float,
    duration: float,
    sample_rate: int = 22050,
    volume: float = 0.3,
    fade_out: bool = True,
) -> None:
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        env = 1.0
        if fade_out:
            env = 1.0 - (i / n_samples) * 0.5
        val = int(volume * env * 32767 * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", val))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(samples))


def _generate_sweep_wav(
    path: Path,
    freq_start: float,
    freq_end: float,
    duration: float,
    sample_rate: int = 22050,
    volume: float = 0.3,
) -> None:
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        frac = i / n_samples
        freq = freq_start + (freq_end - freq_start) * frac
        env = 1.0 - frac * 0.5
        val = int(volume * env * 32767 * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack("<h", val))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(samples))


class SoundManager:
    """Plays simple game sounds generated at runtime.

    All sounds are off by default (``muted=True``). Call ``toggle()``
    or set ``muted = False`` to enable sound.
    """

    def __init__(self) -> None:
        self._muted = True
        self._tmpdir: Path | None = None
        self._effects: dict[str, QSoundEffect] = {}

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        self._muted = value

    def toggle(self) -> None:
        self._muted = not self._muted

    def play(self, name: str) -> None:
        if self._muted:
            return
        effect = self._effects.get(name)
        if effect is None:
            return
        effect.play()

    def ensure_sounds(self) -> None:
        if self._tmpdir is not None:
            return
        self._tmpdir = Path(tempfile.mkdtemp(prefix="breakout_sounds_"))
        self._generate("paddle_hit", _generate_wav, 440, 0.08, volume=0.2)
        self._generate("brick_break", _generate_wav, 660, 0.1, volume=0.25)
        self._generate("wall_bounce", _generate_wav, 330, 0.06, volume=0.15)
        self._generate("game_over", _generate_wav, 196, 0.6, volume=0.3)
        self._generate("level_up", _generate_sweep_wav, 440, 880, 0.4, volume=0.3)
        self._generate("launch", _generate_sweep_wav, 330, 660, 0.15, volume=0.2)

    def _generate(self, name: str, func, *args, **kwargs) -> None:
        path = self._tmpdir / f"{name}.wav"
        func(path, *args, **kwargs)
        effect = QSoundEffect()
        effect.setSource(QUrl.fromLocalFile(str(path)))
        effect.setVolume(1.0)
        self._effects[name] = effect
