"""Pure dataclasses for the CLSM plugin — no Qt, no tttrlib at import time."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClsmSetup:
    """Acquisition setup needed to turn a TTTR stream into a CLSM image.

    Mirrors one preset of ``core.setups.builtin_setups`` plus the routing
    channels selected by the user.
    """

    tttr_type: str = "PTU"
    routine: str = "default"
    frame_marker: list[int] = field(default_factory=lambda: [4])
    line_start_marker: int = 1
    line_stop_marker: int = 2
    event_type_marker: int = 1
    pixel_per_line: int = 0
    channels: list[int] = field(default_factory=lambda: [0])

    @classmethod
    def from_preset(cls, preset: dict, channels: list[int] | None = None) -> ClsmSetup:
        """Build a setup from a ``builtin_setups`` preset dict."""
        return cls(
            tttr_type=preset.get("tttr_type", "PTU"),
            routine=preset.get("routine", "default"),
            frame_marker=list(preset.get("frame_marker", [4])),
            line_start_marker=int(preset.get("line_start_marker", 1)),
            line_stop_marker=int(preset.get("line_stop_marker", 2)),
            event_type_marker=int(preset.get("event_type_marker", 1)),
            pixel_per_line=int(preset.get("pixel_per_line", 0) or 0),
            channels=list(channels if channels is not None else [0]),
        )


@dataclass
class RepresentationResult:
    """Metadata describing a computed image representation."""

    image_type: str
    n_frames: int
    n_lines: int
    n_pixel: int
    total_intensity: float
    output_path: str = ""


@dataclass
class DecayResult:
    """A fluorescence-decay histogram extracted from a pixel selection."""

    time_ns: list[float]
    counts: list[float]
    noise: list[float]
    n_photons: int
    output_path: str = ""


@dataclass
class FrcResult:
    """A Fourier-Ring-Correlation curve for an image."""

    frequency: list[float]
    correlation: list[float]
    output_path: str = ""
