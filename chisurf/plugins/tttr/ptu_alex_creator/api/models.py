"""Request / result models for the ALEX Creator workflow."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class AlexRequest:
    """A convert-or-merge ALEX request normalized from GUI / CLI / RPC."""

    files: list[str] = dataclasses.field(default_factory=list)
    alex_period: int = 8000
    period_shift: int = 0
    output_format: str = "PTU"
    input_format: str = "Auto"
    #: ``"convert"`` = one ALEX file per input; ``"merge"`` = one combined file.
    mode: str = "convert"
    #: output directory for ``convert`` (defaults to each input's parent).
    output_dir: str = ""
    #: explicit output path for ``merge`` (defaults into ``output_dir``).
    output_path: str = ""


@dataclasses.dataclass
class AlexResult:
    """The output of an :class:`AlexRequest`."""

    output_paths: list[str] = dataclasses.field(default_factory=list)
    mode: str = "convert"


__all__ = ["AlexRequest", "AlexResult"]
