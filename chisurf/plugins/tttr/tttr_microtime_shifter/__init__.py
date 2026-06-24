"""
Micro-time Shifter Plugin

Apply global and per-channel micro-time shifts to TTTR files.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name

__all__ = ["MicrotimeShifterTool"]


def __getattr__(name: str):
    """Lazily import the GUI tool (PRD-23: no Qt import as a package side effect).

    Keeps ``from ...tttr_microtime_shifter import MicrotimeShifterTool`` working
    while the Qt-free ``api``/``cli`` submodules can be imported headlessly.
    """
    if name == "MicrotimeShifterTool":
        from .gui.tool import MicrotimeShifterTool

        return MicrotimeShifterTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "plugin":
    from .gui.tool import MicrotimeShifterTool

    window = MicrotimeShifterTool()
    window.show()
