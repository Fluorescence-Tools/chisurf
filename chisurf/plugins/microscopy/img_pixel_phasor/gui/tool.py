"""GUI entrypoint for the phasor-FLIM imaging plugin (toolbar + file drops)."""

from __future__ import annotations

from chisurf.plugins.microscopy.imaging_common.tool_base import ImagingMapTool

from .view_model import PhasorImgViewModel


class ImgPixelPhasorTool(ImagingMapTool):
    """Per-pixel phasor-FLIM imaging tool."""

    def __init__(self, parent=None, embedded: bool = False, view_model=None):
        super().__init__(
            view_model or PhasorImgViewModel(), title="Phasor-FLIM",
            parent=parent, embedded=embedded,
        )


__all__ = ["ImgPixelPhasorTool"]
