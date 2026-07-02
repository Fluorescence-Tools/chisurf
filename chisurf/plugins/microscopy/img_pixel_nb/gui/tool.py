"""GUI entrypoint for the N&B imaging plugin (toolbar + file drops)."""

from __future__ import annotations

from chisurf.plugins.microscopy.imaging_common.tool_base import ImagingMapTool

from .view_model import NBViewModel


class ImgPixelNBTool(ImagingMapTool):
    """Per-pixel Number & Brightness imaging tool."""

    def __init__(self, parent=None, embedded: bool = False, view_model=None):
        super().__init__(
            view_model or NBViewModel(), title="Number & Brightness",
            parent=parent, embedded=embedded,
        )


__all__ = ["ImgPixelNBTool"]
