"""GUI entrypoint for the mean-micro-time imaging plugin (toolbar + file drops)."""

from __future__ import annotations

from chisurf.plugins.microscopy.imaging_common.tool_base import ImagingMapTool

from .view_model import MicroTimeViewModel


class ImgPixelMicroTimeTool(ImagingMapTool):
    """Per-pixel mean micro-time imaging tool."""

    def __init__(self, parent=None, embedded: bool = False, view_model=None):
        super().__init__(
            view_model or MicroTimeViewModel(), title="Mean Micro-Time",
            parent=parent, embedded=embedded,
        )


__all__ = ["ImgPixelMicroTimeTool"]
