"""GUI entrypoint for the Intensity imaging plugin (toolbar + file drops)."""

from __future__ import annotations

from chisurf.plugins.microscopy.imaging_common.tool_base import ImagingMapTool

from .view_model import IntensityViewModel


class ImgPixelIntensityTool(ImagingMapTool):
    """Per-pixel intensity imaging tool (creates the standard imaging HDF5)."""

    def __init__(self, parent=None, embedded: bool = False, view_model=None):
        super().__init__(
            view_model or IntensityViewModel(), title="Intensity",
            parent=parent, embedded=embedded,
        )


__all__ = ["ImgPixelIntensityTool"]
