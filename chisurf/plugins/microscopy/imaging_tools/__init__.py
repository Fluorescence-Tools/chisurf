"""Imaging Tools — combined NavigationPanelTool plugin.

Provides a single unified window with:
- Image Browser (TTTR Image Browser)
- Molecule-wise MLE (sm_image_mle)
- Pixel-wise MLE (img_pixel_mle)
- (separator)
- PSF Determination (psf_determination)
"""

from __future__ import annotations

from pathlib import Path
import json

_manifest_path = Path(__file__).parent / "manifest.json"
_manifest = json.loads(_manifest_path.read_text()) if _manifest_path.exists() else {}
name = _manifest.get("display_name", "Imaging:Tools")


def __getattr__(attr_name: str):
    """Lazy Qt import gate."""
    if attr_name == "ImagingToolsTool":
        from .gui.tool import ImagingToolsTool as _cls
        globals()["ImagingToolsTool"] = _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {attr_name!r}")


if __name__ == "plugin":
    from .gui.tool import ImagingToolsTool
    window = ImagingToolsTool()
    window.show()

__all__ = ["ImagingToolsTool"]
