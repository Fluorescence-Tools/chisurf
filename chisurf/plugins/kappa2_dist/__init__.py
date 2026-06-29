"""K² Distribution calculator for FRET orientation factors.

Provides a GUI tool (``Kappa2Dist``) for computing and visualising the
orientation-factor distribution p(κ²) using Wobbling-in-Cone (WIC),
Diffusion-with-Traps (DWT) and isotropic models.  The widget is built
via AutoForm (PRD-40) from ``k2dist.view.json``.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Structure:FRET:Kappa2 Distribution"

from .k2dgui import Kappa2Dist  # noqa: E402

__all__ = ["Kappa2Dist"]

if __name__ == "plugin":
    window = Kappa2Dist()
    if _manifest is not None:
        apply_manifest_statefulness(window, _manifest)
    window.show()
    try:
        window.raise_()
        window.activateWindow()
    except Exception:
        pass
