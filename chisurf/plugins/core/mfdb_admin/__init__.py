"""mfdb-admin: Multiparametric Fluorescence Database management."""

from __future__ import annotations

try:
    from qtpy import sip
except ImportError:
    try:
        import sip
    except ImportError:
        sip = None

from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness

from .gui.tool import MFDBWidget

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Tools:mfdb-admin"

__all__ = ["MFDBWidget", "name"]


if __name__ == "plugin":
    existing = globals().get("window")
    if existing is not None and sip is not None and sip.isdeleted(existing):
        existing = None
    if existing is None:
        window = MFDBWidget()
        if _manifest is not None:
            apply_manifest_statefulness(window, _manifest)
    else:
        window = existing
    window.show()
    try:
        window.raise_()
        window.activateWindow()
    except Exception:
        pass
