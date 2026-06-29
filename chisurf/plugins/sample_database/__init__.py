"""Legacy compatibility wrapper — delegates to chisurf.plugins.core.mfdb_admin."""

from __future__ import annotations

from pathlib import Path

try:
    from qtpy import sip
except ImportError:
    try:
        import sip
    except ImportError:
        sip = None

from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness
from chisurf.plugins.core.mfdb_admin import MFDBWidget as SampleDatabaseWidget

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))

__all__ = ["SampleDatabaseWidget"]

if __name__ == "plugin":
    existing = globals().get("window")
    if existing is not None and sip is not None and not sip.isdeleted(existing):
        window = existing
    else:
        window = SampleDatabaseWidget()
        if _manifest is not None:
            apply_manifest_statefulness(window, _manifest)
    if window is not None:
        window.show()
        try:
            window.raise_()
            window.activateWindow()
        except Exception:
            pass
