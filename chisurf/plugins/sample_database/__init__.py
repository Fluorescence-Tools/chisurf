"""Sample database management plugin."""

from __future__ import annotations

try:
    from qtpy import sip
except ImportError:
    try:
        import sip
    except ImportError:
        sip = None

from .gui.tool import SampleDatabaseWidget

name = "Tools:Sample Database"

__all__ = ["SampleDatabaseWidget", "name"]


if __name__ == "plugin":
    existing = globals().get("window")
    if existing is not None and sip is not None and sip.isdeleted(existing):
        existing = None
    if existing is None:
        window = SampleDatabaseWidget()
    else:
        window = existing
    window.show()
    try:
        window.raise_()
        window.activateWindow()
    except Exception:
        pass
