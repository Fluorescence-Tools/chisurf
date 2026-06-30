"""TTTR Tools — combined NavigationPanelTool plugin.

A single unified window (left navigation, right panel) that hosts the existing
TTTR file tools as embedded panels:

- ALEX Creator        (ptu_alex_creator)
- Micro-time Shifter  (tttr_microtime_shifter)
- PTU Header Editor   (ptu_header_edit)
- (separator)
- Split / Convert     (tttr_splitter)

Each sub-tool is embedded as-is; this plugin only provides the shared shell.
The individual tools stay importable and standalone-launchable but are hidden
from the ribbon menu (``menu_hidden``) so they appear only inside TTTR Tools.
"""

from __future__ import annotations

import json
from pathlib import Path

_manifest_path = Path(__file__).parent / "manifest.json"
_manifest = json.loads(_manifest_path.read_text()) if _manifest_path.exists() else {}
name = _manifest.get("display_name", "TTTR:TTTR Tools")


def __getattr__(attr_name: str):
    """Lazy Qt import gate (PRD-23: no Qt import as a package side effect)."""
    if attr_name == "TttrToolboxTool":
        from .gui.tool import TttrToolboxTool as _cls

        globals()["TttrToolboxTool"] = _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {attr_name!r}")


if __name__ == "plugin":
    from .gui.tool import TttrToolboxTool

    window = TttrToolboxTool()
    window.show()


__all__ = ["TttrToolboxTool"]
