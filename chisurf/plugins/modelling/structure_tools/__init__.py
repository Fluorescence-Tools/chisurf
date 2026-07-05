"""Structure Tools — combined NavigationPanelTool plugin.

A single unified window (left navigation, right panel) that hosts the existing
structure-modelling tools as embedded panels:

- FPS JSON Editor        (fps_json_editor)
- FRET Docking & Screening (fret)
- Kappa2 Distribution    (kappa2_dist)
- (separator)
- QuEst                  (quenching_estimator)
- HydroPro               (hydropro)
- (separator)
- Trajectory Tools       (traj_tools)

Each sub-tool is embedded as-is; this plugin only provides the shared shell.
"""

from __future__ import annotations

from pathlib import Path
import json

_manifest_path = Path(__file__).parent / "manifest.json"
_manifest = json.loads(_manifest_path.read_text()) if _manifest_path.exists() else {}
name = _manifest.get("display_name", "Structure:Structure Tools")


def __getattr__(attr_name: str):
    """Lazy Qt import gate."""
    if attr_name == "StructureToolsTool":
        from .gui.tool import StructureToolsTool as _cls
        globals()["StructureToolsTool"] = _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {attr_name!r}")


if __name__ == "plugin":
    from .gui.tool import StructureToolsTool
    window = StructureToolsTool()
    window.show()


__all__ = ["StructureToolsTool"]
