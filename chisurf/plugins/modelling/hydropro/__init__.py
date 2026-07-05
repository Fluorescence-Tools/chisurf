"""HydroPro — GUI front-end to the HYDROPRO / HYDRO++ hydrodynamics suite."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
    cli_entrypoint = _manifest.entrypoints.cli or ""
else:
    name = "Structure:Computation:HydroPro"
    cli_entrypoint = ""

__all__ = ["name", "cli_entrypoint"]


if __name__ == "plugin":
    from chisurf.core.plugin.registry import apply_manifest_statefulness
    from chisurf.plugins.modelling.hydropro.gui.tool import HydroProTool

    window = HydroProTool()
    if _manifest is not None:
        apply_manifest_statefulness(window, _manifest)
    window.show()
    try:
        window.raise_()
        window.activateWindow()
    except Exception:
        pass
