"""FPS JSON Editor for Accessible Volume Calculations."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
    cli_entrypoint = _manifest.entrypoints.cli or ""
else:
    name = "Structure:FRET:FPS JSON Editor"
    cli_entrypoint = ""

__all__ = ["cli_entrypoint", "name"]


if __name__ == "plugin":
    from chisurf.core.plugin.registry import apply_manifest_statefulness
    from chisurf.plugins.modelling.fps_json_editor.gui.tool import FpsJsonEditorTool

    window = FpsJsonEditorTool()
    if _manifest is not None:
        apply_manifest_statefulness(window, _manifest)
    window.show()
    try:
        window.raise_()
        window.activateWindow()
    except Exception:
        pass
