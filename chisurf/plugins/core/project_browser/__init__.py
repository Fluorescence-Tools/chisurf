from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Tools:Open Project"

if __name__ == "plugin":
    from chisurf.plugins.core.project_browser.gui.tool import ProjectBrowserTool
    window = ProjectBrowserTool()
    window.show()
