"""ChiSurf Help Plugin.

Documentation browser and help resource viewer for ChiSurf.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
    cli_entrypoint = _manifest.entrypoints.cli or ""
else:
    name = "Help:Documentation"
    cli_entrypoint = "help=chisurf.plugins.core.help.cli.main:cli"

icon = "📖"

from chisurf.plugins.core.help.gui.tool import HelpWidget  # noqa: E402

__all__ = ["HelpWidget"]

# Legacy entry point support — when loaded via the old plugin mechanism
if __name__ == "plugin":
    window = HelpWidget()
    window.show()
