from __future__ import annotations

from pathlib import Path
from chisurf.core.plugin import load_manifest

# Load manifest as source of truth
_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Setup:User Editor"

# When the plugin is loaded, this code will be executed
if __name__ == "plugin":
    from chisurf.plugins.core.user_editor.gui.tool import UserEditorWidget
    # Create an instance of the UserEditorWidget class
    window = UserEditorWidget()
    # Show the window
    window.show()
