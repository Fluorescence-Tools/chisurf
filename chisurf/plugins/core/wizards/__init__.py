"""Wizards hub plugin.

A two-panel launcher: pick a wizard on the left, use it embedded on the right.
The catalogue of wizards is Qt-free data in :mod:`.core.registry`; the GUI host
(:mod:`.gui.tool`) resolves and embeds each wizard widget lazily.
"""

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Main:Tools:Wizards"


if __name__ == "plugin":
    from .gui.tool import WizardHub

    _hub = WizardHub()
    _hub.show()
