"""Time-resolved anisotropy wizard plugin.

Guided construction of a linked VV/VH global anisotropy fit. The plugin follows
the new standard: a Qt-free :mod:`.core` (IRF correction, spectrum I/O, the VV/VH
link plan) reused by both the GUI and CLI, a declarative :mod:`.gui` (AutoForm
over ``anisotropy.view.json`` with embedded interactive widgets) and a :mod:`.cli`.
"""

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Spectroscopy:Fluorescence decay:Anisotropy-Wizard"


if __name__ == "plugin":
    from .gui.tool import AnisotropyWizard

    _wizard = AnisotropyWizard()
    _wizard.show()
