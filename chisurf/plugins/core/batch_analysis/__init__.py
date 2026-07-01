"""Batch-Analysis plugin.

Apply one pre-optimised *template* fit to many datasets/files at once. The
plugin is split into a Qt-free :mod:`.core` (the batch runner + exporters, driven
by both the GUI and the CLI), a declarative :mod:`.gui` (AutoForm over
``batch.view.json``) and a :mod:`.cli`.
"""

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Main:Tools:Batch-Analysis"


if __name__ == "plugin":
    from .gui.tool import BatchProcessingWizard

    _wizard = BatchProcessingWizard()
    _wizard.show()
