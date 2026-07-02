"""
Confocal Laser Scanning Microscopy (CLSM) Image Analysis.

Create representations of CLSM-TTTR data, select pixels interactively, and
export fluorescence-decay histograms for analysis in ChiSurf.

Architecture (mirrors the other ``microscopy`` plugins):

- ``core/``    — Qt-free algorithms (FRC, image representations, decay
  extraction, setup presets). Safe to import headlessly.
- ``api/``     — dataclasses, RPC contract, and orchestration functions.
- ``backend/`` — ``register_services`` wiring ``clsm.*`` RPC methods.
- ``client.py``— transport-agnostic :class:`ClsmClient` (local or ZMQ).
- ``cli/``     — ``csc clsm …`` Click commands.
- ``gui/``     — AutoForm-driven settings panels plus the interactive
  image/brush/decay/FRC canvas (registered as custom AutoForm sections).

Importing this package is intentionally light: nothing here pulls in Qt or
``tttrlib`` so the CLI and headless services stay fast. The GUI is loaded only
when the plugin is launched (``__name__ == "plugin"``) or via the manifest
``gui`` entrypoint.
"""

from __future__ import annotations

#: Plugin-menu label (``Category:Name``).
name = "Imaging:CLSM-Draw"

#: Hidden from the top-level plugin menu (launched via Imaging Tools).
menu_hidden = True

#: ``csc`` CLI registration (AST-scanned; see ``chisurf/core/cli.py``).
cli_entrypoint = "clsm=chisurf.plugins.microscopy.clsm.cli:cli"


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    from chisurf.plugins.microscopy.clsm.gui.tool import CLSMPixelSelect

    app = QApplication(sys.argv)
    win = CLSMPixelSelect()
    win.show()
    sys.exit(app.exec())

if __name__ == "plugin":
    from chisurf.gui.misc_helpers import persist_plugin_state
    from chisurf.plugins.microscopy.clsm.gui.tool import CLSMPixelSelect

    CLSMDrawWidget = persist_plugin_state("clsm_draw")(CLSMPixelSelect)
    clsm = CLSMDrawWidget()
    clsm.show()
