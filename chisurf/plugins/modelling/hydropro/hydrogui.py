"""Backward-compatibility shim for the former monolithic ``hydrogui`` module.

The HydroPro plugin has been split into ``core`` (Qt-free logic), ``gui``
(AutoForm UI), ``cli`` and ``rpc``. This module re-exports the public names that
used to live here so existing imports keep working.
"""

from __future__ import annotations

from .core import (  # noqa: F401
    HydroProSettings,
    construct_input_file,
    parse_diffusion_coefficient,
    run_hydro,
    write_hydropro_input,
)
from .gui.dialogs import DownloadInfoDialog, OutputDialog  # noqa: F401
from .gui.tool import HydroGui, HydroProTool  # noqa: F401


def main() -> None:
    """Standalone launcher (kept for ``python -m ...hydrogui``-style use)."""
    import sys

    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)
    gui = HydroProTool()
    gui.show()
    sys.exit(app.exec_())


__all__ = [
    "HydroProTool",
    "HydroGui",
    "HydroProSettings",
    "OutputDialog",
    "DownloadInfoDialog",
    "parse_diffusion_coefficient",
    "construct_input_file",
    "write_hydropro_input",
    "run_hydro",
    "main",
]
