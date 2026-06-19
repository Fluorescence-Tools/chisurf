"""TTTR Time-Window Splitter.

Split TTTR files into fixed-duration time windows and save them as
BID files (start, stop photon indices).
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
    cli_entrypoint = _manifest.entrypoints.cli or ""
else:
    name = "Tools:Converter:TTTR→Time-Window BIDs"
    cli_entrypoint = ""

from .gui.tool import TTTRTimeWindowTool

if __name__ == "plugin":
    w = TTTRTimeWindowTool()
    w.show()

if __name__ == "__main__":
    import sys

    from qtpy import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    w = TTTRTimeWindowTool()
    w.show()
    sys.exit(app.exec_())

__all__ = ["TTTRTimeWindowTool"]
