"""Breakout Game Plugin

Classic Breakout game with progressive difficulty, multiple brick types,
mouse/keyboard control, and particle effects.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qtpy.QtWidgets import QApplication

from chisurf.core.plugin import load_manifest
from chisurf.plugins.misc.breakout_game.breakout import Breakout

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
name = _manifest.display_name if _manifest is not None else "Tools:Miscellaneous:Breakout"

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = Breakout()
    game.show()
    sys.exit(app.exec())

if __name__ == "plugin":
    game = Breakout()
    game.show()
