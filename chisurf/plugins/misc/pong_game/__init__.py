"""Pong Game Plugin

Classic Pong game with CPU opponent, score tracking, and particle effects.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qtpy.QtWidgets import QApplication

from chisurf.core.plugin import load_manifest
from chisurf.plugins.misc.pong_game.pong_game import Pong

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
name = _manifest.display_name if _manifest is not None else "Tools:Miscellaneous:Pong"

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = Pong()
    game.show()
    sys.exit(app.exec())

if __name__ == "plugin":
    game = Pong()
    game.show()
