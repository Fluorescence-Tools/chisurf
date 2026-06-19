"""Tetris Game Plugin

Classic single-player Tetris game with line clearing and next-piece preview.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qtpy.QtWidgets import QApplication

from chisurf.core.plugin import load_manifest
from chisurf.plugins.misc.tetris_game.tetris import Tetris

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
name = _manifest.display_name if _manifest is not None else "Tools:Miscellaneous:Tetris"

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = Tetris()
    game.show()
    sys.exit(app.exec())

if __name__ == "plugin":
    game = Tetris()
    game.show()
