"""
Breakout Game Plugin

Classic Breakout game with progressive difficulty, multiple brick types,
mouse/keyboard control, and particle effects.
"""

name = "Tools:Miscellaneous:Breakout"

import sys
from qtpy.QtWidgets import QApplication

from chisurf.plugins.misc.breakout_game.breakout import Breakout

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = Breakout()
    game.show()
    sys.exit(app.exec())

if __name__ == "plugin":
    game = Breakout()
    game.show()
