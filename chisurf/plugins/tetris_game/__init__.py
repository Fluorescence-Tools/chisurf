"""
Tetris Game Plugin

This plugin provides a classic single-player Tetris game implemented using PyQt5.
Shapes (Tetrominoes) fall from the top of the board and the player must rotate
and position them to complete horizontal lines.

Features:
- Seven standard Tetromino shapes with rotation
- Line clearing and score tracking
- Keyboard controls:
  • ← → : Move piece left/right
  • ↑    : Rotate piece
  • ↓    : Hard drop
  • P    : Pause/resume game
  • R    : Restart game
- Adjustable drop speed
- Next-piece preview
- Clean and intuitive interface

The game is both a challenging puzzle and a demonstration of using PyQt5 for
creating responsive, event-driven graphical applications.
"""

name = "Miscellaneous:Tetris"

import sys
from PyQt5.QtWidgets import QApplication

from .tetris import Tetris

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = Tetris()
    game.show()
    sys.exit(app.exec_())

if __name__ == "plugin":
    game = Tetris()
    game.show()
