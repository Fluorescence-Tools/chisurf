"""
Pong Game Plugin

This plugin provides a classic Pong game implemented using PyQt5.
It's a simple two-player game where each player controls a paddle
to hit a ball back and forth.

Features:
- Two-player gameplay with keyboard controls
- Score tracking
- Adjustable game speed
- Simple and intuitive interface

The game serves as both a fun diversion and a demonstration of
using PyQt5 for creating interactive graphical applications.
"""

name = "Miscellaneous:Pong"

import sys
from PyQt5.QtWidgets import QApplication

from .pong_game import Pong

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = Pong()
    game.show()
    sys.exit(app.exec_())

if __name__ == "plugin":
    game = Pong()
    game.show()
