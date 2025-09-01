# Tetris Game Plugin

This plugin provides a classic single-player Tetris game implemented using PyQt5.

## Features

- Seven standard Tetromino shapes with rotation
- Line clearing and score tracking
- Keyboard controls for piece movement and game actions
- Adjustable drop speed
- Next-piece preview
- Clean and intuitive interface
- Pause and restart functionality

## Overview

The Tetris Game plugin implements the classic puzzle game where shapes (Tetrominoes) fall from the top of the 
board and the player must rotate and position them to complete horizontal lines. When a line is completed, 
it disappears and the player earns points.

The game features all seven standard Tetromino shapes (I, J, L, O, S, T, Z) with proper rotation mechanics. Players 
can move pieces left and right, rotate them, and perform hard drops to place them quickly. The game includes a 
preview of the next piece, allowing for strategic planning.

This plugin serves as both an entertaining diversion and a demonstration of using PyQt5 for creating responsive, 
event-driven graphical applications within the ChiSurf framework.

## Requirements

- Python packages:
  - PyQt5
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Miscellaneous > Tetris
2. Control the falling pieces using the keyboard:
   - Left/Right arrow keys: Move piece horizontally
   - Up arrow key: Rotate piece
   - Down arrow key: Hard drop (instantly place the piece)
   - P key: Pause/resume the game
   - R key: Restart the game
3. Complete horizontal lines to score points
4. The game ends when the stack of pieces reaches the top of the board

## Applications

While primarily developed as a recreational plugin, the Tetris Game can be used for:
- Demonstrating PyQt5 game development techniques
- Providing a simple example for plugin developers
- Offering a brief recreational break during data analysis sessions
- Showcasing event-driven programming concepts
- Testing the responsiveness of the ChiSurf plugin system

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.