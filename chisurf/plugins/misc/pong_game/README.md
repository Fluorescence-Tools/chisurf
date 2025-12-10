# Pong Game Plugin

This plugin provides a classic Pong game implemented using PyQt5. It was developed primarily for testing the 
ChiSurf plugin system.

## Features

- Two-player gameplay with keyboard controls
- CPU opponent with basic AI
- Score tracking
- Adjustable game speed
- Simple and intuitive interface
- Pause functionality

## Overview

The Pong Game plugin implements the classic arcade game where players control paddles to hit a ball back and forth. 
While it provides a fun diversion, its primary purpose was to serve as a test case for the ChiSurf plugin system, 
demonstrating how interactive graphical applications can be integrated into the ChiSurf framework.

The game features a player-controlled paddle on the left side (using the Up and Down arrow keys) and a CPU-controlled 
paddle on the right side. The ball bounces between the paddles, and points are scored when the ball passes a paddle.

As a test plugin, it showcases various aspects of plugin development including:
- Integration with the ChiSurf menu system
- Creating interactive PyQt5 widgets
- Handling user input
- Implementing game logic with timers

## Requirements

- Python packages:
  - PyQt5
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Miscellaneous > Pong
2. Control your paddle (left side) using the Up and Down arrow keys
3. Try to hit the ball past the CPU-controlled paddle (right side)
4. Press P to pause/resume the game
5. The score is displayed at the top of the screen

## Applications

While primarily developed as a test plugin, the Pong Game can be used for:
- Demonstrating PyQt5 game development techniques
- Testing the plugin system's ability to handle interactive graphical applications
- Providing a simple example for plugin developers
- Offering a brief recreational break during data analysis sessions

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.