# Processing Plugin

This plugin provides a specialized browser interface for running and visualizing Processing language sketches 
within ChiSurf. Processing is a flexible software sketchbook and language for learning how to code within the context 
of the visual arts.

## Features

- Browser-based interface for running Processing sketches
- Support for interactive Processing.js sketches
- Navigation controls (URL bar, Go, Reload, Home buttons)
- Ability to load custom Processing sketches
- Security features to prevent loading non-local resources

## Overview

The Processing plugin integrates the Processing programming language into ChiSurf through a browser interface. 
Processing is a popular language designed for creating visual arts, animations, and interactive content. This plugin 
allows users to run Processing sketches directly within ChiSurf, enabling the creation of custom interactive 
visualizations for scientific data.

The plugin uses Processing.js, a JavaScript port of the Processing language, to run sketches in a browser environment. 
This approach allows for seamless integration with ChiSurf's interface while maintaining the full capabilities of the 
Processing language.

## Requirements

- Python packages:
  - PyQt5
  - PyQtWebEngine
  - chisurf core modules
- Internet connection (for loading the Processing.js library)

## Usage

1. Launch the plugin from the ChiSurf menu: Miscellaneous > Processing
2. The default example (Conway's Game of Life) will load automatically
3. Interact with the example using:
   - Space bar to pause/resume the simulation
   - Mouse clicks to activate/deactivate cells when paused
   - 'R' key to randomly reset the grid
   - 'C' key to clear the grid
4. To load a different Processing sketch:
   - Create an HTML file that loads your .pde file (similar to pr_example.html)
   - Enter the file path in the URL bar and click "Go"
   - Or use the "Home" button to return to the default example

## Example Sketch

The plugin comes with Conway's Game of Life as a default example. This cellular automaton simulation demonstrates 
how Processing can be used to create interactive visualizations. The example shows:

- Grid-based visualization
- Animation with timer-based updates
- User interaction through keyboard and mouse
- State management and rule-based simulation

## Creating Your Own Sketches

To create your own Processing sketches for use with this plugin:

1. Create a .pde file with your Processing code
2. Create an HTML file that loads your .pde file (use pr_example.html as a template)
3. Place both files in an accessible location
4. Enter the path to your HTML file in the plugin's URL bar

## Applications

The Processing plugin can be used for:
- Creating custom interactive visualizations for scientific data
- Developing educational demonstrations of scientific concepts
- Prototyping new visualization techniques
- Enhancing data presentation with interactive elements

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.