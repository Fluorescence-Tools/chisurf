# Style Manager Plugin

This plugin provides tools for managing QSS (Qt Style Sheets) styles in ChiSurf, allowing users to customize the application's appearance.

## Features

- QSS style editor with syntax highlighting
- Creation and editing of style files
- Application of styles to the ChiSurf interface
- Management of style files in the user's settings folder
- Reset to default styles functionality
- Integration with ChiSurf's settings system

## Overview

The Style Manager plugin enables users to customize the appearance of the ChiSurf application through Qt Style Sheets (QSS). QSS is a powerful styling mechanism similar to CSS that allows for detailed customization of Qt-based user interfaces.

The plugin provides an intuitive editor with syntax highlighting for QSS properties, values, selectors, and comments. Users can create new style files, edit existing ones, and apply them to the application in real-time. The plugin also manages the storage of style files in the user's settings folder, ensuring that customizations persist across application sessions.

For users who want to experiment with different looks or revert to the original appearance, the plugin includes functionality to reset all style files to their default values.

## Requirements

- Python packages:
  - PyQt5
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > Style Manager
2. Work with existing styles:
   - Select a style file from the dropdown menu
   - Edit the QSS code in the text editor
   - Click "Save" to save your changes
   - Click "Apply" to apply the style to the application
3. Create a new style:
   - Click the "New" button
   - Enter a name for the style file
   - Edit the QSS code in the text editor
   - Save and apply as above
4. Reset styles to defaults:
   - Click the "Clear All Styles" button
   - Confirm the action in the dialog
   - The styles will be reset to the original defaults

## Applications

The Style Manager plugin can be used for:
- Customizing the appearance of ChiSurf to match personal preferences
- Creating high-contrast themes for better visibility
- Designing specialized themes for different types of analysis
- Learning QSS styling techniques
- Testing UI component appearance with different styles
- Creating consistent visual themes across multiple installations

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.