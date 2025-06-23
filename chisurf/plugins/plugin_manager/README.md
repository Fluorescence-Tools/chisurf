# Plugin Manager Plugin

This plugin allows you to manage all installed plugins in ChiSurf.

## Features

- View all available plugins
- Enable or disable plugins
- View plugin descriptions
- Import plugins from external directories
- Rename plugins
- Configure plugin order in menus
- Add plugins to the toolbar for quick access
- Filter plugins by name
- Separate display of built-in and user plugins

## Overview

The Plugin Manager provides a comprehensive interface for managing ChiSurf plugins. It allows users to view, enable, 
disable, and configure all installed plugins through an intuitive graphical interface.

The plugin displays a list of all available plugins with their icons and status indicators. When a plugin is selected, 
its details are shown, including its name, path, and description. Users can enable or disable plugins, change their 
order in menus, and add them to the toolbar for quick access.

One of the key features is the ability to import plugins from external directories, making it easy to extend ChiSurf's 
functionality with custom or third-party plugins. The plugin manager handles security elevation automatically when 
needed for system-protected directories on Windows, macOS, and Linux.

## Requirements

- Python packages:
  - PyQt5
  - pyyaml
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > Plugin Manager
2. View and manage plugins:
   - Browse the list of available plugins
   - Select a plugin to view its details
   - Enable or disable plugins using the checkbox
   - Add plugins to the toolbar using the "Show in toolbar" checkbox
   - Reorder plugins using the "Move Up" and "Move Down" buttons
3. Import external plugins:
   - Click the "Import Plugin" button
   - Select a plugin directory
   - Confirm the import
4. Rename plugins:
   - Select a plugin
   - Click the "Rename" button
   - Enter a new name
5. Filter plugins:
   - Enter text in the filter box to find specific plugins
6. Save your changes:
   - Click "Save Settings" to apply your changes
   - Restart ChiSurf for some changes to take effect

## Applications

- Customizing the ChiSurf interface by enabling/disabling plugins
- Extending ChiSurf with custom or third-party plugins
- Organizing plugins in the menu system
- Creating a personalized toolbar with frequently used plugins
- Managing plugin visibility and accessibility

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.