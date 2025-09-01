# Model Manager Plugin

This plugin allows you to manage all models and experiments in ChiSurf.

## Features

- View all available models and experiments
- Enable or disable models and experiments
- View model and experiment descriptions
- Configure which models and experiments are available in ChiSurf
- Save configuration settings to the ChiSurf settings file
- Refresh model and experiment lists

## Overview

The Model Manager provides a convenient interface for configuring which models and experiments are available in ChiSurf. 
This is particularly useful for customizing the ChiSurf environment to focus on specific types of analysis or to hide 
models and experiments that are not relevant to your work.

The plugin organizes models and experiments in separate tabs, making it easy to browse through the available options 
and view their descriptions. Each model or experiment can be individually enabled or disabled, allowing for fine-grained 
control over the ChiSurf environment.

Settings are saved to the ChiSurf settings file, ensuring that your configuration persists across sessions.

## Requirements

- Python packages:
  - PyQt5
  - pyyaml
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > Model Manager
2. Browse available models and experiments:
   - The Models tab shows all available fitting models
   - The Experiments tab shows all available experiment types
3. View details for a model or experiment:
   - Select an item from the list to view its description
   - Details appear in the right panel
4. Enable or disable models and experiments:
   - Check or uncheck the "Disabled" checkbox for each item
   - Disabled items will not appear in ChiSurf menus and dialogs
5. Save your settings:
   - Click the "Save Settings" button to persist your configuration
   - Settings are saved to the ChiSurf settings file shown at the top of the window
6. Refresh the lists:
   - Click the "Refresh Lists" button to update the model and experiment lists
   - This is useful after installing new plugins or models

## Applications

- Customizing the ChiSurf environment for specific analysis workflows
- Simplifying the user interface by hiding unused models and experiments
- Managing which models and experiments are available to users
- Organizing models and experiments for different types of analysis
- Troubleshooting by selectively enabling or disabling specific components

## Benefits

- Streamlines the ChiSurf interface by hiding unused functionality
- Provides a central location for managing all available models and experiments
- Makes it easy to see which models and experiments are available
- Allows for quick enabling or disabling of components without editing configuration files manually
- Helps users focus on relevant tools for their specific analysis needs

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.