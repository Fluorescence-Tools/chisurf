# ChiSurf Plugins

This directory contains plugins for ChiSurf. Plugins extend the functionality of ChiSurf with additional tools, 
visualizations, and analysis capabilities.

## Plugin Structure

Each plugin is a Python package in the `chisurf.plugins` namespace. A typical plugin has the following structure:

```
plugin_name/
├── __init__.py      # Main plugin code
├── create_icon.py   # Script to create the plugin icon
└── icon.png         # Plugin icon
```

The `__init__.py` file must define a variable called `name` that specifies the plugin's name and category. For example:

```python
name = "Category:Plugin Name"
```

The plugin's code should be executed when the module is loaded with `__name__ == "plugin"`. For example:

```python
if __name__ == "plugin":
    # Create and show the plugin's main window
    window = MyPluginWidget()
    window.show()
```

## Creating a Plugin

### Using Cookiecutter (Recommended)

The easiest way to create a new plugin is to use the provided [cookiecutter](https://github.com/cookiecutter/cookiecutter) template.

1. Install cookiecutter:

```bash
pip install cookiecutter
```

2. Generate a new plugin project:

```bash
cookiecutter path/to/chisurf/plugins/cookiecutter-chisurf-plugin
```

3. Follow the prompts to configure your plugin.

4. Customize your plugin by editing the generated files.

5. Install your plugin by copying the generated directory to one of these locations:
   - The ChiSurf built-in plugins directory (requires admin/root access)
   - The user plugins directory: `~/.chisurf/plugins/` (recommended for custom plugins)

### Manual Creation

If you prefer to create a plugin manually:

1. Create a new directory with your plugin's name in one of these locations:
   - The ChiSurf built-in plugins directory: `chisurf/plugins/` (requires admin/root access)
   - The user plugins directory: `~/.chisurf/plugins/` (recommended for custom plugins)

2. Create an `__init__.py` file with the following structure:

```python
import sys
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout

# Define the plugin name - this will appear in the Plugins menu
name = "Category:Plugin Name"

"""
Plugin Name

A brief description of what your plugin does.

Author: Your Name <your.email@example.com>
Year: 2023
"""

class MyPluginWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plugin Name")
        self.resize(800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Add your plugin UI components here

    # Add your plugin methods here

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the plugin widget class
    window = MyPluginWidget()
    # Show the window
    window.show()
```

3. Copy the `create_icon_template.py` file from the plugins directory to your plugin directory as `create_icon.py`.

4. Customize the `create_icon.py` file to create an icon that represents your plugin's functionality.

5. Run the `create_icon.py` script to generate the `icon.png` file.

## Plugin Categories

Plugins are organized into categories in the ChiSurf menu. Common categories include:

- Tools
- Structure
- Analysis
- Visualization
- Processing
- Import/Export

Choose an appropriate category for your plugin based on its functionality.

## Custom User Plugins

ChiSurf supports loading plugins from a user-specific directory located at `~/.chisurf/plugins/`. This allows you to:

1. Create and use your own plugins without modifying the main ChiSurf installation
2. Keep your custom plugins when updating ChiSurf
3. Share plugins with other users without requiring them to modify their ChiSurf installation

To use custom user plugins:

1. Create the directory `~/.chisurf/plugins/` if it doesn't exist already (ChiSurf will create it automatically on startup)
2. Place your plugin directories in this location, following the same structure as built-in plugins
3. Restart ChiSurf to load the new plugins

Custom user plugins will appear in the Plugin Manager with a "[user]" indicator to distinguish them from built-in plugins.

## Plugin Manager

ChiSurf includes a Plugin Manager that allows you to enable or disable plugins, and configure how they appear in the menu system. You can access the Plugin Manager from the Tools menu.

## Contributing

If you've created a useful plugin, consider contributing it to the ChiSurf project. Contact the ChiSurf development team for more information.
