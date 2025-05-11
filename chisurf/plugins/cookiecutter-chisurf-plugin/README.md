# ChiSurf Plugin Template

This is a [cookiecutter](https://github.com/cookiecutter/cookiecutter) template for creating new plugins for ChiSurf.

## Requirements

- [cookiecutter](https://github.com/cookiecutter/cookiecutter)
- Python 3.6+
- PyQt5

## Usage

1. Install cookiecutter if you haven't already:

```bash
pip install cookiecutter
```

2. Generate a new ChiSurf plugin project:

```bash
cookiecutter path/to/cookiecutter-chisurf-plugin
```

3. Answer the prompts to configure your plugin:

- `plugin_name`: The name of your plugin directory (lowercase, no spaces)
- `plugin_display_name`: The display name of your plugin (will appear in the menu)
- `plugin_category`: The category of your plugin (e.g., Tools, Structure, Analysis)
- `plugin_description`: A brief description of what your plugin does
- `author_name`: Your name
- `author_email`: Your email
- `widget_class_name`: The name of the main widget class for your plugin

4. Navigate to your new plugin directory:

```bash
cd your_plugin_name
```

5. Customize your plugin:
   - Edit `__init__.py` to implement your plugin's functionality
   - Run `create_icon.py` to generate a custom icon for your plugin

6. Install your plugin:
   - Copy your plugin directory to the ChiSurf plugins directory
   - Restart ChiSurf

## Template Structure

- `__init__.py`: The main plugin file with the plugin name, description, and widget class
- `create_icon.py`: A script to create a custom icon for your plugin

## License

This template is licensed under the same license as ChiSurf.