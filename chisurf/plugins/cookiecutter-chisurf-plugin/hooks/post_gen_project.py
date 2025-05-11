#!/usr/bin/env python
import os

# Get the cookiecutter variables
plugin_name = "{{ cookiecutter.plugin_name }}"
plugin_display_name = "{{ cookiecutter.plugin_display_name }}"

print(f"\nPlugin '{plugin_display_name}' created successfully!")
print(f"To use your plugin, copy the '{plugin_name}' directory to the ChiSurf plugins directory.")
print("Don't forget to run create_icon.py to generate a custom icon for your plugin.")
