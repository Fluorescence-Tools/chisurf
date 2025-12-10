"""
Settings module for lltf.

This module provides functions for loading and managing settings.
"""

import os
import yaml
import pkg_resources

PACKAGE_NAME = "chisurf.plugins.lltf.core"

def get_default_settings():
    """
    Get the default settings for lltf.

    Returns
    -------
    dict
        Default settings
    """
    # Get the path to the default settings file bundled with the plugin
    settings_file = pkg_resources.resource_filename(PACKAGE_NAME, 'settings/lifetime_settings.yml')

    # Load the settings
    with open(settings_file, 'r') as f:
        settings = yaml.safe_load(f)

    return settings

def load_settings(filename):
    """
    Load settings from a file.

    Parameters
    ----------
    filename : str
        Path to the settings file

    Returns
    -------
    dict
        Settings
    """
    # Load the settings
    with open(filename, 'r') as f:
        settings = yaml.safe_load(f)

    return settings

def save_settings(settings, filename):
    """
    Save settings to a file.

    Parameters
    ----------
    settings : dict
        Settings to save
    filename : str
        Path to the settings file
    """
    # Save the settings
    with open(filename, 'w') as f:
        yaml.dump(settings, f, default_flow_style=False)
