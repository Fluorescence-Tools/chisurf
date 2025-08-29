"""
Plugin entry point for TTTR Image Browser.
This module allows loading via a plugin manager that imports ...tttr_image_browser.__plugin__.
"""
from chisurf.plugins.tttr_image_browser.__init__ import TTTRImageBrowser

# Note: Do not execute UI code on import. The Plugin Manager may import this module
# during discovery. Creating or showing windows here would cause unwanted popups.


def create_plugin_widget(parent=None):
    """Factory for the TTTRImageBrowser widget without side effects on import."""
    return TTTRImageBrowser() if parent is None else TTTRImageBrowser(parent)
