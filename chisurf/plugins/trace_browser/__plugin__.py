"""
Plugin entry point for Trace Browser.
This module allows loading via a plugin manager that imports ...trace_browser.__plugin__.
"""
from chisurf.plugins.trace_browser.__init__ import TraceBrowser

# Note: Do not execute UI code on import. The Plugin Manager may import this module
# during discovery. Creating or showing windows here would cause unwanted popups.


def create_plugin_widget(parent=None):
    """Factory for the TraceBrowser widget without side effects on import."""
    return TraceBrowser() if parent is None else TraceBrowser(parent)
