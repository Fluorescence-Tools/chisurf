"""Plugin entry point for Burst Browser.
This allows loading via a plugin manager that imports ...burst_browser.__plugin__.
"""
from chisurf.plugins.burst_browser import BurstBrowserWidget


def create_plugin_widget(parent=None):
    """Factory for the BurstBrowser widget without side effects on import."""
    return BurstBrowserWidget(parent=parent)
