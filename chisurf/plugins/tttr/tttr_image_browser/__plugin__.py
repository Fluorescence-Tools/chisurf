"""Plugin entry point for TTTR Image Browser.

This module allows loading via a plugin manager that imports ...tttr_image_browser.__plugin__.
"""

from __future__ import annotations

from chisurf.plugins.tttr.tttr_image_browser.__init__ import TTTRImageBrowser


def create_plugin_widget(parent=None):
    """Factory for the TTTRImageBrowser widget without side effects on import.

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget.

    Returns
    -------
    TTTRImageBrowser
        The created widget.
    """
    return TTTRImageBrowser(parent) if parent is not None else TTTRImageBrowser()
