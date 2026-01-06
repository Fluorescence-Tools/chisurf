# -*- coding: utf-8 -*-
"""
ChiSurf Ribbon Integration

This module provides ribbon interface integration for ChiSurf main window.
It uses the pyqtribbon implementation to create a modern ribbon interface.
"""

import os
import sys
from pathlib import Path
import json
import functools
from math import ceil

from PyQt5.QtCore import Qt, QSize, QTimer, QEvent, QObject
from PyQt5.QtGui import QIcon, QKeySequence, QFont
from PyQt5.QtWidgets import QAction, QMessageBox
from PyQt5 import QtWidgets

from pyqtribbon import RibbonBar

import chisurf
from chisurf import logging

# Import mixins from split modules
from .ribbon_base import ChiSurfRibbonIntegration as BaseIntegration
from .ribbon_plugins import PluginMethodsMixin
from .ribbon_auto_fold import AutoFoldMethodsMixin
from .ribbon_categories import CategoryMethodsMixin
from .ribbon_utils import UtilityMethodsMixin


class ChiSurfRibbonIntegration(
    BaseIntegration,
    PluginMethodsMixin,
    AutoFoldMethodsMixin,
    CategoryMethodsMixin,
    UtilityMethodsMixin
):
    """
    Integration class for adding ribbon interface to ChiSurf main window.

    This class provides methods to convert the existing menu/toolbar structure
    into a modern ribbon interface while maintaining all existing functionality.
    """

    def __init__(self, main_window):
        """
        Initialize ribbon integration for ChiSurf main window.

        Parameters
        ----------
        main_window : chisurf.gui.main.Main
            The ChiSurf main window instance
        """
        # Call the base class __init__ from ribbon_base.py
        super().__init__(main_window)


def setup_chisurf_ribbon(main_window, ribbon_style=None):
    """
    Convenience function to setup ribbon interface for ChiSurf.
    
    Parameters
    ----------
    main_window : chisurf.gui.main.Main
        The ChiSurf main window instance
    ribbon_style : int, optional
        Ribbon style to use (pyqtribbon uses RibbonStyle constants)
        If None, uses default style
        
    Returns
    -------
    ChiSurfRibbonIntegration or None
        The ribbon integration instance if successful, None otherwise
    """
    try:
        integration = ChiSurfRibbonIntegration(main_window)
        if integration.setup_ribbon_interface(ribbon_style=ribbon_style):
            return integration
        else:
            return None
    except Exception as e:
        logging.error(f"Failed to setup ChiSurf ribbon: {e}")
        return None
