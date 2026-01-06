# -*- coding: utf-8 -*-
"""
ChiSurf Ribbon Integration

This module provides ribbon interface integration for ChiSurf main window.
It uses the ribbon implementation to create a modern ribbon interface.
"""

import os
import sys
from pathlib import Path
import json
import functools
from math import ceil

from qtpy import QtCore, QtGui, QtWidgets

import chisurf
from chisurf import logging

# Import mixins from split modules
from .ribbon_base import ChiSurfRibbonIntegration as BaseIntegration
from .ribbon_plugins import PluginMethodsMixin
from .ribbon_auto_fold import AutoFoldMethodsMixin
from .ribbon_categories import CategoryMethodsMixin
from .ribbon_file import FileCategoryMixin
from .ribbon_utils import UtilityMethodsMixin


class ChiSurfRibbonIntegration(
    BaseIntegration,
    PluginMethodsMixin,
    AutoFoldMethodsMixin,
    CategoryMethodsMixin,
    FileCategoryMixin,
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
        Ribbon style to use (ribbon uses RibbonStyle constants)
        If None, uses default style
        
    Returns
    -------
    ChiSurfRibbonIntegration or None
        The ribbon integration instance if successful, None otherwise
    """
    logging.info(f"DEBUG: Starting setup_chisurf_ribbon with ribbon_style={ribbon_style}")
    try:
        logging.info("DEBUG: Creating ChiSurfRibbonIntegration instance")
        integration = ChiSurfRibbonIntegration(main_window)
        logging.info("DEBUG: Calling setup_ribbon_interface")
        if integration.setup_ribbon_interface(ribbon_style=ribbon_style):
            logging.info("DEBUG: Ribbon setup successful, returning integration")
            return integration
        else:
            logging.warning("DEBUG: Ribbon setup failed, returning None")
            return None
    except Exception as e:
        logging.error(f"Failed to setup ChiSurf ribbon: {e}")
        import traceback
        logging.error(f"DEBUG: Exception traceback: {traceback.format_exc()}")
        return None
