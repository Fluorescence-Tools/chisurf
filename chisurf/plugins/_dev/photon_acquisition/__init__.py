"""
Single-Molecule Acquisition

This plugin provides tools for acquiring single molecule data using various TCSPC hardware.
It displays fluorescence decays of user-defined channels (up to 4) and correlation curves.
Data is acquired into RAM and saved at the end of data acquisition.

Supported hardware:
- Becker & Hickl SPC 830/150/160/180 devices
- PicoQuant devices (using snAPI)

Features:
- Acquisition of time-tagged time-resolved (TTTR) data
- Display of fluorescence decays for up to 4 user-defined channels
- Display of correlation curves
- Configurable acquisition time
- RAM usage monitoring
- Data saving at the end of acquisition

The plugin is designed for single-molecule experiments where real-time monitoring of
fluorescence decays and correlation curves is essential for data quality assessment
and experimental optimization.
"""

name = "Dev:Tools:Acquisition"

import os
import time
import psutil
import numpy as np
from pathlib import Path
import logging as _py_logging


# ---------------------------------------------------------------------------
# Optional GUI / plugin imports
# ---------------------------------------------------------------------------

GUI_AVAILABLE = False

try:
    from qtpy.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QPushButton,
        QLabel,
        QSpinBox,
        QDoubleSpinBox,
        QComboBox,
        QFileDialog,
        QProgressBar,
        QCheckBox,
        QGroupBox,
        QTabWidget,
        QMessageBox,
        QMenuBar,
        QAction,
        QTextEdit,
        QDialog,
        QMdiSubWindow,
        QDockWidget,
        QToolButton,
        QSizePolicy,
        QLCDNumber,
    )
    from qtpy.QtCore import QTimer, QThread, Qt, Signal
    import pyqtgraph as pg
    GUI_AVAILABLE = True
except Exception:  # Qt stack / pyqtgraph not available – CLI-only use is still allowed
    GUI_AVAILABLE = False


if GUI_AVAILABLE:
    # Import the BH SPC wrapper from the BH-specific subpackage
    from .tcspc_devices.bh_spc import (
        DLLOperationMode,
        InitStatus,
        ParID,
        SPCMError,
        BHSPC,
        minimal_spcm_ini,
        ini_file,
        BHSPCCardSetupDialog,
    )
    # Use the new photon_sources alias for the generic TCSPCDevice factory
    from .photon_sources import TCSPCDevice

    # Import tttrlib for correlation
    import tttrlib

    # Import for saving data
    from chisurf.fio.ascii import save_xy
    from chisurf.fio.fluorescence.fcs.kristine import write_kristine

    import chisurf
    from chisurf import logging

    # Import main classes
    from .main import AcquisitionThread, SMAcquisitionManager

    # Import window classes
    from .windows import *  # noqa: F401,F403

    # Import controller classes
    from .controllers import *  # noqa: F401,F403

    # Module-level logger for this file (chisurf logging)
    logger = logging.getLogger(__name__)
else:
    # Fallback logger when GUI/plugin stack is not available
    logger = _py_logging.getLogger(__name__)




# Initialize the plugin when loaded (only when GUI stack is available)
if __name__ == "plugin" and GUI_AVAILABLE:
    import chisurf
    logger.info("Loading SM Acquisition plugin...")
    acquisition_manager = SMAcquisitionManager()
    # Note: acquisition_manager might be partially initialized if one already exists
