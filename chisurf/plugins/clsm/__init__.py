"""
Confocal Laser Scanning Microscopy (CLSM) Image Analysis

This plugin provides a graphical interface for analyzing and processing CLSM image data 
from Time-Tagged Time-Resolved (TTTR) measurements.

Features:
- Loading and visualization of CLSM-TTTR data files
- Creation of various image representations (intensity, mean microtime)
- Interactive pixel selection with adjustable brush tools
- Region of interest (ROI) creation and management
- Generation of fluorescence decay histograms from selected pixels
- Fourier Ring Correlation (FRC) analysis for image resolution estimation
- Export of decay histograms for further analysis in ChiSurf
- Support for multiple frames and channels

The CLSM plugin is particularly useful for fluorescence lifetime imaging microscopy (FLIM) 
data analysis, allowing researchers to extract time-resolved fluorescence information from 
specific regions within microscopy images. The pixel selection tools enable precise 
isolation of structures of interest, while the integrated decay histogram generation 
provides immediate feedback on the fluorescence decay characteristics of the selected area.

The plugin supports various TTTR file formats and microscope setups, with configurable 
parameters for frame markers, line markers, and pixel settings to accommodate different 
CLSM acquisition systems.
"""

import clsmview.gui

name = "Imaging:CLSM-Draw"

import sys

import chisurf
from quest.lib.tools.dye_diffusion import TransientDecayGenerator

from PyQt5.QtWidgets import *

log = chisurf.logging.info


if __name__ == '__main__':
    app = QApplication(sys.argv)
    clsm = clsmview.gui.CLSMPixelSelect()
    clsm.show()
    sys.exit(app.exec_())

if __name__ == "plugin":
    clsm = clsmview.gui.CLSMPixelSelect()
    clsm.show()
