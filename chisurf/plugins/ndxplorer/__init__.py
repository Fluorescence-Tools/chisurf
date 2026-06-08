"""
ndXplorer

This plugin provides a powerful interface for analyzing and visualizing multidimensional 
fluorescence data within ChiSurf.

Features:
- Burst analysis for single-molecule fluorescence experiments
- Multiparameter fluorescence detection (MFD) analysis
- Interactive selection and filtering of burst events
- Visualization of multidimensional data through histograms and plots
- Support for FRET efficiency calculations and proximity ratio analysis
- Application to both solution-based measurements and image spectroscopy data

The ndXplorer tool is particularly useful for analyzing complex fluorescence datasets 
where multiple parameters need to be correlated, such as fluorescence intensity, 
lifetime, anisotropy, and spectral information. It provides an intuitive interface 
for exploring relationships between different fluorescence parameters.

For single-molecule experiments, ndXplorer enables detailed burst analysis with 
capabilities to select, filter, and categorize individual molecule detection events 
based on multiple criteria. The tool also supports advanced FRET analysis with 
various correction factors and calculation methods.

When working with image spectroscopy data, ndXplorer allows pixel-by-pixel analysis 
of multiparameter fluorescence information, enabling spatial correlation of 
spectroscopic properties.
"""

name = "Main:Tools:ndXplorer"

import chisurf as cs
log = cs.logging.info


if __name__ == '__main__':
    import sys
    from qtpy.QtWidgets import QApplication
    import ndxplorer
    app = QApplication(sys.argv)
    ndx = ndxplorer.NDXplorer()
    ndx.show()
    ndx.raise_()
    ndx.activateWindow()
    sys.exit(app.exec())

if __name__ == "plugin":
    import ndxplorer
    ndx = ndxplorer.NDXplorer()
    ndx.show()
    ndx.raise_()
    ndx.activateWindow()
