"""
Test script to verify that the save_burst_params_button only triggers the save_burst_selection_parameters method once.

This script imports the necessary modules and creates a simple UI to test the button click.
"""

import sys
from qtpy import QtWidgets, QtCore
import chisurf.gui.widgets.wizard.tttr_photonfilter.tttr_photon_filter as tttr_photon_filter

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Create empty windows and detectors dictionaries as required by the constructor
    windows = {}
    detectors = {}
    
    # Create the widget with required arguments
    widget = tttr_photon_filter.WizardTTTRPhotonFilter(windows=windows, detectors=detectors)
    
    # Show the widget
    widget.show()
    
    print("Test application started. Click the 'Save Parameters' button to test.")
    print("If the message appears only once, the fix is working correctly.")
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()