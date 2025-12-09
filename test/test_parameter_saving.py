"""
Test script to verify that all parameters are correctly collected by the get_burst_selection_parameters method.

This script:
1. Initializes the WizardTTTRPhotonFilter class
2. Sets up some test parameters
3. Calls get_burst_selection_parameters
4. Verifies that all parameters are included in the returned dictionary
"""

import sys
from qtpy import QtWidgets, QtCore

# Import the module to test
from chisurf.gui.widgets.wizard.tttr_photon_filter import WizardTTTRPhotonFilter

def test_parameter_collection():
    """Test that all parameters are correctly collected."""
    
    # Create a Qt application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create empty windows and detectors dictionaries as required by the constructor
    windows = {}
    detectors = {}
    
    # Create the widget with required arguments
    widget = WizardTTTRPhotonFilter(windows=windows, detectors=detectors)
    
    # Set some test parameters
    widget.spinBox.setValue(100)  # photon_threshold
    widget.doubleSpinBox.setValue(2.0)  # count_rate_window_ms
    widget.checkBox.setChecked(True)  # invert_filter
    widget.checkBox_4.setChecked(True)  # filter_active
    widget.checkBox_5.setChecked(True)  # use_gap_fill
    widget.spinBox_7.setValue(5)  # max_gap
    widget.doubleSpinBox_4.setValue(1.5)  # trace_bin_width
    widget.spinBox_6.setValue(50)  # number_of_burst_bins
    widget.lineEdit_4.setText("1,2,3")  # channels
    widget.spinBox_5.setValue(8)  # decay_coarse
    
    # Get the parameters
    params = widget.get_burst_selection_parameters()
    
    # Verify that all parameters are included
    print("Checking collected parameters...")
    assert params["photon_threshold"] == 100, "Incorrect photon_threshold"
    assert params["count_rate_window_ms"] == 2.0, "Incorrect count_rate_window_ms"
    assert params["invert_filter"] == True, "Incorrect invert_filter"
    assert params["filter_active"] == True, "Incorrect filter_active"
    assert params["use_gap_fill"] == True, "Incorrect use_gap_fill"
    assert params["max_gap"] == 5, "Incorrect max_gap"
    assert params["trace_bin_width"] == 1.5, "Incorrect trace_bin_width"
    assert params["number_of_burst_bins"] == 50, "Incorrect number_of_burst_bins"
    assert params["channels"] == [1, 2, 3], "Incorrect channels"
    assert params["decay_coarse"] == 8, "Incorrect decay_coarse"
    
    print("All parameters were correctly collected!")
    return True

if __name__ == "__main__":
    try:
        success = test_parameter_collection()
        if success:
            print("\nTest completed successfully!")
        else:
            print("\nTest failed!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")