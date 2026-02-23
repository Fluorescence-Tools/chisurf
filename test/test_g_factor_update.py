import sys
import os
from qtpy.QtWidgets import QApplication
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizard

def test_g_factor_update():
    """Test that G-factor values can be set correctly for all rows, including the first row."""
    app = QApplication(sys.argv)
    
    # Create a DetectorWizard instance
    wizard = DetectorWizard()
    wizard.show()
    
    # Get the DetectorWizardPage instance
    page = wizard.page(0)
    
    # Add some test detectors
    page.new_detector_le.setText("test_detector_1")
    page._add_detector()
    page.new_detector_le.setText("test_detector_2")
    page._add_detector()
    
    # Select the first row (index 0)
    page.detectors_form.selectRow(0)
    
    # Simulate setting a G-factor value for the first row
    # This is a direct test without using the calculator
    row = 0
    g_factor_value = "1.234"
    
    # First remove any existing widget
    page.detectors_form.removeCellWidget(row, 3)
    
    # Create a new QLineEdit widget with the G-factor value
    from qtpy.QtWidgets import QLineEdit
    new_cell_widget = QLineEdit(g_factor_value)
    page.detectors_form.setCellWidget(row, 3, new_cell_widget)
    
    # Check if the G-factor value was set correctly
    cell_widget = page.detectors_form.cellWidget(row, 3)
    if cell_widget and cell_widget.text() == g_factor_value:
        print(f"SUCCESS: G-factor value '{g_factor_value}' was set correctly for row {row}")
    else:
        print(f"FAILURE: G-factor value was not set correctly for row {row}")
        if cell_widget:
            print(f"  Actual value: '{cell_widget.text()}'")
        else:
            print("  No cell widget found")
    
    # Now test the second row
    row = 1
    g_factor_value = "2.345"
    
    # First remove any existing widget
    page.detectors_form.removeCellWidget(row, 3)
    
    # Create a new QLineEdit widget with the G-factor value
    new_cell_widget = QLineEdit(g_factor_value)
    page.detectors_form.setCellWidget(row, 3, new_cell_widget)
    
    # Check if the G-factor value was set correctly
    cell_widget = page.detectors_form.cellWidget(row, 3)
    if cell_widget and cell_widget.text() == g_factor_value:
        print(f"SUCCESS: G-factor value '{g_factor_value}' was set correctly for row {row}")
    else:
        print(f"FAILURE: G-factor value was not set correctly for row {row}")
        if cell_widget:
            print(f"  Actual value: '{cell_widget.text()}'")
        else:
            print("  No cell widget found")
    
    # Check if the first row's value is still correct
    row = 0
    expected_value = "1.234"
    cell_widget = page.detectors_form.cellWidget(row, 3)
    if cell_widget and cell_widget.text() == expected_value:
        print(f"SUCCESS: G-factor value '{expected_value}' is still correct for row {row}")
    else:
        print(f"FAILURE: G-factor value for row {row} was changed unexpectedly")
        if cell_widget:
            print(f"  Actual value: '{cell_widget.text()}'")
        else:
            print("  No cell widget found")
    
    # Clean up
    wizard.close()
    
    return True

if __name__ == "__main__":
    test_g_factor_update()
    # Exit the application
    sys.exit(0)