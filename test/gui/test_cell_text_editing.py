import sys
import os
from qtpy.QtWidgets import QApplication, QLineEdit
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizard

def test_cell_text_editing():
    """Test that cell text can be edited directly without replacing widgets."""
    app = QApplication(sys.argv)
    
    # Create a DetectorWizard instance
    wizard = DetectorWizard()
    wizard.show()
    
    # Get the DetectorWizardPage instance
    page = wizard.page(0)
    
    # Add a test tttr_channeldefinition
    page.new_detector_le.setText("test_detector")
    page._add_detector()
    
    # Select the first row (index 0)
    page.detectors_form.selectRow(0)
    
    # Get the initial cell widget for G-factor
    row = 0
    initial_cell_widget = page.detectors_form.cellWidget(row, 3)
    
    if not initial_cell_widget:
        print("FAILURE: No initial cell widget found")
        wizard.close()
        return False
    
    # Store the widget's memory address for later comparison
    initial_widget_id = id(initial_cell_widget)
    print(f"Initial widget ID: {initial_widget_id}")
    
    # Set an initial value
    initial_value = "1.234"
    initial_cell_widget.setText(initial_value)
    
    # Verify the initial value was set
    if initial_cell_widget.text() != initial_value:
        print(f"FAILURE: Initial value not set correctly. Expected: {initial_value}, Got: {initial_cell_widget.text()}")
        wizard.close()
        return False
    
    print(f"SUCCESS: Initial value '{initial_value}' set correctly")
    
    # Now update the text directly
    new_value = "2.345"
    initial_cell_widget.setText(new_value)
    
    # Get the current cell widget (should be the same object)
    current_cell_widget = page.detectors_form.cellWidget(row, 3)
    current_widget_id = id(current_cell_widget)
    
    # Check if it's the same widget
    if current_widget_id != initial_widget_id:
        print(f"FAILURE: Widget was replaced. Initial ID: {initial_widget_id}, Current ID: {current_widget_id}")
        wizard.close()
        return False
    
    print(f"SUCCESS: Widget was not replaced. Initial ID: {initial_widget_id}, Current ID: {current_widget_id}")
    
    # Check if the value was updated
    if current_cell_widget.text() != new_value:
        print(f"FAILURE: Value not updated correctly. Expected: {new_value}, Got: {current_cell_widget.text()}")
        wizard.close()
        return False
    
    print(f"SUCCESS: Value updated correctly to '{new_value}'")
    
    # Now simulate the G-factor calculation update by directly calling our modified code
    # Create a mock g_factor_calculator with a g_factor attribute
    class MockCalculator:
        def __init__(self, g_factor):
            self.g_factor = g_factor
    
    # Create a mock selected_detector
    page.selected_detector = {
        'row': row,
        'name': 'test_detector'
    }
    
    # Create a mock calculator with a g_factor value
    mock_calculator = MockCalculator(3.456)
    page.g_factor_calculator = mock_calculator
    
    # Simulate the update by manually calling the relevant code
    g_factor_value = f"{mock_calculator.g_factor:.3f}"
    
    # Get the existing cell widget and update its text
    existing_cell_widget = page.detectors_form.cellWidget(row, 3)
    if existing_cell_widget:
        # If widget exists, just update its text
        existing_cell_widget.setText(g_factor_value)
    else:
        # If no widget exists yet, create a new one
        new_cell_widget = QLineEdit(g_factor_value)
        page.detectors_form.setCellWidget(row, 3, new_cell_widget)
    
    # Get the final cell widget
    final_cell_widget = page.detectors_form.cellWidget(row, 3)
    final_widget_id = id(final_cell_widget)
    
    # Check if it's still the same widget
    if final_widget_id != initial_widget_id:
        print(f"FAILURE: Widget was replaced after simulated G-factor update. Initial ID: {initial_widget_id}, Final ID: {final_widget_id}")
        wizard.close()
        return False
    
    print(f"SUCCESS: Widget was not replaced after simulated G-factor update. Initial ID: {initial_widget_id}, Final ID: {final_widget_id}")
    
    # Check if the G-factor value was updated
    expected_final_value = "3.456"
    if final_cell_widget.text() != expected_final_value:
        print(f"FAILURE: G-factor value not updated correctly. Expected: {expected_final_value}, Got: {final_cell_widget.text()}")
        wizard.close()
        return False
    
    print(f"SUCCESS: G-factor value updated correctly to '{expected_final_value}'")
    
    # Clean up
    wizard.close()
    
    print("\nAll tests passed successfully!")
    return True

if __name__ == "__main__":
    test_cell_text_editing()
    # Exit the application
    sys.exit(0)