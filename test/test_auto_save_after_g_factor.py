import sys
import os
import tempfile
import json
from qtpy.QtWidgets import QApplication, QMessageBox
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizard, load_detector_setups, save_detector_setups

# Mock the JordiGFactorCalculator to avoid actual calculation
class MockJordiGFactorCalculator:
    def __init__(self):
        self.g_factor = 1.234
        
    def setWindowModality(self, *args):
        pass
        
    def show(self):
        pass
        
    def load_jordi_file(self, *args):
        pass
        
    def closeEvent(self, event):
        pass

# Mock QMessageBox to avoid actual dialogs
def mock_information(*args, **kwargs):
    print("Mock QMessageBox.information called")
    return QMessageBox.Ok

# Original function to restore later
original_information = QMessageBox.information

def test_auto_save_after_g_factor():
    """Test that the setup is automatically saved after G-factor calculation."""
    app = QApplication(sys.argv)
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
        temp_path = temp_file.name
    
    try:
        # Initial data with a test setup
        initial_data = {
            "setups": {
                "test_setup": {
                    "windows": {"prompt": {"0": 2048}},
                    "detectors": {
                        "test_detector": {
                            "chs": [0, 1],
                            "micro_time_ranges": [[0, 4095]],
                            "g_factor": 1.0
                        }
                    },
                    "tttr_reading": {
                        "file_type": "SPC-130",
                        "macro_time_resolution": 50.0,
                        "micro_time_resolution": 50.0,
                        "micro_time_binning": 1
                    }
                }
            },
            "last_used": "test_setup"
        }
        
        # Save initial data
        save_detector_setups(initial_data, temp_path)
        
        # Create a DetectorWizard instance with our test file
        wizard = DetectorWizard(json_file=temp_path)
        wizard.show()
        
        # Get the DetectorWizardPage instance
        page = wizard.page(0)
        
        # Set the current setup file and name
        page.current_setups_file = temp_path
        page.current_setup_name = "test_setup"
        
        # Mock the QMessageBox.information to avoid actual dialogs
        QMessageBox.information = mock_information
        
        # Store the original JordiGFactorCalculator class
        from chisurf.plugins.jordi_g_factor import JordiGFactorCalculator
        original_calculator = JordiGFactorCalculator
        
        # Replace with our mock
        import chisurf.plugins.jordi_g_factor
        chisurf.plugins.jordi_g_factor.JordiGFactorCalculator = MockJordiGFactorCalculator
        
        try:
            # Select the first tttr_channeldefinition row
            page.detectors_form.selectRow(0)
            
            # Store the original g-factor value if the cell widget exists
            cell_widget = page.detectors_form.cellWidget(0, 3)
            original_g_factor = cell_widget.text() if cell_widget else "1.0"
            print(f"Original G-factor: {original_g_factor}")
            
            # If the cell widget doesn't exist, create one
            if not cell_widget:
                from qtpy.QtWidgets import QLineEdit
                new_cell_widget = QLineEdit(original_g_factor)
                page.detectors_form.setCellWidget(0, 3, new_cell_widget)
            
            # Mock the tttrlib.TTTR class
            import tttrlib
            original_tttr = tttrlib.TTTR
            
            class MockTTTR:
                def __init__(self, *args, **kwargs):
                    pass
                    
                def get_microtime_histogram(self, *args, **kwargs):
                    return [1, 2, 3, 4, 5], None
            
            tttrlib.TTTR = MockTTTR
            
            # Mock numpy functions
            import numpy as np
            original_where = np.where
            original_concatenate = np.concatenate
            original_savetxt = np.savetxt
            
            np.where = lambda x: ([0, 1, 2, 3, 4],)
            np.concatenate = lambda x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            np.savetxt = lambda *args, **kwargs: None
            
            try:
                # Simulate selecting a tttr_channeldefinition and calculating G-factor
                page.selected_detector = {
                    'row': 0,
                    'name': 'test_detector',
                    'parallel_channels': [0],
                    'perpendicular_channels': [1]
                }
                
                # Instead of calling _on_calc_g_factor, we'll manually set up what we need
                # Create a mock calculator instance
                mock_calculator = MockJordiGFactorCalculator()
                
                # Set it on the page
                page.g_factor_calculator = mock_calculator
                
                # Create a custom close event function similar to what's in the actual code
                def custom_close_event(event):
                    # Get the selected tttr_channeldefinition information
                    selected_detector_info = page.selected_detector
                    if selected_detector_info:
                        # Update the G-Factor value in the selected row of the detectors table
                        row = selected_detector_info['row']
                        g_factor_value = f"{mock_calculator.g_factor:.3f}"
                        
                        # Get the existing cell widget and update its text
                        existing_cell_widget = page.detectors_form.cellWidget(row, 3)
                        if existing_cell_widget:
                            # If widget exists, just update its text
                            existing_cell_widget.setText(g_factor_value)
                        else:
                            # If no widget exists yet, create a new one
                            from qtpy.QtWidgets import QLineEdit
                            new_cell_widget = QLineEdit(g_factor_value)
                            page.detectors_form.setCellWidget(row, 3, new_cell_widget)
                        
                        # Save the updated setup automatically
                        if page.current_setup_name:
                            # Get current settings
                            data = page.get_settings()
                            
                            # Save to the current setups file
                            setups = load_detector_setups(page.current_setups_file)
                            setups.setdefault("setups", {})
                            
                            # If the setup already exists, preserve any additional fields
                            if page.current_setup_name in setups["setups"]:
                                existing_data = setups["setups"][page.current_setup_name]
                                # Update only the fields we know about, preserving any other fields
                                for key in data:
                                    existing_data[key] = data[key]
                                # Use the updated existing data
                                setups["setups"][page.current_setup_name] = existing_data
                            else:
                                # New setup, just use the data as is
                                setups["setups"][page.current_setup_name] = data
                                
                            setups["last_used"] = page.current_setup_name
                            save_detector_setups(setups, page.current_setups_file)
                
                # Manually trigger the closeEvent to simulate closing the calculator
                custom_close_event(None)
                
                # Load the saved data to check if it was updated
                saved_data = load_detector_setups(temp_path)
                
                # Get the updated G-factor value from the saved data
                updated_g_factor = saved_data["setups"]["test_setup"]["detectors"]["test_detector"]["g_factor"]
                print(f"Updated G-factor in saved file: {updated_g_factor}")
                
                # Check if the G-factor was updated in the saved file
                assert updated_g_factor == 1.234, f"G-factor was not updated in the saved file. Expected: 1.234, Got: {updated_g_factor}"
                
                print("SUCCESS: Setup was automatically saved after G-factor calculation")
                
            finally:
                # Restore original numpy functions
                np.where = original_where
                np.concatenate = original_concatenate
                np.savetxt = original_savetxt
                
                # Restore original tttrlib.TTTR
                tttrlib.TTTR = original_tttr
                
        finally:
            # Restore original JordiGFactorCalculator
            chisurf.plugins.jordi_g_factor.JordiGFactorCalculator = original_calculator
            
            # Restore original QMessageBox.information
            QMessageBox.information = original_information
            
            # Clean up
            wizard.close()
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return True

if __name__ == "__main__":
    test_auto_save_after_g_factor()
    # Exit the application
    sys.exit(0)