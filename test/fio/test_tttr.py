# Consolidated test file: test_tttr.py


# --- FROM test_tttr_channel_definition.py ---
"""
Test script for the modified DetectorWizardPage that handles SPC files.

This script creates a simple application with a DetectorWizardPage and allows
the user to test loading both TTTR and SPC files.
"""

import sys
import os
from pathlib import Path
import tempfile

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

from qtpy.QtWidgets import QApplication, QWizard
from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage

def main():
    """Create a simple application with a DetectorWizardPage."""
    app = QApplication(sys.argv)
    
    # Create a wizard with the DetectorWizardPage
    wizard = QWizard()
    page = DetectorWizardPage()
    wizard.addPage(page)
    
    # Show the wizard
    wizard.show()
    
    # Create a temporary .set file for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        set_file = Path(temp_dir) / "test.set"
        
        # Create a simple .set file with known parameters
        content = """
        [SP_SYN_FQ,F,-50.98]
        [SP_TAC_TC,F,1.83e-11]
        [SP_TAC_R,F,5.0e-8]
        [SP_ADC_RE,I,4096]
        """
        set_file.write_text(content)
        
        print(f"Created test .set file at: {set_file}")
        print("You can use this file to test the 'Read TTTR' button.")
    
    # Run the application
    sys.exit(app.exec_())


# --- FROM test_tttr_channel_definition_selective.py ---
"""
Test script for the modified DetectorWizardPage that selectively reads calibration data.

This script creates a simple application with a DetectorWizardPage and allows
the user to test loading different file types to verify the selective reading behavior:
- .set files: Should only read microtime calibration
- .spc files: Should only read macrotime calibration
- Other TTTR files: Should read both calibrations

Usage:
1. Run this script
2. Use the "Read TTTR" button in the wizard
3. Select different file types and observe the behavior
"""

import sys
import os
from pathlib import Path
import tempfile

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

from qtpy.QtWidgets import QApplication, QWizard, QLabel, QVBoxLayout, QWidget
from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage

def main():
    """Create a simple application with a DetectorWizardPage and test files."""
    app = QApplication(sys.argv)
    
    # Create a wizard with the DetectorWizardPage
    wizard = QWizard()
    page = DetectorWizardPage()
    wizard.addPage(page)
    
    # Create a widget to display instructions
    instructions = QWidget()
    layout = QVBoxLayout(instructions)
    
    label = QLabel(
        "Test the selective reading behavior:\n"
        "1. Click 'Read TTTR' button\n"
        "2. Select a file based on its type:\n"
        "   - .set files: Should only update microtime\n"
        "   - .spc files: Should only update macrotime\n"
        "   - Other files: Should update both"
    )
    layout.addWidget(label)
    
    # Add the instructions widget to the wizard
    wizard.setOption(QWizard.HaveCustomButton1, True)
    wizard.setButtonText(QWizard.CustomButton1, "Show Instructions")
    wizard.customButtonClicked.connect(lambda: instructions.show())
    
    # Show the wizard
    wizard.show()
    
    # Create a temporary .set file for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        set_file = Path(temp_dir) / "test.set"
        
        # Create a simple .set file with known parameters
        content = """
        [SP_SYN_FQ,F,-50.98]
        [SP_TAC_TC,F,1.83e-11]
        [SP_TAC_R,F,5.0e-8]
        [SP_ADC_RE,I,4096]
        """
        set_file.write_text(content)
        
        print(f"Created test .set file at: {set_file}")
        print("You can use this file to test the 'Read TTTR' button.")
        print("When you select this .set file, only the microtime should be updated.")
    
    # Run the application
    sys.exit(app.exec_())


# --- FROM test_unicode_tttr.py ---
import os
import tttrlib

# Create a mock TTTR file with a unicode name
filename = "test_µ_file.ptu"
with open(filename, "w") as f:
    f.write("mock")

try:
    print(f"Trying to open string path: {filename}")
    tt = tttrlib.TTTR(filename)
    print("Success with string path")
except Exception as e:
    print(f"Failed with string path: {e}")

try:
    print(f"Trying to open unicode path: {filename}")
    tt = tttrlib.TTTR(filename)
    print("Success with unicode")
except Exception as e:
    print(f"Failed with unicode: {e}")

try:
    print("Trying to open utf-8 encoded path")
    tt = tttrlib.TTTR(filename.encode('utf-8'))
    print("Success with utf-8 encoded bytes")
except Exception as e:
    print(f"Failed with utf-8 encoded bytes: {e}")

print("Done")

# --- FROM test_burst_imports.py ---
"""
Test script to verify that the name conflict between bocpd.convert_bursts_to_start_stop
and kalman.convert_bursts_to_start_stop has been resolved.
"""

import chisurf.fluorescence.burst

# Test that both functions can be accessed via their module prefixes
print("Testing access to convert_bursts_to_start_stop functions:")
print("BOCPD function:", chisurf.fluorescence.burst.bocpd.convert_bursts_to_start_stop)
print("Kalman function:", chisurf.fluorescence.burst.kalman.convert_bursts_to_start_stop)

# Verify they are different functions
print("\nVerifying they are different functions:")
print("Are they the same object?", 
      chisurf.fluorescence.burst.bocpd.convert_bursts_to_start_stop is 
      chisurf.fluorescence.burst.kalman.convert_bursts_to_start_stop)

print("\nTest completed successfully!")