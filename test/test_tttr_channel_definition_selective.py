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
from chisurf.gui.widgets.wizard.tttr_channel_definition import DetectorWizardPage

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

if __name__ == "__main__":
    main()