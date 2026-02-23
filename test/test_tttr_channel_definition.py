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

if __name__ == "__main__":
    main()