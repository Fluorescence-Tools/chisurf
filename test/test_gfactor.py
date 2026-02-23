import sys
from qtpy.QtWidgets import QApplication
from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage

def main():
    app = QApplication(sys.argv)
    
    # Create a DetectorWizardPage
    wizard_page = DetectorWizardPage()
    
    # Add some tttr_channeldefinition rows
    wizard_page._add_detector_row("Detector1", "0, 1", "0-2048", "1.00", "0.00", "0.00")
    wizard_page._add_detector_row("Detector2", "2, 3", "0-2048", "1.00", "0.00", "0.00")
    
    # Show the wizard page
    wizard_page.show()
    
    # Print instructions
    print("Test Instructions:")
    print("1. Select the first row in the detectors table")
    print("2. Click the 'Calculate G-Factor' button")
    print("3. Observe the debug output")
    print("4. Close the G-Factor calculator")
    print("5. Check if the G-factor was set correctly in the first row")
    print("6. Repeat steps 1-5 for the second row to compare")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()