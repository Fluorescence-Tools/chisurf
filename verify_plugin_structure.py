import sys
import os
import unittest
from pathlib import Path

# Mocking Qt and other dependencies if necessary, but here we just want to check imports
# and basic instantiation which might require a QApplication if we go deep.
# However, the goal is to check if 'TwoDFCSPlugin' can be imported and instantiated.

# Add the parent directory of 'chisurf' to sys.path
sys.path.append(os.getcwd())

class TestPluginStructure(unittest.TestCase):
    def test_imports(self):
        print("Testing plugin imports...")
        try:
            from chisurf.plugins.fcs.fcs_2d import TwoDFCSPlugin
            print("Successfully imported TwoDFCSPlugin from chisurf.plugins.fcs.fcs_2d")
        except ImportError as e:
            self.fail(f"Failed to import TwoDFCSPlugin: {e}")

    def test_gui_package(self):
        print("Testing gui package structure...")
        try:
            from chisurf.plugins.fcs.fcs_2d.gui import TwoDFCSWizard
            from chisurf.plugins.fcs.fcs_2d.gui.worker import TwoDFCSWorker
            from chisurf.plugins.fcs.fcs_2d.gui.tabs.data_tab import DataTab
            print("Successfully imported gui components")
        except ImportError as e:
            self.fail(f"Failed to import gui components: {e}")

    def test_fit_package(self):
        print("Testing fit package structure...")
        try:
            from chisurf.plugins.fcs.fcs_2d.fit import TwoDMEMFitter, OneDMEMFitter
            print("Successfully imported fit components")
        except ImportError as e:
            self.fail(f"Failed to import fit components: {e}")

if __name__ == "__main__":
    unittest.main()
