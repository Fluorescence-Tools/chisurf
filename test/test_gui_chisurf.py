import utils
import os
import sys
import unittest
from qtpy.QtWidgets import QApplication
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../chisurf/')
)
utils.set_search_paths(TOPDIR)
import chisurf.gui


app = QApplication(sys.argv)


class Tests(unittest.TestCase):
    """
    Test the kappa2 distribution GUI
    """

    def test_setup(self):
        """
        Create the GUI
        """
        self.cs_app = chisurf.gui.gui()
        self.cs_app.exit()


if __name__ == "__main__":
    unittest.main()
