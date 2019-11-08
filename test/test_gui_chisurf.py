import os
import sys
import unittest
from qtpy.QtWidgets import QApplication
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

import chisurf
import chisurf.widgets
import chisurf.macros
import chisurf.gui

app = QApplication(sys.argv)
cs_app = chisurf.gui.qt_app()


class Tests(unittest.TestCase):
    """
    Test the kappa2 distribution GUI
    """

    def test_setup(self):
        """
        Create the GUI
        """
        cs = chisurf.cs

        cs.comboBox_experimentSelect.setCurrentIndex(1)
        cs.comboBox_setupSelect.setCurrentIndex(0)
        filename_decay = "./test/data/tcspc/ibh_sample/Decay_577D.txt"
        filename_irf = "./test/data/tcspc/ibh_sample/Prompt.txt"

        cs.current_setup.skiprows = 11
        cs.current_setup.reading_routine = 'csv'
        cs.current_setup.is_jordi = False
        cs.current_setup.use_header = True
        cs.current_setup.matrix_columns = []
        cs.current_setup.polarization = 'vm'
        cs.current_setup.rep_rate = 10.0
        cs.current_setup.dt = 0.0141

        chisurf.macros.add_dataset(
            filename=filename_decay
        )
        chisurf.macros.add_dataset(
            filename=filename_irf
        )

        self.assertEqual(
            len(chisurf.imported_datasets),
            3
        )

        # click on decay_item
        items = chisurf.widgets.get_all_items(
            cs.dataset_selector
        )
        rect = cs.dataset_selector.visualItemRect(
            items[1]
        )
        QTest.mouseClick(
            cs.dataset_selector.viewport(),
            Qt.LeftButton,
            Qt.NoModifier,
            rect.center()
        )

        # click on add fit
        QTest.mouseClick(cs.pushButton_2, Qt.LeftButton)


if __name__ == "__main__":
    unittest.main()
