import utils
import os
import sys
import unittest
from qtpy.QtWidgets import QApplication
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)
import chisurf.fio
import chisurf.tools


app = QApplication(sys.argv)


class Tests(unittest.TestCase):
    """
    Test the decay_histogram GUI
    """

    def setUp(self):
        """
        Create the GUI
        """
        self.form = chisurf.tools.tttr.decay.HistogramTTTR()

    def test_load_data(self):
        import glob
        make_decay_button = self.form.tcspc_setup_widget.pushButton

        self.assertEqual(
            len(
                self.form.curve_selector.get_data_sets()
            ),
            0
        )

        spcFileWidget = self.form.tcspc_setup_widget.spcFileWidget
        filenames = glob.glob("./test/data/tttr/BH/132/*.spc")
        file_type = "bh132"
        spcFileWidget.onLoadSample(
            event=None,
            filenames=filenames,
            file_type=file_type
        )
        QTest.mouseClick(make_decay_button, Qt.LeftButton)

        self.assertEqual(
            len(
                self.form.curve_selector.get_data_sets()
            ),
            1
        )
        print(
            self.form.curve_selector.get_data_sets()
        )

if __name__ == "__main__":
    unittest.main()
