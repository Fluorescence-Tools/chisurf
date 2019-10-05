import utils
import os
import sys
import unittest
from qtpy.QtWidgets import QApplication
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

TOPDIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
utils.set_search_paths(TOPDIR)
import mfm.io
import mfm.tools


app = QApplication(sys.argv)


class Tests(unittest.TestCase):
    """
    Test the decay_histogram GUI
    """

    def setUp(self):
        """
        Create the GUI
        """
        self.form = mfm.tools.tttr.decay_histogram.HistogramTTTR()

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
        filenames = glob.glob("./data/tttr/BH/132/*.spc")
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


    """
    def setFormToZero(self):
        '''Set all ingredients to zero in preparation for setting just one
        to a nonzero value.
        '''
        self.form.ui.tequilaScrollBar.setValue(0)
        self.form.ui.tripleSecSpinBox.setValue(0)
        self.form.ui.limeJuiceLineEdit.setText("0.0")
        self.form.ui.iceHorizontalSlider.setValue(0)

    def test_tequilaScrollBar(self):
        '''Test the tequila scroll bar'''
        self.setFormToZero()

        # Test the maximum.  This one goes to 11.
        self.form.ui.tequilaScrollBar.setValue(12)
        self.assertEqual(self.form.ui.tequilaScrollBar.value(), 11)

        # Test the minimum of zero.
        self.form.ui.tequilaScrollBar.setValue(-1)
        self.assertEqual(self.form.ui.tequilaScrollBar.value(), 0)

        self.form.ui.tequilaScrollBar.setValue(5)

        # Push OK with the left mouse button
        okWidget = self.form.ui.buttonBox.button(self.form.ui.buttonBox.Ok)
        QTest.mouseClick(okWidget, Qt.LeftButton)
        self.assertEqual(self.form.jiggers, 5)

    def test_tripleSecSpinBox(self):
        '''Test the triple sec spin box.
        Testing the minimum and maximum is left as an exercise for the reader.
        '''
        self.setFormToZero()
        self.form.ui.tripleSecSpinBox.setValue(2)

        # Push OK with the left mouse button
        okWidget = self.form.ui.buttonBox.button(self.form.ui.buttonBox.Ok)
        QTest.mouseClick(okWidget, Qt.LeftButton)
        self.assertEqual(self.form.jiggers, 2)

    def test_limeJuiceLineEdit(self):
        '''Test the lime juice line edit.
        Testing the minimum and maximum is left as an exercise for the reader.
        '''
        self.setFormToZero()
        # Clear and then type "3.5" into the lineEdit widget
        self.form.ui.limeJuiceLineEdit.clear()
        QTest.keyClicks(self.form.ui.limeJuiceLineEdit, "3.5")

        # Push OK with the left mouse button
        okWidget = self.form.ui.buttonBox.button(self.form.ui.buttonBox.Ok)
        QTest.mouseClick(okWidget, Qt.LeftButton)
        self.assertEqual(self.form.jiggers, 3.5)

    def test_iceHorizontalSlider(self):
        '''Test the ice slider.
        Testing the minimum and maximum is left as an exercise for the reader.
        '''
        self.setFormToZero()
        self.form.ui.iceHorizontalSlider.setValue(4)

        # Push OK with the left mouse button
        okWidget = self.form.ui.buttonBox.button(self.form.ui.buttonBox.Ok)
        QTest.mouseClick(okWidget, Qt.LeftButton)
        self.assertEqual(self.form.jiggers, 4)

    def test_liters(self):
        '''Test the jiggers-to-liters conversion.'''
        self.setFormToZero()
        self.assertAlmostEqual(self.form.liters, 0.0)
        self.form.ui.iceHorizontalSlider.setValue(1)

    def test_blenderSpeedButtons(self):
        '''Test the blender speed buttons'''
        self.form.ui.speedButton1.click()
        self.assertEqual(self.form.speedName, "&Mix")
        self.form.ui.speedButton2.click()
    """


if __name__ == "__main__":
    unittest.main()
