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
import chisurf.gui.tools


app = QApplication(sys.argv)


class Tests(unittest.TestCase):
    """
    Test the kappa2 distribution GUI
    """

    def setUp(self):
        """
        Create the GUI
        """
        self.form = chisurf.gui.tools.kappa2_distribution.kappa2dist.Kappa2Dist()

    def test_defaults(self):
        self.assertEqual(self.form.doubleSpinBox_2.value(), 0.380)
        self.assertEqual(self.form.doubleSpinBox.value(), 0.050)
        self.assertEqual(self.form.doubleSpinBox_5.value(), 0.100)
        self.assertEqual(self.form.doubleSpinBox_3.value(), 1.500000)
        self.assertEqual(self.form.spinBox.value(), 131)
        self.assertEqual(self.form.doubleSpinBox_7.value(), 0.005)

    def test_calculation_1(self):
        okWidget = self.form.pushButton
        QTest.mouseClick(okWidget, Qt.LeftButton)

        self.assertAlmostEqual(self.form.doubleSpinBox_10.value(), 0.7545, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_9.value(), 0.1907, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_6.value(), 1.0164, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_8.value(), 0.0424, places=2)

    def test_calculation_2(self):
        check_box = self.form.checkBox
        check_box.setCheckState(True)

        okWidget = self.form.pushButton
        QTest.mouseClick(okWidget, Qt.LeftButton)

        self.assertAlmostEqual(self.form.doubleSpinBox_10.value(), 0.7275, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_9.value(), 0.2194, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_6.value(), 1.0085, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_8.value(), 0.0493, places=2)

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
