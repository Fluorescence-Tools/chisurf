import utils
import os
import sys
import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)
import mfm.tools


app = QApplication(sys.argv)


class Kappa2Distribution(unittest.TestCase):
    """
    Test the kappa2 distribution GUI
    """

    def setUp(self):
        """
        Create the GUI
        """
        self.form = mfm.tools.kappa2_distribution.kappa2dist.Kappa2Dist()

    def setFormToZero(self):
        '''Set all ingredients to zero in preparation for setting just one
        to a nonzero value.
        '''
        self.form.ui.tequilaScrollBar.setValue(0)
        self.form.ui.tripleSecSpinBox.setValue(0)
        self.form.ui.limeJuiceLineEdit.setText("0.0")
        self.form.ui.iceHorizontalSlider.setValue(0)

    def test_defaults(self):
        '''Test the GUI in its default state'''
        self.assertEqual(self.form.ui.tequilaScrollBar.value(), 8)
        self.assertEqual(self.form.ui.tripleSecSpinBox.value(), 4)
        self.assertEqual(self.form.ui.limeJuiceLineEdit.text(), "12.0")
        self.assertEqual(self.form.ui.iceHorizontalSlider.value(), 12)
        self.assertEqual(self.form.ui.speedButtonGroup.checkedButton().text(), "&Karate Chop")

        # Class is in the default state even without pressing OK
        self.assertEqual(self.form.jiggers, 36.0)
        self.assertEqual(self.form.speedName, "&Karate Chop")

        # Push OK with the left mouse button
        okWidget = self.form.ui.buttonBox.button(self.form.ui.buttonBox.Ok)
        QTest.mouseClick(okWidget, Qt.LeftButton)
        self.assertEqual(self.form.jiggers, 36.0)
        self.assertEqual(self.form.speedName, "&Karate Chop")

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
        self.assertAlmostEqual(self.form.liters, 0.0444)
        self.form.ui.iceHorizontalSlider.setValue(2)
        self.assertAlmostEqual(self.form.liters, 0.0444 * 2)

    def test_blenderSpeedButtons(self):
        '''Test the blender speed buttons'''
        self.form.ui.speedButton1.click()
        self.assertEqual(self.form.speedName, "&Mix")
        self.form.ui.speedButton2.click()
        self.assertEqual(self.form.speedName, "&Whip")
        self.form.ui.speedButton3.click()
        self.assertEqual(self.form.speedName, "&Puree")
        self.form.ui.speedButton4.click()
        self.assertEqual(self.form.speedName, "&Chop")
        self.form.ui.speedButton5.click()
        self.assertEqual(self.form.speedName, "&Karate Chop")
        self.form.ui.speedButton6.click()
        self.assertEqual(self.form.speedName, "&Beat")
        self.form.ui.speedButton7.click()
        self.assertEqual(self.form.speedName, "&Smash")
        self.form.ui.speedButton8.click()
        self.assertEqual(self.form.speedName, "&Liquefy")
        self.form.ui.speedButton9.click()
        self.assertEqual(self.form.speedName, "&Vaporize")


if __name__ == "__main__":
    unittest.main()
