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
    Test the kappa2 distribution GUI
    """

    def setUp(self):
        """
        Create the GUI
        """
        self.form = chisurf.tools.tttr.correlate.CorrelateTTTR()

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

        self.assertAlmostEqual(self.form.doubleSpinBox_10.value(), 0.6605, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_9.value(), 0.1398, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_6.value(), 0.9951, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_8.value(), 0.0363, places=2)
