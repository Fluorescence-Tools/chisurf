import utils
import os
import sys
import unittest
from qtpy.QtWidgets import QApplication
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt
import pathlib

TOPDIR = pathlib.Path(__file__).parent.parent

utils.set_search_paths(TOPDIR)

from chisurf.plugins.kappa2_dist.k2dgui import Kappa2Dist


app = QApplication(sys.argv)


class Tests(unittest.TestCase):

    def setUp(self):
        self.form = Kappa2Dist()

    def test_defaults(self):
        self.assertEqual(self.form._model.r_0, 0.380)
        self.assertEqual(self.form._model.r_Dinf, 0.050)
        self.assertEqual(self.form._model.r_Ainf, 0.100)
        self.assertEqual(self.form._model.step, 1.5)
        self.assertEqual(self.form._model.n_bins, 131)
        self.assertEqual(self.form._model.r_ADinf, 0.005)

    def test_calculation_1(self):
        okWidget = self.form.pushButton
        QTest.mouseClick(okWidget, Qt.LeftButton)

        self.assertAlmostEqual(self.form._model.k2_mean, 0.7, places=1)
        self.assertAlmostEqual(self.form._model.k2_sd, 0.2, places=1)
        self.assertAlmostEqual(self.form._model.Rapp_mean, 1.0, places=1)
        self.assertAlmostEqual(self.form._model.RappSD, 0.0, places=1)

    def test_calculation_2(self):
        self.form._model.rAD_known = True

        okWidget = self.form.pushButton
        QTest.mouseClick(okWidget, Qt.LeftButton)

        self.assertAlmostEqual(self.form._model.k2_mean, 0.7, places=1)
        self.assertAlmostEqual(self.form._model.k2_sd, 0.2, places=1)
        self.assertAlmostEqual(self.form._model.Rapp_mean, 1.0, places=1)
        self.assertAlmostEqual(self.form._model.RappSD, 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
