from __future__ import annotations

from collections import OrderedDict
from PyQt5 import QtWidgets

import mfm
import mfm.experiments
import mfm.experiments.data
from mfm.experiments.reader import ExperimentReader


class GlobalFitSetup(ExperimentReader, QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self)
        ExperimentReader.__init__(self, *args, **kwargs)
        self.hide()

        self.parameterWidgets = []
        self.parameters = OrderedDict([])

    def autofitrange(self, fit, threshold=10.0, area=0.999):
        return 0, 0

    def read(self, filename=None, **kwargs):
        d = mfm.experiments.data.DataCurve(setup=self, name="Global-fit")
        return d

    def __str__(self):
        s = 'Global-Fit\n'
        s += 'Name: \t%s \n' % self.name
        return s


