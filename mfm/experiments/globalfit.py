from collections import OrderedDict

from PyQt5 import QtWidgets

import mfm
from mfm.experiments import Reader


class GlobalFitSetup(Reader, QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self)
        Reader.__init__(self, *args, **kwargs)
        self.hide()

        self.parameterWidgets = []
        self.parameters = OrderedDict([])

    def autofitrange(self, fit, threshold=10.0, area=0.999):
        return 0, 0

    def read(self, filename=None, **kwargs):
        d = mfm.curve.DataCurve(setup=self, name="Global-fit")
        return d

    def __str__(self):
        s = 'Global-Fit\n'
        s += 'Name: \t%s \n' % self.name
        return s


