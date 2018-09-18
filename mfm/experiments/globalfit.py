from PyQt4 import QtGui
from collections import OrderedDict

from mfm.experiments import Setup
import mfm


class GlobalFitSetup(QtGui.QWidget, Setup):

    def __init__(self, *args, **kwargs):
        QtGui.QWidget.__init__(self)
        Setup.__init__(self, *args, **kwargs)
        self.hide()

        self.parameterWidgets = []
        self.parameters = OrderedDict([])

    def autofitrange(self, fit, threshold=10.0, area=0.999):
        return 0, 0

    def load_data(self, filename=None, **kwargs):
        d = mfm.curve.DataCurve(setup=self)
        d.name = "Global-fit"
        return d

    def __str__(self):
        s = 'Global-Fit\n'
        s += 'Name: \t%s \n' % self.name
        return s


