from PyQt4 import QtGui
import numpy as np

import mfm
from . import Setup
from mfm.io.widgets import CsvWidget

settings = mfm.settings['correlator']


class FCS(QtGui.QWidget, Setup):

    name = "FCS"

    def __init__(self, experiment, *args, **kwargs):
        QtGui.QWidget.__init__(self)
        mfm.Base.__init__(self, *args, **kwargs)
        self.experiment = experiment
        self.hide()
        self.parent = kwargs.get('parent', None)


class FCSKristine(FCS, mfm.io.ascii.Csv, QtGui.QWidget):

    name = 'Kristine'

    def __init__(self, experiment, **kwargs):
        QtGui.QWidget.__init__(self)
        FCS.__init__(self, experiment)
        mfm.io.ascii.Csv.__init__(self, **kwargs)
        #CsvWidget.__init__(self, **kwargs)
        self.skiprows = 0
        self.use_header = False
        #self.spinBox.setEnabled(False)
        self.parent = kwargs.get('parent', None)
        #self.groupBox.hide()

    def load_data(self, filename=None, **kwargs):
        """Uses eihter the error provided by the correlator (4. column)
        or calculates the error based on the correlation curve,
        the aquisition time and the count-rate.

        :param filename:
        :param kwargs:
        :return:
        """
        d = mfm.curve.DataCurve(setup=self, **kwargs)
        d.setup = self
        self.load(filename=filename, skiprows=0, use_header=None, verbose=mfm.verbose)

        # In Kristine file-type
        # First try to use experimental errors
        x, y = self.data[0], self.data[1]
        i = np.where(x > 0.0)
        x = x[i]
        y = y[i]
        try:
            w = 1. / self.data[3][i]
        except IndexError:
            # In case this doesnt work
            # use calculated errors using count-rate and duration
            try:
                dur, cr = self.data[2, 0], self.data[2, 1]
                w = mfm.fluorescence.fcs.weights(x, y, dur, cr)
            except IndexError:
                # In case everything fails
                # Use no errors at all but uniform weighting
                w = np.ones_like(y)
        d.set_data(self.filename, x, y, weights=w)
        return d


class FCSCsv(FCS, CsvWidget):

    name = 'CSV'

    def __init__(self, experiment, **kwargs):
        QtGui.QWidget.__init__(self)
        FCS.__init__(self, experiment)
        CsvWidget.__init__(self, **kwargs)
        self.skiprows = 0
        self.use_header = False
        self.spinBox.setEnabled(False)
        self.parent = kwargs.get('parent', None)
        self.groupBox.hide()

    def load_data(self, filename=None, **kwargs):
        d = mfm.curve.DataCurve(setup=self)
        d.setup = self

        self.load(filename=filename, skiprows=0, use_header=None, verbose=mfm.verbose)

        x, y = self.data[0], self.data[1]
        w = self.data[2]

        d.set_data(self.filename, x, y, w)
        return d


