from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

import mfm
import mfm.widgets
import mfm.fluorescence
import mfm.experiments
import mfm.experiments.data
import mfm.io.ascii
from . import reader


class FCS(
    reader.ExperimentReader,
    mfm.io.widgets.CsvWidget
):

    name = "FCS"

    def __init__(
            self,
            use_header: bool = False,
            skiprows: int = 0,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.hide()
        self.skiprows = skiprows
        self.use_header = use_header
        self.spinBox.setEnabled(False)
        self.parent = kwargs.get('parent', None)
        self.groupBox.hide()

    def read(
            self,
            filename: str = None,
            **kwargs
    ):
        d = mfm.experiments.data.DataCurve(
            setup=self
        )
        d.setup = self

        mfm.io.widgets.CsvWidget.load(
            self,
            filename=filename,
            skiprows=0,
            use_header=None,
            verbose=mfm.verbose
        )

        x, y = self.data[0], self.data[1]
        w = self.data[2]

        d.set_data(self.filename, x, y, w)
        return d


class FCSKristine(
    mfm.io.ascii.Csv,
    FCS
):

    name = 'Kristine'

    def __init__(
            self,
            *args,
            **kwargs
    ):
        FCS.__init__(
            self,
            *args,
            **kwargs
        )
        QtWidgets.QWidget.__init__(self)
        mfm.io.ascii.Csv.__init__(self, **kwargs)

    def read(
            self,
            filename: str = None,
            verbose: bool = None,
            **kwargs
    ):
        """Uses either the error provided by the correlator (4. column)
        or calculates the error based on the correlation curve,
        the aquisition time and the count-rate.

        :param filename:
        :param verbose:
        :param kwargs:
        :return:
        """
        d = mfm.experiments.data.DataCurve(setup=self)
        d.setup = self
        if filename is None:
            filename = mfm.widgets.get_filename(
                'Kristine-Correlation file',
                file_type='All files (*.cor)'
            )

        self.load(
            filename=filename,
            skiprows=0,
            use_header=None,
            verbose=verbose
        )

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
        d.set_data(
            self.filename,
            x,
            y,
            ey=w
        )
        return d


