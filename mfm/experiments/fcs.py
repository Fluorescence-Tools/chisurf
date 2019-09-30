from __future__ import annotations

from qtpy import QtWidgets

import mfm
import mfm.io.ascii
import mfm.io.widgets
import mfm.widgets
import mfm.fluorescence
import mfm.experiments.data
import mfm.io.fluorescence
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
        return mfm.io.fluorescence.read_kristine(
            data=self.data
        )


