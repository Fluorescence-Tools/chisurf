from __future__ import annotations

from qtpy import QtWidgets

import mfm
import chisurf.fio.ascii
import chisurf.fio.widgets
import chisurf.widgets
import mfm.fluorescence
import experiments.data
import chisurf.fio.fluorescence
from . import reader


class FCS(
    reader.ExperimentReader
):

    def __init__(
            self,
            name: str,
            use_header: bool = False,
            experiment_reader='Kristine',
            skiprows: int = 0,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.name = name
        self.skiprows = skiprows
        self.use_header = use_header
        self.experiment_reader = experiment_reader

    def read(
            self,
            filename: str = None,
            verbose: bool = None,
            **kwargs
    ):
        if self.experiment_reader == 'Kristine':
            return chisurf.fio.fluorescence.read_fcs_kristine(
                filename=filename,
                verbose=verbose
            )
        else:
            return chisurf.fio.fluorescence.read_fcs(
                filename=filename,
                setup=self,
                skiprows=self.skiprows,
                use_header=self.use_header
            )


class FCSController(
    reader.ExperimentReaderController,
    QtWidgets.QWidget
):

    @property
    def filename(
            self
    ) -> str:
        return self.get_filename()

    def __init__(
            self,
            file_type='Kristine files (*.cor)',
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.file_type = file_type
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.layout = layout
        self.layout.addWidget(
            chisurf.fio.widgets.CsvWidget()
        )

    def get_filename(
            self
    ) -> str:
        return chisurf.widgets.get_filename(
                'FCS-CSV files',
                file_type=self.file_type
            )


