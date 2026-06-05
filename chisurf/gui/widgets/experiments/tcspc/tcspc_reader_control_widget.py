from __future__ import annotations

from chisurf.gui import QtWidgets

import pathlib

import chisurf
import chisurf.gui.widgets
import chisurf.gui.widgets.fio
from chisurf.core.experiments.core import reader

from .csv_tcspc_widget import CsvTCSPCWidget


class TCSPCReaderControlWidget(
    reader.ExperimentReaderController,
    QtWidgets.QWidget,
):
    def get_filename(self) -> pathlib.Path:
        return chisurf.gui.widgets.get_filename(
            description="CSV-TCSPC file",
            file_type="All files (*.*)",
            working_path=None,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.layout = layout
        csv_widget = chisurf.gui.widgets.fio.CsvWidget()
        self.layout.addWidget(csv_widget)
        self.csv_tcspc_widget = CsvTCSPCWidget()
        self.layout.addWidget(self.csv_tcspc_widget)

    def updateUI(self):
        """Update UI elements based on current_setup properties."""
        # Call updateUI on the CsvTCSPCWidget
        self.csv_tcspc_widget.updateUI()
