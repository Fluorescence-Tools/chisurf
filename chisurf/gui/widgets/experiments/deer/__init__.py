from __future__ import annotations

"""Minimal GUI controller for the DEER experiment reader.

Provides file selection for Bruker BES3T (``.DSC``/``.DTA``), CSV and text DEER
traces and renders the reader's AutoForm settings (phase correction,
normalisation, experiment type) when available.
"""

import pathlib

from qtpy import QtWidgets

import chisurf as cs
from chisurf.core.experiments.core import reader


class DeerController(reader.ExperimentReaderController, QtWidgets.QWidget):
    """File-selection controller for :class:`DeerReader`."""

    def __init__(self, *args, **kwargs):
        """Build a compact controller: a hint label + reader settings form."""
        super().__init__(*args, **kwargs)
        self._preview_filename = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        label = QtWidgets.QLabel(
            "DEER: open a Bruker BES3T (.DSC/.DTA) or CSV/text trace."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        reader_obj = getattr(self, "experiment_reader", None)
        if reader_obj is not None and hasattr(reader_obj, "view_spec"):
            try:
                from chisurf.gui.autoform import AutoForm

                layout.addWidget(AutoForm(reader_obj, parent=self))
            except Exception:
                pass

    def get_filename(self) -> pathlib.Path:
        """Return the selected DEER file path via an open-file dialog."""
        if self._preview_filename is not None:
            return pathlib.Path(self._preview_filename)
        fn = cs.gui.widgets.open_files(
            description="DEER trace (BES3T .DSC/.DTA, CSV, text)",
            file_type="DEER files (*.DSC *.DTA *.csv *.txt *.dat);;All files (*.*)",
            working_path=None,
        )
        if isinstance(fn, (list, tuple)):
            return pathlib.Path(fn[0]) if fn else pathlib.Path("")
        return pathlib.Path(fn) if fn else pathlib.Path("")

    @property
    def filename(self) -> str:
        """Return the currently selected file path as a string."""
        fn = self.get_filename()
        return str(fn) if fn is not None else ""


__all__ = ["DeerController"]
