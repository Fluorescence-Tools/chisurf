from __future__ import annotations

import pathlib

from qtpy import QtWidgets

import chisurf.gui.widgets
import chisurf.gui.widgets.fio
from chisurf.experiments.core import reader


class FCSController(reader.ExperimentReaderController, QtWidgets.QWidget):

    def get_filename(self) -> pathlib.Path:
        return chisurf.gui.widgets.get_filename('FCS-CSV files', file_type=self.file_type)

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

        # Noise / weighting model selector for FCS correlation amplitudes.
        # This controls the ExperimentReader.weight_mode attribute so that
        # read_fcs can recompute correlation_amplitude_weights using the
        # selected noise model.
        noise_layout = QtWidgets.QHBoxLayout()
        noise_layout.setContentsMargins(0, 0, 0, 0)
        noise_layout.setSpacing(4)

        noise_label = QtWidgets.QLabel("Noise model:")
        self.noise_model_combo = QtWidgets.QComboBox()

        # First entry: keep whatever the file/reader provides (no re-weighting
        # in read_fcs, i.e. weight_mode=None).
        self.noise_model_combo.addItem("From file (default)", userData=None)
        # Uniform weights (no noise model).
        self.noise_model_combo.addItem("None / uniform", userData="none")
        # Photon-noise model using the Suren estimate.
        self.noise_model_combo.addItem("Photon-noise (Suren)", userData="photon_noise")
        # Starchev variance model.
        self.noise_model_combo.addItem("Starchev", userData="starchev")
        # PyCorrFit-style spline local-variance weights (default 5 knots).
        self.noise_model_combo.addItem("Spline local variance (5 knots)", userData="spline5")

        self.noise_model_combo.currentIndexChanged.connect(self._on_noise_model_changed)

        noise_layout.addWidget(noise_label)
        noise_layout.addWidget(self.noise_model_combo, 1)
        self.layout.addLayout(noise_layout)

        # CSV-style format controls (header, skiprows, columns, etc.).
        self.csv_widget = chisurf.gui.widgets.fio.CsvWidget()
        self.layout.addWidget(self.csv_widget)

        # Sync initial combobox selection with the underlying reader, if
        # available. By default FCS.weight_mode is None, which corresponds to
        # "From file (default)".
        try:
            current_mode = getattr(self.experiment_reader, 'weight_mode', None)
        except Exception:
            current_mode = None

        try:
            for i in range(self.noise_model_combo.count()):
                if self.noise_model_combo.itemData(i) == current_mode:
                    self.noise_model_combo.setCurrentIndex(i)
                    break
        except Exception:
            pass

    def _on_noise_model_changed(self, index: int) -> None:
        """Qt slot: update the FCS reader's noise/weighting mode."""

        try:
            mode = self.noise_model_combo.itemData(index)
        except Exception:
            mode = None

        reader_obj = getattr(self, 'experiment_reader', None)
        if reader_obj is None:
            return

        try:
            reader_obj.weight_mode = mode
        except Exception:
            pass

    def updateUI(self):
        """Update UI elements based on current_setup properties."""
        import chisurf
        try:
            setup = chisurf.cs.current_setup
        except Exception:
            return

        # Sync noise model combo from reader's weight_mode
        try:
            weight_mode = getattr(setup, 'weight_mode', None)
            for i in range(self.noise_model_combo.count()):
                if self.noise_model_combo.itemData(i) == weight_mode:
                    self.noise_model_combo.blockSignals(True)
                    self.noise_model_combo.setCurrentIndex(i)
                    self.noise_model_combo.blockSignals(False)
                    break
        except Exception:
            pass

    def onParametersChanged(self):
        """Push current parameters into cs.current_setup.

        This is a pass-through to _on_noise_model_changed since the FCS
        controller only has one interactive parameter.
        """
        self._on_noise_model_changed(self.noise_model_combo.currentIndex())


__all__ = ["FCSController"]
