from __future__ import annotations

from typing import Optional

from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QGroupBox,
)


class BrickMicSetupDialog(QDialog):
    """Simple setup dialog for BrickMic acquisition backend.

    Exposes all hardware acquisition parameters that are specific to the
    BrickMic backend, plus SPC file options:

    - Number of channels (``channels``)
    - Start and end counter indices (``start_ctr``, ``end_ctr``)
    - Acquisition rate in Hz (``acquisition_rate``)
    - Buffer size per channel (``buffer_size``)
    - Photons per SPC file (``N_ph_per_file``)

    The SPC output directory itself is controlled in the main
    acquisition dock.
    """

    def __init__(self, device, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BrickMic Setup")
        self.setModal(True)

        # The plugin passes the TCSPCDevice factory as ``device``.
        # Extract the underlying BrickMicDevice if present.
        self._factory = device
        self._brickmic = getattr(device, "device", None)

        self._init_ui()
        self._load_from_device()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        hw_group = QGroupBox("BrickMic Hardware Acquisition")
        hw_layout = QFormLayout(hw_group)

        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(1, 8)
        self.channels_spin.setValue(2)
        hw_layout.addRow("Channels:", self.channels_spin)

        self.start_ctr_spin = QSpinBox()
        self.start_ctr_spin.setRange(0, 31)
        self.start_ctr_spin.setValue(0)
        hw_layout.addRow("Start counter index:", self.start_ctr_spin)

        self.end_ctr_spin = QSpinBox()
        self.end_ctr_spin.setRange(0, 31)
        self.end_ctr_spin.setValue(1)
        hw_layout.addRow("End counter index:", self.end_ctr_spin)

        self.acq_rate_spin = QDoubleSpinBox()
        self.acq_rate_spin.setRange(1_000.0, 5_000_000.0)
        self.acq_rate_spin.setDecimals(0)
        self.acq_rate_spin.setSingleStep(100_000.0)
        self.acq_rate_spin.setValue(1_000_000.0)
        self.acq_rate_spin.setSuffix(" Hz")
        hw_layout.addRow("Acquisition rate:", self.acq_rate_spin)

        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(10_000, 20_000_000)
        self.buffer_size_spin.setSingleStep(100_000)
        self.buffer_size_spin.setValue(2_000_000)
        hw_layout.addRow("Buffer size / channel:", self.buffer_size_spin)

        layout.addWidget(hw_group)

        spc_group = QGroupBox("SPC File Output")
        spc_layout = QFormLayout(spc_group)
        self.n_ph_per_file_spin = QSpinBox()
        self.n_ph_per_file_spin.setRange(1000, 1_000_000_000)
        self.n_ph_per_file_spin.setSingleStep(10_000)
        self.n_ph_per_file_spin.setValue(100_000)
        spc_layout.addRow("Photons per .spc file:", self.n_ph_per_file_spin)
        layout.addWidget(spc_group)

        button_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_row.addStretch(1)
        button_row.addWidget(ok_btn)
        button_row.addWidget(cancel_btn)
        layout.addLayout(button_row)

    def _load_from_device(self) -> None:
        brick = self._brickmic
        if brick is None:
            return

        # Hardware parameters
        try:
            if hasattr(brick, "channels"):
                self.channels_spin.setValue(int(getattr(brick, "channels")))
        except Exception:
            pass

        try:
            if hasattr(brick, "start_ctr"):
                self.start_ctr_spin.setValue(int(getattr(brick, "start_ctr")))
        except Exception:
            pass

        try:
            if hasattr(brick, "end_ctr"):
                self.end_ctr_spin.setValue(int(getattr(brick, "end_ctr")))
        except Exception:
            pass

        try:
            if hasattr(brick, "acquisition_rate"):
                self.acq_rate_spin.setValue(float(getattr(brick, "acquisition_rate")))
        except Exception:
            pass

        try:
            if hasattr(brick, "buffer_size"):
                self.buffer_size_spin.setValue(int(getattr(brick, "buffer_size")))
        except Exception:
            pass

        # SPC options
        if hasattr(brick, "N_ph_per_file"):
            try:
                value = int(getattr(brick, "N_ph_per_file"))
                if value > 0:
                    self.n_ph_per_file_spin.setValue(value)
            except Exception:
                pass

    def accept(self) -> None:  # type: ignore[override]
        brick = self._brickmic
        if brick is not None:
            # Hardware parameters: keep them consistent
            try:
                start = int(self.start_ctr_spin.value())
                end = int(self.end_ctr_spin.value())
                if end < start:
                    end = start
                channels = max(1, end - start + 1)

                if hasattr(brick, "start_ctr"):
                    setattr(brick, "start_ctr", start)
                if hasattr(brick, "end_ctr"):
                    setattr(brick, "end_ctr", end)
                if hasattr(brick, "channels"):
                    setattr(brick, "channels", channels)
            except Exception:
                pass

            try:
                if hasattr(brick, "acquisition_rate"):
                    setattr(
                        brick,
                        "acquisition_rate",
                        int(self.acq_rate_spin.value()),
                    )
            except Exception:
                pass

            try:
                if hasattr(brick, "buffer_size"):
                    setattr(
                        brick,
                        "buffer_size",
                        int(self.buffer_size_spin.value()),
                    )
            except Exception:
                pass

            # SPC options
            try:
                if hasattr(brick, "N_ph_per_file"):
                    setattr(
                        brick,
                        "N_ph_per_file",
                        int(self.n_ph_per_file_spin.value()),
                    )
            except Exception:
                pass

        super().accept()
