"""PicoQuant Setup Dialog.

Dialog for configuring PicoQuant device settings.
"""

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox,
    QMessageBox,
)
from qtpy.QtCore import Qt


class PicoQuantSetupDialog(QDialog):
    """Setup dialog for PicoQuant device settings."""

    def __init__(self, device=None, parent=None):
        """Initialize the PicoQuant setup dialog."""
        super().__init__(parent)
        self.device = device
        self.setWindowTitle("PicoQuant Setup")
        self.setModal(True)
        self.resize(400, 300)

        # Load current parameters (PicoQuant devices have minimal configuration)
        self.params = {
            'measurement_mode': 1,  # T2 mode
            'reference_source': 0,  # Internal
            'max_photons': 100000,
        }

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Info group
        info_group = QGroupBox("PicoQuant Device Information")
        info_layout = QVBoxLayout(info_group)

        info_label = QLabel(
            "PicoQuant devices are configured automatically through the snAPI.\n"
            "Most settings are handled by the device firmware and INI files.\n"
            "No additional hardware configuration is typically required."
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)

        layout.addWidget(info_group)

        # Basic settings (mostly informational)
        settings_group = QGroupBox("Basic Settings")
        settings_layout = QFormLayout(settings_group)

        # Measurement mode (read-only for now)
        self.mode_label = QLabel("T3 Mode (Time-resolved)")
        settings_layout.addRow("Measurement Mode:", self.mode_label)

        # Reference source (read-only for now)
        self.ref_label = QLabel("Internal")
        settings_layout.addRow("Reference Source:", self.ref_label)

        # Max photons
        self.max_photons_spin = QSpinBox()
        self.max_photons_spin.setRange(1000, 10000000)
        self.max_photons_spin.setValue(self.params['max_photons'])
        self.max_photons_spin.setSingleStep(10000)
        settings_layout.addRow("Max Photons:", self.max_photons_spin)

        layout.addWidget(settings_group)

        # Status info
        if self.device and hasattr(self.device, 'api') and self.device.api:
            status_group = QGroupBox("Device Status")
            status_layout = QVBoxLayout(status_group)

            try:
                devices = self.device.api.detect_devices()
                if devices:
                    status_text = f"Found {len(devices)} device(s)\n"
                    for dev in devices:
                        status_text += f"Device: {dev.get('device_id', 'Unknown')}\n"
                        status_text += f"Status: {dev.get('status', 'Unknown')}\n"
                else:
                    status_text = "No devices detected"
            except Exception as e:
                status_text = f"Error checking device status: {e}"

            status_label = QLabel(status_text)
            status_layout.addWidget(status_label)
            layout.addWidget(status_group)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def get_parameters(self):
        """Get the current parameter values."""
        params = self.params.copy()
        params['max_photons'] = self.max_photons_spin.value()
        return params
