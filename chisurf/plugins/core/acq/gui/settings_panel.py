import logging
import copy
import pathlib
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QStackedWidget, QFormLayout, QScrollArea,
    QLineEdit, QToolButton
)
from qtpy.QtCore import QTimer

import chisurf as cs
from chisurf.settings import gui as gui_settings

logger = logging.getLogger(__name__)

class AcquisitionSettingsWidget(QWidget):
    """
    Centralized settings panel for Photon Acquisition.
    Embeds hardware card setup and advanced acquisition parameters.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Ensure acquisition settings dict exists
        self.config = gui_settings.setdefault("acquisition", {})
        
        # Reference to the currently embedded device setup dialog
        self._device_dialog = None
        
        # Periodic sync: collect params from embedded dialog and persist
        # Must be created before _load_settings() since it calls _on_device_type_changed
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self._sync_dialog_params)
        
        self._build_ui()
        self._load_settings()
        self._connect_signals()
        self._sync_timer.start(1000)  # sync every 1 second

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout(output_group)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Standard folder for new measurements")
        self.output_path_browse = QToolButton()
        self.output_path_browse.setText("...")
        self.output_path_browse.clicked.connect(self._browse_output_path)
        output_layout.addWidget(QLabel("Folder:"))
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_path_browse)
        main_layout.addWidget(output_group)

        # 1. Advanced Settings Group
        advanced_group = QGroupBox("Advanced Configuration")
        advanced_layout = QFormLayout(advanced_group)
        
        self.chunk_size_spinbox = QSpinBox()
        self.chunk_size_spinbox.setRange(1000, 65536)
        self.chunk_size_spinbox.setValue(16384)
        self.chunk_size_spinbox.setSuffix(" photons")
        self.chunk_size_spinbox.setToolTip("Number of photons to read per chunk (each photon = 32 bits)")
        advanced_layout.addRow("Chunk Size:", self.chunk_size_spinbox)
        
        self.real_time_sim_checkbox = QCheckBox("Real-time Sim")
        self.real_time_sim_checkbox.setChecked(False)
        self.real_time_sim_checkbox.setToolTip("Pace simulation to roughly 1s wall time = 1s sim time")
        advanced_layout.addRow("", self.real_time_sim_checkbox)
        
        main_layout.addWidget(advanced_group)
        
        # 2. Hardware / Device Setup Group
        device_group = QGroupBox("Device Configuration")
        device_layout = QVBoxLayout(device_group)
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Active Device Type:"))
        
        self.device_type_combo = QComboBox()
        self.device_type_combo.addItems(["Simulation", "Becker-Hickl", "PicoQuant", "BrickMic"])
        header_layout.addWidget(self.device_type_combo)
        header_layout.addStretch()
        
        device_layout.addLayout(header_layout)
        
        # Stacked widget for device-specific dialogs/widgets
        self.device_stack = QStackedWidget()
        
        # We will wrap the stacked widget in a scroll area since device setups can be large
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.device_stack)
        scroll.setFrameShape(QScrollArea.NoFrame)
        
        device_layout.addWidget(scroll)
        main_layout.addWidget(device_group)

    def _load_settings(self):
        """Load settings from centralized config."""
        self.output_path_edit.setText(self._default_output_path())
        self.chunk_size_spinbox.setValue(self.config.get("chunk_size", 16384))
        self.real_time_sim_checkbox.setChecked(self.config.get("real_time_sim", False))
        
        device_type = self.config.get("device_type", "Simulation")
        idx = self.device_type_combo.findText(device_type)
        if idx >= 0:
            self.device_type_combo.setCurrentIndex(idx)
            
        self._on_device_type_changed(device_type)

    def _connect_signals(self):
        self.output_path_edit.editingFinished.connect(self._save_settings)
        self.chunk_size_spinbox.valueChanged.connect(self._save_settings)
        self.real_time_sim_checkbox.toggled.connect(self._save_settings)
        self.device_type_combo.currentTextChanged.connect(self._on_device_type_changed)

    def _default_output_path(self) -> str:
        """Return the configured output folder or a sensible fallback."""
        configured = str(self.config.get("output_path", "") or "").strip()
        if configured:
            return configured

        working_path = getattr(cs, "working_path", "") or ""
        if working_path:
            return str(pathlib.Path(working_path) / "acquisition")

        return str(pathlib.Path.home() / "chisurf" / "acquisition")

    def _browse_output_path(self) -> None:
        """Pick the standard output folder from a directory chooser."""
        from qtpy.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(
            self,
            "Select acquisition output folder",
            self.output_path_edit.text() or self._default_output_path(),
        )
        if path:
            self.output_path_edit.setText(path)
            self._save_settings()

    def _save_settings(self, *_):
        """Save settings back to centralized config and persist to disk."""
        output_path = str(self.output_path_edit.text()).strip()
        if output_path:
            output_path = str(pathlib.Path(output_path).expanduser())
        self.config["output_path"] = output_path
        self.config["chunk_size"] = self.chunk_size_spinbox.value()
        self.config["real_time_sim"] = self.real_time_sim_checkbox.isChecked()
        self.config["device_type"] = self.device_type_combo.currentText()
        self._persist_to_disk()

    def _persist_to_disk(self):
        """Write the full acquisition config to the user's settings YAML."""
        try:
            from chisurf.core.settings.settings_utils import set_acquisition_settings
            set_acquisition_settings(copy.deepcopy(self.config))
        except Exception as e:
            logger.warning(f"Failed to persist acquisition settings: {e}")

    def _sync_dialog_params(self):
        """Periodically collect parameters from the embedded dialog and persist to config."""
        if self._device_dialog is None:
            return
        if not hasattr(self._device_dialog, 'get_parameters'):
            return
        try:
            params = self._device_dialog.get_parameters()
            self.config["simulation_params"] = copy.deepcopy(params)
            self._persist_to_disk()
        except Exception as e:
            logger.debug(f"Failed to sync dialog params: {e}")

    @staticmethod
    def _hide_dialog_buttons(dialog):
        """Hide OK/Cancel buttons from a dialog embedded as a widget."""
        from qtpy.QtWidgets import QPushButton
        for btn in dialog.findChildren(QPushButton):
            text = btn.text().strip().lower()
            if text in ('ok', 'cancel'):
                btn.hide()

    def _on_device_type_changed(self, device_type: str):
        self._save_settings()
        
        # Stop sync timer while we rebuild
        self._sync_timer.stop()
        self._device_dialog = None
        
        # Only show Real-time Sim for Simulation device
        self.real_time_sim_checkbox.setVisible(device_type == "Simulation")
        
        # Clear stack
        while self.device_stack.count() > 0:
            widget = self.device_stack.widget(0)
            self.device_stack.removeWidget(widget)
            widget.deleteLater()
            
        # Dynamically load the correct configuration widget
        setup_widget = QWidget()
        layout = QVBoxLayout(setup_widget)
        
        try:
            if device_type == "Simulation":
                from chisurf.plugins.core.acq.tcspc_devices.simulation.setup_dialog import EnhancedSimulationSetupDialog
                # We instantiate the dialog but use it as a widget
                dialog = EnhancedSimulationSetupDialog()
                # Apply any previously saved simulation params
                saved_params = self.config.get("simulation_params")
                if saved_params and hasattr(dialog, '_apply_parameters'):
                    dialog._apply_parameters(saved_params)
                # Hide OK/Cancel buttons — they call accept()/reject() which closes the widget
                self._hide_dialog_buttons(dialog)
                self._device_dialog = dialog
                layout.addWidget(dialog)
            elif device_type == "PicoQuant":
                from chisurf.plugins.core.acq.tcspc_devices.picoquant.setup_dialog import PicoQuantSetupDialog
                dialog = PicoQuantSetupDialog()
                self._device_dialog = dialog
                layout.addWidget(dialog)
            elif device_type == "Becker-Hickl":
                from chisurf.plugins.core.acq.tcspc_devices.bh_spc.card_setup_dialog import BHSPCCardSetupDialog
                # BH requires a device instance usually, but we can pass None or create a mock if needed
                # For now we assume we can instantiate it without crashing
                dialog = BHSPCCardSetupDialog(None)
                self._device_dialog = dialog
                layout.addWidget(dialog)
            else:
                layout.addWidget(QLabel(f"No specific configuration available for {device_type}."))
        except Exception as e:
            logger.error(f"Error loading setup widget for {device_type}: {e}")
            layout.addWidget(QLabel(f"Error loading {device_type} configuration.\n{e}"))
            
        self.device_stack.addWidget(setup_widget)
        self.device_stack.setCurrentIndex(0)
        
        # Do an immediate sync and restart timer
        self._sync_dialog_params()
        self._sync_timer.start(1000)
