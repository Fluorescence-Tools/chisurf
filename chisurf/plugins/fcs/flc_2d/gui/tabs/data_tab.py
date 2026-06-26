"""
Data Input tab for 2D-FLCS wizard.
"""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QComboBox, QMessageBox, QFileDialog
)
from qtpy.QtCore import Signal
import pyqtgraph as pg
import numpy as np
from pathlib import Path
import traceback

from chisurf import logging
from ...helpers import get_filetype_from_path

class DataTab(QWidget):
    """Data Input tab for 2D-FLCS analysis."""
    
    data_loaded = Signal(object)  # Emits tttr_data
    file_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.wizard = parent  # Reference to main wizard for settings/detector access
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # File selection
        file_select_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select TTTR data file...")
        self.file_path_edit.textChanged.connect(self._on_file_path_changed)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_data_file)
        file_select_layout.addWidget(QLabel("TTTR File:"))
        file_select_layout.addWidget(self.file_path_edit)
        file_select_layout.addWidget(self.browse_button)
        layout.addLayout(file_select_layout)
        
        # Detector selection
        detector_select_layout = QHBoxLayout()
        detector_select_layout.addWidget(QLabel("Detector:"))
        self.detector_combo = QComboBox()
        self.detector_combo.setMinimumWidth(200)
        self.detector_combo.currentTextChanged.connect(self._on_detector_changed)
        detector_select_layout.addWidget(self.detector_combo)
        
        self.refresh_detectors_button = QPushButton("Refresh")
        self.refresh_detectors_button.clicked.connect(self.refresh_detector_combo)
        self.refresh_detectors_button.setToolTip("Refresh detector list from detector setup")
        detector_select_layout.addWidget(self.refresh_detectors_button)
        
        detector_select_layout.addStretch()
        layout.addLayout(detector_select_layout)
        
        # Data info
        self.data_info_label = QLabel("No data loaded")
        layout.addWidget(self.data_info_label)
        
        # Data preview (intensity trace + decay)
        preview_group = self._build_data_preview_group()
        layout.addWidget(preview_group)
        
        layout.addStretch()

    def _build_data_preview_group(self) -> QGroupBox:
        """Create preview group showing intensity trace and fluorescence decay."""
        preview_group = QGroupBox("Data Preview")
        preview_layout = QHBoxLayout(preview_group)

        # Intensity trace (macro count stream)
        self.mcs_plot = pg.PlotWidget(title="Intensity Trace (all photons)")
        self.mcs_plot.setMinimumHeight(220)
        self.mcs_plot.setLabel('bottom', 'Time', units='s')
        self.mcs_plot.setLabel('left', 'Counts')
        self.mcs_plot.showGrid(x=True, y=True, alpha=0.1)
        self.mcs_curve = self.mcs_plot.plot([], [], pen=pg.mkPen('#64b5f6', width=1.5))
        preview_layout.addWidget(self.mcs_plot, 1)

        # Fluorescence decay (microtime histogram)
        self.decay_plot = pg.PlotWidget(title="Fluorescence Decay (all photons)")
        self.decay_plot.setMinimumHeight(220)
        self.decay_plot.setLabel('bottom', 'Microtime', units='ns')
        self.decay_plot.setLabel('left', 'Counts')
        self.decay_plot.showGrid(x=True, y=True, alpha=0.1)
        self.decay_plot.setLogMode(False, True)
        self.decay_curve = self.decay_plot.plot([], [], pen=pg.mkPen('#ff8a65', width=1.5))
        preview_layout.addWidget(self.decay_plot, 1)

        return preview_group

    def _browse_data_file(self):
        """Browse for TTTR data file(s)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select TTTR Data File", "", 
            "TTTR Files (*.ptu *.pt3 *.ht3 *.hdf5 *.h5 *.spc *.sdt *.bin);;All Files (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def _on_file_path_changed(self):
        """Auto-load data when file path changes."""
        # This will be handled by the main wizard to trigger loading
        self.file_changed.emit(self.file_path_edit.text())

    def _on_detector_changed(self, detector_name):
        """Handle detector selection change."""
        self.logger.info(f"2D-FLCS: Detector changed to {detector_name}")

    def refresh_detector_combo(self):
        """Refresh detector list from detector setup."""
        self.logger.info("2D-FLCS: refresh_detector_combo called")
        try:
            self.detector_combo.clear()
            
            if not self.wizard or self.wizard.detector_wizard_page is None:
                self.detector_combo.addItem("Detector setup not available")
                return
            
            settings = self.wizard._get_detector_settings()
            if not settings:
                self.detector_combo.addItem("No detectors configured")
                return
            
            detectors = settings.get('detectors', {})
            if not detectors:
                self.detector_combo.addItem("No detectors configured")
                return
                
            for name in detectors.keys():
                self.detector_combo.addItem(name)
                
        except Exception as e:
            self.logger.error(f"Error refreshing detector combo: {e}")
            self.detector_combo.addItem("Error loading detectors")

    def get_selected_detector(self):
        """Get the currently selected detector."""
        return self.detector_combo.currentText()

    def update_data_preview(self, tttr_data, tttr_obj, preview_time_window, preview_decay_coarse):
        """Update intensity trace and decay previews."""
        if tttr_data is None or tttr_obj is None:
            self.mcs_curve.setData([], [])
            self.decay_curve.setData([], [])
            return

        try:
            # Update intensity trace (macro count stream)
            time_window = max(preview_time_window, 1e-6)
            counts = tttr_obj.get_intensity_trace(time_window_length=time_window)
            x_axis = np.arange(len(counts), dtype=np.float64) * time_window
            self.mcs_curve.setData(x_axis, counts)

            # Update fluorescence decay (microtime histogram)
            coarse = max(int(preview_decay_coarse), 1)
            counts, bins = tttr_obj.get_microtime_histogram(coarse)
            if len(bins) > 1:
                bins_ns = bins[:-1] * 1e9
                counts = counts[:len(bins_ns)]
                self.decay_curve.setData(bins_ns, counts)
            else:
                self.decay_curve.setData([], [])

        except Exception as exc:
            self.logger.error(f"2D-FLCS: Failed updating data previews: {exc}")
            self.mcs_curve.setData([], [])
            self.decay_curve.setData([], [])
