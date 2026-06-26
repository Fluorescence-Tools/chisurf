"""
2D-FCS Creation tab for 2D-FLCS wizard.
"""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
    QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox, QCheckBox
)
from qtpy.QtCore import Signal
import pyqtgraph as pg
import numpy as np

from chisurf import logging
from ...helpers import set_plot_image

class FDCTab(QWidget):
    """2D-FDC creation tab for 2D-FLCS analysis."""
    
    create_clicked = Signal(dict)
    log_intensity_changed = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Parameters
        params_group = QGroupBox("2D-FCS Parameters")
        params_layout = QGridLayout(params_group)
        
        # Time parameters
        params_layout.addWidget(QLabel("dT (μs):"), 0, 0)
        self.dT_spinbox = QDoubleSpinBox()
        self.dT_spinbox.setRange(0.001, 1000000.0)
        self.dT_spinbox.setValue(100.0)
        self.dT_spinbox.setDecimals(3)
        self.dT_spinbox.setToolTip("Macro-time bin width used when building the correlation grid (microseconds).")
        params_layout.addWidget(self.dT_spinbox, 0, 1)
        
        params_layout.addWidget(QLabel("ddT (μs):"), 0, 2)
        self.ddT_spinbox = QDoubleSpinBox()
        self.ddT_spinbox.setRange(0.001, 100000.0)
        self.ddT_spinbox.setValue(50.0)
        self.ddT_spinbox.setDecimals(3)
        self.ddT_spinbox.setToolTip("Step size for sliding the macro-time window across the trace (microseconds).")
        params_layout.addWidget(self.ddT_spinbox, 0, 3)
        
        # Micro time parameters
        params_layout.addWidget(QLabel("tMin (ns):"), 1, 0)
        self.tMin_spinbox = QDoubleSpinBox()
        self.tMin_spinbox.setRange(0.0, 100.0)
        self.tMin_spinbox.setValue(1.0)
        self.tMin_spinbox.setToolTip("Lower micro-time gate (nanoseconds) for photons entering the correlation.")
        params_layout.addWidget(self.tMin_spinbox, 1, 1)
        
        params_layout.addWidget(QLabel("tMax (ns):"), 1, 2)
        self.tMax_spinbox = QDoubleSpinBox()
        self.tMax_spinbox.setRange(0.0, 1000.0)
        self.tMax_spinbox.setValue(12.0)
        self.tMax_spinbox.setToolTip("Upper micro-time gate (nanoseconds) for photons entering the correlation.")
        params_layout.addWidget(self.tMax_spinbox, 1, 3)
        
        params_layout.addWidget(QLabel("Log Points:"), 2, 0)
        self.logt_imax_spinbox = QSpinBox()
        self.logt_imax_spinbox.setRange(10, 1000)
        self.logt_imax_spinbox.setValue(100)
        self.logt_imax_spinbox.setToolTip("Number of logarithmically spaced lag bins for the log-scale correlation map.")
        params_layout.addWidget(self.logt_imax_spinbox, 2, 1)
        
        layout.addWidget(params_group)
        
        # Create button
        self.create_fdc_button = QPushButton("Create 2D-FCS")
        self.create_fdc_button.clicked.connect(self._on_create_clicked)
        self.create_fdc_button.setToolTip("Generate the 2D-FCS correlation matrices using the parameters above.")
        layout.addWidget(self.create_fdc_button)
        
        # FDC matrix preview
        preview_group = self._build_fdc_preview_group()
        layout.addWidget(preview_group)
        
        layout.addStretch()

    def _build_fdc_preview_group(self) -> QGroupBox:
        """Create the preview group that shows 2D decay matrices."""
        preview_group = QGroupBox("2D Decay Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Controls row
        controls_layout = QHBoxLayout()
        self.log_intensity_checkbox = QCheckBox("Log Intensity")
        self.log_intensity_checkbox.setChecked(True)
        self.log_intensity_checkbox.setToolTip("Display intensity values on logarithmic scale")
        self.log_intensity_checkbox.stateChanged.connect(lambda s: self.log_intensity_changed.emit(bool(s)))
        controls_layout.addWidget(self.log_intensity_checkbox)
        controls_layout.addStretch()
        preview_layout.addLayout(controls_layout)
        
        # Plots row
        plots_layout = QHBoxLayout()
        
        # Linear matrix plot
        self.lin_plot = pg.PlotWidget(title="Linear 2D-FCS")
        self.lin_plot.setMinimumHeight(260)
        self.lin_plot.setLabel('bottom', 'τ₁ (ns)')
        self.lin_plot.setLabel('left', 'τ₂ (ns)')
        self.lin_plot.showGrid(x=True, y=True, alpha=0.1)
        self.lin_image = pg.ImageItem()
        try:
            viridis = pg.colormap.get('viridis').getLookupTable()
            self.lin_image.setLookupTable(viridis)
        except Exception:
            pass
        self.lin_plot.addItem(self.lin_image)
        plots_layout.addWidget(self.lin_plot, 1)
        
        # Logarithmic matrix plot
        self.log_plot = pg.PlotWidget(title="Log-Scale 2D-FCS")
        self.log_plot.setMinimumHeight(260)
        self.log_plot.setLabel('bottom', 'τ₁ (ns)')
        self.log_plot.setLabel('left', 'τ₂ (ns)')
        self.log_plot.showGrid(x=True, y=True, alpha=0.1)
        self.log_image = pg.ImageItem()
        try:
            magma = pg.colormap.get('magma').getLookupTable()
            self.log_image.setLookupTable(magma)
        except Exception:
            pass
        self.log_plot.addItem(self.log_image)
        plots_layout.addWidget(self.log_plot, 1)
        
        preview_layout.addLayout(plots_layout)
        return preview_group

    def _on_create_clicked(self):
        params = {
            'dT': self.dT_spinbox.value(),
            'ddT': self.ddT_spinbox.value(),
            'tMin': self.tMin_spinbox.value(),
            'tMax': self.tMax_spinbox.value(),
            'logt_imax': self.logt_imax_spinbox.value()
        }
        self.create_clicked.emit(params)

    def get_params(self):
        return {
            'dT': self.dT_spinbox.value(),
            'ddT': self.ddT_spinbox.value(),
            'tMin': self.tMin_spinbox.value(),
            'tMax': self.tMax_spinbox.value(),
            'logt_imax': self.logt_imax_spinbox.value()
        }

    def update_previews(self, fdc_data, metadata, micro_binning_factor):
        """Update FDC preview plots."""
        if not fdc_data or self.lin_image is None or self.log_image is None:
            return
            
        mat_lin = fdc_data.get('mat_lin')
        mat_lin_t = fdc_data.get('mat_lin_t')
        mat_log = fdc_data.get('mat_log')
        mat_log_t = fdc_data.get('mat_log_t')
        
        use_log_intensity = self.log_intensity_checkbox.isChecked()
        micro_res = metadata.get('micro_time_resolution', 0)
        
        if mat_lin_t is not None and len(mat_lin_t) > 0 and micro_res > 0:
            time_axis_ns = mat_lin_t * micro_res * 1e9 * micro_binning_factor
            set_plot_image(self.lin_image, mat_lin, time_axis_ns, log_scale=use_log_intensity, micro_binning_factor=micro_binning_factor)
            set_plot_image(self.log_image, mat_log, time_axis_ns, log_scale=use_log_intensity, micro_binning_factor=micro_binning_factor)
        else:
            set_plot_image(self.lin_image, mat_lin, mat_lin_t, log_scale=use_log_intensity, micro_binning_factor=micro_binning_factor)
            set_plot_image(self.log_image, mat_log, mat_lin_t, log_scale=use_log_intensity, micro_binning_factor=micro_binning_factor)
