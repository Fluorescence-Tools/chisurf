"""
Results tab for 2D-FLCS wizard.
"""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QTextEdit
)
from qtpy.QtCore import Signal, QRectF
import pyqtgraph as pg
import numpy as np

from chisurf import logging

class ResultsTab(QWidget):
    """Results visualization tab for 2D-FLCS analysis."""
    
    export_data_clicked = Signal()
    export_plots_clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Results plots (model and residuals side by side)
        plots_layout = QHBoxLayout()
        
        # Results plot (fitted model)
        self.results_plot = pg.PlotWidget(title="2D-MEM Fitted Model")
        self.results_plot.setMinimumHeight(300)
        self.results_plot.setLabel('bottom', 'τ₁ (ns)')
        self.results_plot.setLabel('left', 'τ₂ (ns)')
        self.results_plot.showGrid(x=True, y=True, alpha=0.1)
        self.results_image = pg.ImageItem()
        try:
            plasma = pg.colormap.get('plasma').getLookupTable()
            self.results_image.setLookupTable(plasma)
        except Exception:
            pass
        self.results_plot.addItem(self.results_image)
        plots_layout.addWidget(self.results_plot)
        
        # Residuals plot (data - model)
        self.residuals_plot = pg.PlotWidget(title="2D-MEM Residuals (Data - Model)")
        self.residuals_plot.setMinimumHeight(300)
        self.residuals_plot.setLabel('bottom', 'τ₁ (ns)')
        self.residuals_plot.setLabel('left', 'τ₂ (ns)')
        self.residuals_plot.showGrid(x=True, y=True, alpha=0.1)
        self.residuals_image = pg.ImageItem()
        try:
            viridis = pg.colormap.get('viridis').getLookupTable()
            self.residuals_image.setLookupTable(viridis)
        except Exception:
            pass
        self.residuals_plot.addItem(self.residuals_image)
        plots_layout.addWidget(self.residuals_plot)
        
        layout.addLayout(plots_layout)
        
        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(120)
        self.results_text.setText("No data loaded")
        layout.addWidget(QLabel("Analysis Results:"))
        layout.addWidget(self.results_text)
        
        # Export buttons
        button_layout = QHBoxLayout()
        
        self.export_data_button = QPushButton("Export Data")
        self.export_data_button.clicked.connect(self.export_data_clicked.emit)
        button_layout.addWidget(self.export_data_button)
        
        self.export_plots_button = QPushButton("Export Plots")
        self.export_plots_button.clicked.connect(self.export_plots_clicked.emit)
        button_layout.addWidget(self.export_plots_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()

    def set_results_text(self, text):
        self.results_text.setText(text)

    def update_results_plots(self, result, fdc_data, micro_binning_factor):
        """Update results and residuals plots."""
        if not result or 'model' not in result or self.results_image is None:
            return

        model = result['model']
        if model is not None and getattr(model, 'size', 0) > 0:
            data = np.array(model, copy=False)
            data = np.log10(data + 1.0)
            self.results_image.setImage(data, autoLevels=True)
            
            if fdc_data and 'mat_lin_t' in fdc_data:
                time_axis = fdc_data['mat_lin_t']
                if time_axis is not None and len(time_axis) > 1:
                    step = float(time_axis[1] - time_axis[0]) * micro_binning_factor
                    start = float(time_axis[0]) * micro_binning_factor
                    width = step * data.shape[1]
                    height = step * data.shape[0]
                    self.results_image.setRect(QRectF(start, start, width, height))
        else:
            self.results_image.clear()

        # Residuals
        if result and 'model' in result and self.residuals_image is not None and fdc_data and 'mat_lin' in fdc_data:
            model = result['model']
            data_mat = fdc_data['mat_lin']
            if model is not None and data_mat is not None and model.shape == data_mat.shape:
                weights = 1.0 / np.sqrt(np.abs(data_mat) + 1e-10)
                residuals = (model - data_mat) * weights
                abs_max = np.max(np.abs(residuals))
                levels = [-abs_max, abs_max] if abs_max > 0 else [-1, 1]
                self.residuals_image.setImage(residuals, levels=levels)
                
                if fdc_data and 'mat_lin_t' in fdc_data:
                    time_axis = fdc_data['mat_lin_t']
                    if time_axis is not None and len(time_axis) > 1:
                        step = float(time_axis[1] - time_axis[0]) * micro_binning_factor
                        start = float(time_axis[0]) * micro_binning_factor
                        width = step * residuals.shape[1]
                        height = step * residuals.shape[0]
                        self.residuals_image.setRect(QRectF(start, start, width, height))
            else:
                self.residuals_image.clear()
        elif self.residuals_image is not None:
            self.residuals_image.clear()
