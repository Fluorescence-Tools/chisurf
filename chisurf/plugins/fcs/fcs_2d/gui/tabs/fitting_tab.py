"""
MEM Fitting tab for 2D-FLCS wizard.
"""

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, 
    QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox
)
from qtpy.QtCore import Signal

from chisurf import logging

class FittingTab(QWidget):
    """2D-MEM fitting tab for 2D-FLCS analysis."""
    
    start_fitting_clicked = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Fitting parameters
        fitting_group = QGroupBox("2D-MEM Fitting Parameters")
        fitting_layout = QGridLayout(fitting_group)
        
        # Number of states
        fitting_layout.addWidget(QLabel("Number of States:"), 0, 0)
        self.n_states_spinbox = QSpinBox()
        self.n_states_spinbox.setRange(1, 10)
        self.n_states_spinbox.setValue(3)
        fitting_layout.addWidget(self.n_states_spinbox, 0, 1)
        
        # Tau range
        fitting_layout.addWidget(QLabel("Tau Min (ns):"), 0, 2)
        self.tau_min_spinbox = QDoubleSpinBox()
        self.tau_min_spinbox.setRange(0.001, 100.0)
        self.tau_min_spinbox.setValue(0.1)
        self.tau_min_spinbox.setDecimals(3)
        fitting_layout.addWidget(self.tau_min_spinbox, 0, 3)
        
        fitting_layout.addWidget(QLabel("Tau Max (ns):"), 1, 0)
        self.tau_max_spinbox = QDoubleSpinBox()
        self.tau_max_spinbox.setRange(0.1, 1000.0)
        self.tau_max_spinbox.setValue(10.0)
        self.tau_max_spinbox.setDecimals(3)
        fitting_layout.addWidget(self.tau_max_spinbox, 1, 1)
        
        fitting_layout.addWidget(QLabel("Tau Step:"), 1, 2)
        self.tau_step_spinbox = QDoubleSpinBox()
        self.tau_step_spinbox.setRange(0.001, 1.0)
        self.tau_step_spinbox.setValue(0.05)
        self.tau_step_spinbox.setDecimals(3)
        fitting_layout.addWidget(self.tau_step_spinbox, 1, 3)
        
        # Regularization
        fitting_layout.addWidget(QLabel("Regulator Const:"), 2, 0)
        self.regulator_spinbox = QDoubleSpinBox()
        self.regulator_spinbox.setRange(1e-6, 1e6)
        self.regulator_spinbox.setValue(0.1)
        self.regulator_spinbox.setDecimals(6)
        fitting_layout.addWidget(self.regulator_spinbox, 2, 1)
        
        # Initial y0
        fitting_layout.addWidget(QLabel("Initial y0:"), 2, 2)
        self.y0_spinbox = QDoubleSpinBox()
        self.y0_spinbox.setRange(0.0, 1e6)
        self.y0_spinbox.setValue(5000.0)
        fitting_layout.addWidget(self.y0_spinbox, 2, 3)
        
        layout.addWidget(fitting_group)
        
        # Fit button
        self.fit_button = QPushButton("Start 2D-MEM Fitting")
        self.fit_button.clicked.connect(self._on_fit_clicked)
        layout.addWidget(self.fit_button)
        
        layout.addStretch()

    def _on_fit_clicked(self):
        params = self.get_params()
        self.start_fitting_clicked.emit(params)

    def get_params(self):
        return {
            'n_components': self.n_states_spinbox.value(),
            'tau_range': (self.tau_min_spinbox.value(), self.tau_max_spinbox.value()),
            'tau_step': self.tau_step_spinbox.value(),
            'regulator': self.regulator_spinbox.value(),
            'y0_initial': self.y0_spinbox.value()
        }
