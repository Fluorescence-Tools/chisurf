"""
GMM Settings Dialog for BrickMicWizard

This module provides a dialog for adjusting Gaussian Mixture Model (GMM) settings
used in the BrickMicWizard for histogram fitting.
"""

from qtpy import QtWidgets


class GMMSettingsDialog(QtWidgets.QDialog):
    """Dialog for adjusting GMM (Gaussian Mixture Model) settings."""

    def __init__(self, parent=None, gmm_settings=None):
        super().__init__(parent)
        self.setWindowTitle("GMM Settings")
        self.resize(400, 300)

        # Default settings if none provided
        self.gmm_settings = gmm_settings or {
            'covariance_type': 'full',
            'random_state': 42,
            'max_iter': 300,
            'n_init': 10,
            'tol': 1e-3,
            'max_components': 10,
            'reg_covar': 1e-6
        }

        # Create layout
        layout = QtWidgets.QVBoxLayout(self)

        # Create form layout for settings
        form_layout = QtWidgets.QFormLayout()

        # Covariance type
        self.covariance_combo = QtWidgets.QComboBox()
        self.covariance_combo.addItems(['full', 'tied', 'diag', 'spherical'])
        index = self.covariance_combo.findText(self.gmm_settings['covariance_type'])
        if index >= 0:
            self.covariance_combo.setCurrentIndex(index)
        form_layout.addRow("Covariance Type:", self.covariance_combo)

        # Random state
        self.random_state_spin = QtWidgets.QSpinBox()
        self.random_state_spin.setRange(0, 1000)
        self.random_state_spin.setValue(self.gmm_settings['random_state'])
        form_layout.addRow("Random State:", self.random_state_spin)

        # Max iterations
        self.max_iter_spin = QtWidgets.QSpinBox()
        self.max_iter_spin.setRange(10, 1000)
        self.max_iter_spin.setValue(self.gmm_settings['max_iter'])
        form_layout.addRow("Max Iterations:", self.max_iter_spin)

        # Number of initializations
        self.n_init_spin = QtWidgets.QSpinBox()
        self.n_init_spin.setRange(1, 20)
        self.n_init_spin.setValue(self.gmm_settings['n_init'])
        form_layout.addRow("Number of Initializations:", self.n_init_spin)

        # Convergence threshold
        self.tol_spin = QtWidgets.QDoubleSpinBox()
        self.tol_spin.setRange(1e-6, 1e-1)
        self.tol_spin.setDecimals(6)
        self.tol_spin.setSingleStep(1e-4)
        self.tol_spin.setValue(self.gmm_settings['tol'])
        form_layout.addRow("Convergence Threshold:", self.tol_spin)

        # Max components for auto-determination
        self.max_components_spin = QtWidgets.QSpinBox()
        self.max_components_spin.setRange(2, 20)
        self.max_components_spin.setValue(self.gmm_settings['max_components'])
        form_layout.addRow("Max Components for Auto:", self.max_components_spin)

        # Regularization parameter for covariance matrices
        self.reg_covar_spin = QtWidgets.QDoubleSpinBox()
        self.reg_covar_spin.setRange(1e-10, 1e-1)
        self.reg_covar_spin.setDecimals(10)
        self.reg_covar_spin.setSingleStep(1e-7)
        self.reg_covar_spin.setValue(self.gmm_settings.get('reg_covar', 1e-6))
        form_layout.addRow("Covariance Regularization:", self.reg_covar_spin)

        layout.addLayout(form_layout)

        # Add buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_settings(self):
        """Return the current GMM settings from the dialog."""
        return {
            'covariance_type': self.covariance_combo.currentText(),
            'random_state': self.random_state_spin.value(),
            'max_iter': self.max_iter_spin.value(),
            'n_init': self.n_init_spin.value(),
            'tol': self.tol_spin.value(),
            'max_components': self.max_components_spin.value(),
            'reg_covar': self.reg_covar_spin.value()
        }
