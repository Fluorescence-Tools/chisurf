from __future__ import annotations

import numpy as np
from chisurf.gui import QtWidgets

import chisurf.gui.decorators
from chisurf.core.fluorescence.general import \
    distance_to_fret_rate_constant, distance_to_fret_efficiency, fret_efficiency_to_lifetime, \
    lifetime_to_fret_efficiency, fretrate_to_distance, fret_efficiency_to_distance, gaussian2rates


class FRETCalculator(QtWidgets.QWidget):

    name = "FRET-Calculator"

    @chisurf.gui.decorators.init_with_ui("calculator/fret_calculator/calc_tau2r.ui", path=chisurf.core.settings.plugin_path)
    def __init__(self, kappa2=0.667, *args, **kwargs):
        self.kappa2 = kappa2

        ## User-interface
        self.doubleSpinBox.editingFinished.connect(self.onTau0Changed)
        self.doubleSpinBox_2.editingFinished.connect(self.onTauChanged)
        self.doubleSpinBox_3.editingFinished.connect(self.onR0Changed)
        self.doubleSpinBox_9.editingFinished.connect(self.onRChanged)
        self.doubleSpinBox_6.editingFinished.connect(self.onSigmaChanged)
        self.doubleSpinBox_4.editingFinished.connect(self.onEChanged)
        self.doubleSpinBox_5.editingFinished.connect(self.onkFRETChanged)
        self.onkFRETChanged()


    def onRChanged(self):
        self.blockSignals(True)
        
        # Check if we should use distance distribution mode
        if self.sigma > 0:
            # Use Gaussian distance distribution
            # Calculate FRET rates using gaussian2rates
            rates = gaussian2rates(
                means=[self.R],
                sigmas=[self.sigma],
                amplitudes=[1.0],
                tau0=self.tau0,
                kappa2=self.kappa2,
                R0=self.R0,
                n_points=64,
                interleaved=False
            )
            
            # Calculate average FRET rate
            avg_rate = np.sum(rates[:, 0] * rates[:, 1])
            self.kFRET = avg_rate
            
            # Calculate average FRET efficiency
            avg_efficiency = 0.0
            total_weight = 0.0
            
            for i in range(len(rates)):
                weight = rates[i, 0]
                rate = rates[i, 1]
                efficiency = rate * self.tau0 / (1 + rate * self.tau0)
                avg_efficiency += weight * efficiency
                total_weight += weight
                
            if total_weight > 0:
                avg_efficiency /= total_weight
                
            self.E = avg_efficiency
            self.tau = fret_efficiency_to_lifetime(self.E, self.tau0)
        else:
            # Use single distance mode (original implementation)
            self.kFRET = distance_to_fret_rate_constant(
                self.R,
                self.R0,
                self.tau0,
                self.kappa2
            )
            self.E = distance_to_fret_efficiency(
                self.R,
                self.R0
            )
            self.tau = fret_efficiency_to_lifetime(
                self.E,
                self.tau0
            )
            
        self.blockSignals(False)

    def onkFRETChanged(self):
        self.blockSignals(True)
        
        # Calculate R from kFRET (this is the same regardless of sigma)
        self.R = fretrate_to_distance(
            self.kFRET,
            self.R0,
            self.tau0,
            self.kappa2
        )
        
        # Set sigma to 0 temporarily to avoid recursive calculations
        current_sigma = self.sigma
        self.sigma = 0
        
        # Calculate E and tau using the single-distance approach
        self.E = distance_to_fret_efficiency(
            self.R,
            self.R0
        )
        self.tau = fret_efficiency_to_lifetime(
            self.E,
            self.tau0
        )
        
        # Restore sigma and recalculate if needed
        self.sigma = current_sigma
        if current_sigma > 0:
            # Call onRChanged to handle the distance distribution calculations
            self.blockSignals(False)  # Allow signals for onRChanged
            self.onRChanged()
            self.blockSignals(True)   # Block signals again
            
        self.blockSignals(False)

    def onTauChanged(self):
        self.blockSignals(True)
        
        # Calculate E from tau
        self.E = lifetime_to_fret_efficiency(
            self.tau,
            self.tau0
        )
        
        # Calculate R from E
        self.R = fret_efficiency_to_distance(
            self.E,
            self.R0
        )
        
        # Set sigma to 0 temporarily to avoid recursive calculations
        current_sigma = self.sigma
        self.sigma = 0
        
        # Calculate kFRET using the single-distance approach
        self.kFRET = distance_to_fret_rate_constant(
            self.R,
            self.R0,
            self.tau0,
            self.kappa2
        )
        
        # Restore sigma and recalculate if needed
        self.sigma = current_sigma
        if current_sigma > 0:
            # Call onRChanged to handle the distance distribution calculations
            self.blockSignals(False)  # Allow signals for onRChanged
            self.onRChanged()
            self.blockSignals(True)   # Block signals again
            
        self.blockSignals(False)

    def onEChanged(self):
        self.blockSignals(True)
        
        # Calculate R from E
        self.R = fret_efficiency_to_distance(
            self.E,
            self.R0
        )
        
        # Set sigma to 0 temporarily to avoid recursive calculations
        current_sigma = self.sigma
        self.sigma = 0
        
        # Calculate kFRET and tau using the single-distance approach
        self.kFRET = distance_to_fret_rate_constant(
            self.R,
            self.R0,
            self.tau0,
            self.kappa2
        )
        self.tau = fret_efficiency_to_lifetime(
            self.E,
            self.tau0
        )
        
        # Restore sigma and recalculate if needed
        self.sigma = current_sigma
        if current_sigma > 0:
            # Call onRChanged to handle the distance distribution calculations
            self.blockSignals(False)  # Allow signals for onRChanged
            self.onRChanged()
            self.blockSignals(True)   # Block signals again
            
        self.blockSignals(False)

    def onTau0Changed(self):
        self.blockSignals(True)
        
        # Set sigma to 0 temporarily to avoid recursive calculations
        current_sigma = self.sigma
        self.sigma = 0
        
        # Calculate E, tau, and kFRET using the single-distance approach
        self.E = distance_to_fret_efficiency(
            self.R,
            self.R0
        )
        self.tau = fret_efficiency_to_lifetime(
            self.E,
            self.tau0
        )
        self.kFRET = distance_to_fret_rate_constant(
            self.R,
            self.R0,
            self.tau0,
            self.kappa2
        )
        
        # Restore sigma and recalculate if needed
        self.sigma = current_sigma
        if current_sigma > 0:
            # Call onRChanged to handle the distance distribution calculations
            self.blockSignals(False)  # Allow signals for onRChanged
            self.onRChanged()
            self.blockSignals(True)   # Block signals again
        
        
        self.blockSignals(False)

    def onR0Changed(self):
        self.blockSignals(True)
        
        # Set sigma to 0 temporarily to avoid recursive calculations
        current_sigma = self.sigma
        self.sigma = 0
        
        # Calculate E, tau, and kFRET using the single-distance approach
        self.E = distance_to_fret_efficiency(
            self.R,
            self.R0
        )
        self.tau = fret_efficiency_to_lifetime(
            self.E,
            self.tau0
        )
        self.kFRET = distance_to_fret_rate_constant(
            self.R,
            self.R0,
            self.tau0,
            self.kappa2
        )
        
        # Restore sigma and recalculate if needed
        self.sigma = current_sigma
        if current_sigma > 0:
            # Call onRChanged to handle the distance distribution calculations
            self.blockSignals(False)  # Allow signals for onRChanged
            self.onRChanged()
            self.blockSignals(True)   # Block signals again
        
        # Update HomoFRET mirrors and recompute if available
        if hasattr(self, "homo_group"):
            self._update_R0_label()
            self.compute_homo()
            
        self.blockSignals(False)

    @property
    def kFRET(self) -> float:
        return float(self.doubleSpinBox_5.value())

    @kFRET.setter
    def kFRET(
            self,
            value: float
    ):
        self.doubleSpinBox_5.setValue(value)

    @property
    def tau0(self) -> float:
        return float(self.doubleSpinBox.value())

    @property
    def tau(self) -> float:
        return float(self.doubleSpinBox_2.value())

    @tau.setter
    def tau(
            self,
            v: float
    ):
        self.doubleSpinBox_2.setValue(v)

    @property
    def R0(self) -> float:
        return float(self.doubleSpinBox_3.value())

    @property
    def R(self) -> float:
        return float(self.doubleSpinBox_9.value())

    @R.setter
    def R(self, v: float):
        self.doubleSpinBox_9.setValue(v)

    @property
    def E(self) -> float:
        return float(self.doubleSpinBox_4.value())

    @E.setter
    def E(self, v: float):
        self.doubleSpinBox_4.setValue(v)
        
    @property
    def sigma(self) -> float:
        return float(self.doubleSpinBox_6.value())
        
    @sigma.setter
    def sigma(self, v: float):
        self.doubleSpinBox_6.setValue(v)
        
    def onSigmaChanged(self):
        """Handle changes to the sigma parameter (width of distance distribution)"""
        self.blockSignals(True)
        # When sigma changes, recalculate based on the current distance
        self.onRChanged()
        self.blockSignals(False)

