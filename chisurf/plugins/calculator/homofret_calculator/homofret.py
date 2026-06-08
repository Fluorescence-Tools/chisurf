from __future__ import annotations

import numpy as np
from chisurf.gui import QtWidgets
import chisurf as cs
import chisurf.gui.decorators


class HomoFRETCalculator(QtWidgets.QWidget):

    name = "HomoFRET-Calculator"

    @cs.gui.decorators.init_with_ui("calculator/homofret_calculator/calc_homofret.ui", path=cs.core.settings.plugin_path)
    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        # Wire signals
        self.spin_tRM.editingFinished.connect(self.compute_homo)
        self.spin_rho.editingFinished.connect(self.compute_homo)
        self.spin_tau0.editingFinished.connect(self.compute_homo)
        self.spin_R0.editingFinished.connect(self.compute_homo)
        self.spin_Rhomo.editingFinished.connect(self.compute_backmap)
        # Initial compute
        self.compute_homo()

    def compute_homo(self):
        """Compute homoFRET exchange rate from anisotropy and convert to distance.

        Equations:
          k_homo = 0.5 * (1/t_RM - 1/ρ)
          R_DA = R0 * (k_homo * tau0)^(-1/6)

        Units:
          - tau0 in ns
          - t_RM, ρ in ns
          - k_homo in 1/ns
          - R0 in user units (Å or nm), R_DA in same units
        """
        try:
            t_RM = float(self.spin_tRM.value())
            rho = float(self.spin_rho.value())
            tau0_ns = float(self.spin_tau0.value())
            R0 = float(self.spin_R0.value())
        except Exception:
            return

        if t_RM <= 0 or rho <= 0 or tau0_ns <= 0 or R0 <= 0:
            return

        diff = (1.0 / t_RM) - (1.0 / rho)
        k_homo = 0.5 * diff
        if k_homo < 0:
            k_homo = 0.0

        try:
            self.spin_kHomo.setValue(float(k_homo))
        except Exception:
            pass

        Rh = np.nan
        prod = k_homo * tau0_ns
        if prod > 0.0:
            Rh = R0 * (prod) ** (-1.0 / 6.0)

        if np.isfinite(Rh) and Rh > 0:
            try:
                self.spin_Rhomo.setValue(float(Rh))
            except Exception:
                pass


    def compute_backmap(self):
        """Backmap from R_DA to t_RM using HomoFRET relations.

        Equations (ns units):
          k_homo = (R0 / R_DA)^6 / tau0
          t_RM   = 1 / (2*k_homo + 1/ρ)
        """
        # Read inputs
        try:
            R_DA = float(self.spin_Rhomo.value())
            R0 = float(self.spin_R0.value())
            tau0_ns = float(self.spin_tau0.value())
            rho_ns = float(self.spin_rho.value())
        except Exception:
            return

        # Validate inputs
        if R_DA <= 0 or R0 <= 0 or tau0_ns <= 0 or rho_ns <= 0:
            return

        # Compute k_homo (1/ns)
        try:
            ratio = R0 / R_DA
            k_homo = (ratio ** 6) / tau0_ns
        except Exception:
            return

        # Compute t_RM (ns)
        denom = (2.0 * k_homo) + (1.0 / rho_ns)
        t_RM = np.nan
        if denom > 0:
            t_RM = 1.0 / denom

        # Update outputs without triggering other computations
        try:
            self.spin_kHomo.blockSignals(True)
            self.spin_tRM.blockSignals(True)
            if np.isfinite(k_homo) and k_homo >= 0:
                self.spin_kHomo.setValue(float(k_homo))
            if np.isfinite(t_RM) and t_RM > 0:
                self.spin_tRM.setValue(float(t_RM))
        finally:
            self.spin_kHomo.blockSignals(False)
            self.spin_tRM.blockSignals(False)
