import sys
import math
from typing import Dict, List

from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel, QDoubleSpinBox, QRadioButton,
    QGroupBox, QHBoxLayout, QVBoxLayout, QSpacerItem, QSizePolicy, QComboBox,
    QTextEdit, QPushButton, QDialog, QDialogButtonBox, QButtonGroup, QCheckBox
)
from PyQt5.QtCore import Qt

# ========= Constants & conversions =========
KB = 1.380649e-23  # J/K
NA = 6.02214076e23 # 1/mol
N_PER_nM_fL = NA * 1e-9 * 1e-15  # ≈ 0.602214076


def mPa_s_to_Pa_s(eta_mPa_s: float) -> float: return eta_mPa_s * 1e-3

def Pa_s_to_mPa_s(eta_Pa_s: float) -> float: return eta_Pa_s * 1e3

def nm_to_m(x_nm: float) -> float: return x_nm * 1e-9

def m_to_nm(x_m: float) -> float: return x_m * 1e9

def um2s_to_m2s(D_um2_s: float) -> float: return D_um2_s * 1e-12

def m2s_to_um2s(D_m2_s: float) -> float: return D_m2_s * 1e12

def us_to_s(t_us: float) -> float: return t_us * 1e-6

def s_to_us(t_s: float) -> float: return t_s * 1e6

def m3_to_fL(v_m3: float) -> float: return v_m3 / 1e-18

def fL_to_m3(v_fL: float) -> float: return v_fL * 1e-18


def water_viscosity_Pa_s(T_K: float) -> float:
    """Water viscosity (Pa·s), Kapusta 2010 app note: η(T)=A·10^(B/(T−C))."""
    A, B, C = 2.414e-5, 247.8, 140.0
    if T_K <= C:
        raise ValueError("Temperature must be > 140 K for the viscosity model.")
    return A * 10.0 ** (B / (T_K - C))


def stokes_einstein_D(T_K: float, eta_Pa_s: float, r_h_m: float) -> float:
    return KB * T_K / (6.0 * math.pi * eta_Pa_s * r_h_m)


def stokes_einstein_rh(T_K: float, eta_Pa_s: float, D_m2_s: float) -> float:
    return KB * T_K / (6.0 * math.pi * eta_Pa_s * D_m2_s)


def veff_from_tau_D_S(tau_s: float, D_m2_s: float, S: float) -> float:
    # Veff = π^(3/2) * S * (4 D τ)^(3/2)
    return (math.pi ** 1.5) * S * (4.0 * D_m2_s * tau_s) ** 1.5


def D_from_tau_Veff_S(tau_s: float, Veff_m3: float, S: float) -> float:
    denom = (math.pi ** 1.5) * S
    if denom <= 0 or tau_s <= 0: return float('nan')
    inner = Veff_m3 / denom
    if inner <= 0: return float('nan')
    return (inner ** (2.0 / 3.0)) / (4.0 * tau_s)


def scale_D_from_25C(D25_um2_s: float, T_K: float, eta_Pa_s: float) -> float:
    if T_K <= 0 or eta_Pa_s <= 0: return float('nan')
    eta_25 = 8.9e-4  # Pa·s (water @ 25 °C)
    return D25_um2_s * (T_K / 298.15) * (eta_25 / eta_Pa_s)


# ========= Dyes (inline dict; accurate refs from Kapusta 2010) =========
# Units: D25_um2_s in µm²/s (10^-6 cm²/s)
DYE_DATA: Dict[str, Dict] = {
    "Rhodamine 6G (Rh6G)": {
        "D25_um2_s": 414,
        "sources": [
            {
                "citation": "Kapusta (2010) Absolute Diffusion Coefficients, PicoQuant App Note — Table",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["2fFCS", "PFG-NMR", "PB/CF"]
            }
        ]
    },
    "Rhodamine B": {
        "D25_um2_s": 427,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (PB/CF 4.27 ± 0.04; PFG-NMR also listed)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["PB/CF", "PFG-NMR"]
            }
        ]
    },
    "Rhodamine 123": {
        "D25_um2_s": 460,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (PFG-NMR)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["PFG-NMR"]
            }
        ]
    },
    "Rhodamine 110": {
        "D25_um2_s": 470,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (PFG-NMR)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["PFG-NMR"]
            }
        ]
    },
    "Fluorescein": {
        "D25_um2_s": 425,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (PB/CF 4.25 ± 0.01)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["PB/CF"]
            }
        ]
    },
    "Oregon Green 488": {
        "D25_um2_s": 411,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (2fFCS ~4.11)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["2fFCS"]
            }
        ]
    },
    "ATTO 488 (COOH)": {
        "D25_um2_s": 400,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (2fFCS 4.0 ± 0.1)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["2fFCS"]
            }
        ]
    },
    "ATTO 655 (COOH)": {
        "D25_um2_s": 426,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (2fFCS/PFG-NMR)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["2fFCS", "PFG-NMR"]
            }
        ]
    },
    "ATTO 655 (maleimide)": {
        "D25_um2_s": 407,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (2fFCS/PFG-NMR/pmFCS)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["2fFCS", "PFG-NMR", "pmFCS"]
            }
        ]
    },
    "ATTO 655 (NHS)": {
        "D25_um2_s": 425,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (2fFCS)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["2fFCS"]
            }
        ]
    },
    "Cy5": {
        "D25_um2_s": 360,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (2fFCS 3.6 ± 0.1)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["2fFCS"]
            }
        ]
    },
    "Alexa 647": {
        "D25_um2_s": 330,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (2fFCS 3.3 ± 0.1)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["2fFCS"]
            }
        ]
    },
    "Alexa 633": {
        "D25_um2_s": 340,
        "sources": [
            {
                "citation": "Kapusta (2010) App Note — Table (2fFCS 3.4 ± 0.1)",
                "url": "https://www.picoquant.com/images/uploads/page/files/7353/appnote_diffusioncoefficients.pdf",
                "methods": ["2fFCS"]
            }
        ]
    }
}

# ========= GUI =========
class ConfocalCalcWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FCS Confocal Calculator — τ, D, rₕ, Veff, Concentration")
        self._in_update = False
        self._last_edited = 'conc'  # driver for N↔c coupling
        self._setup_ui()
        self._connect_signals()
        self._recompute()

    def _setup_ui(self):
        # Spin boxes
        self.tau_us = QDoubleSpinBox(); self._cfg(self.tau_us, 1e-3, 1e9, 3, 70.0)
        self.D_um2_s = QDoubleSpinBox(); self._cfg(self.D_um2_s, 1e-4, 1e6, 6, 400.0)
        self.rh_nm = QDoubleSpinBox(); self._cfg(self.rh_nm, 1e-3, 1e6, 6, 0.5)
        self.S = QDoubleSpinBox(); self._cfg(self.S, 0.1, 20.0, 4, 5.0)
        self.veff_fL = QDoubleSpinBox(); self._cfg(self.veff_fL, 1e-6, 1e9, 6, 0.4)
        self.temp_C = QDoubleSpinBox(); self._cfg(self.temp_C, -50.0, 200.0, 2, 20.0)
        self.eta_mPa_s = QDoubleSpinBox(); self._cfg(self.eta_mPa_s, 0.01, 10000.0, 4, 0.890)
        self.conc_nM = QDoubleSpinBox(); self._cfg(self.conc_nM, 0.0, 1e9, 6, 1.0)
        self.num_mols = QDoubleSpinBox(); self._cfg(self.num_mols, 0.0, 1e12, 3, 0.602214*0.4)

        # Make large steps comfortable
        self.num_mols.setSingleStep(0.1)
        self.conc_nM.setSingleStep(0.01)

        # Water η(T)
        self.use_water_eta = QCheckBox("Use water η(T)")
        self.use_water_eta.setChecked(True)

        # Dye block
        self.dye_combo = QComboBox(); self.dye_combo.addItems(list(DYE_DATA.keys()))
        self.scale_dref = QRadioButton("Apply with Temp/η scaling")
        self.scale_dref.setChecked(True)
        self.scale_dref_none = QRadioButton("Apply at 25 °C (no scaling)")
        self.scale_group = QButtonGroup(self); self.scale_group.addButton(self.scale_dref); self.scale_group.addButton(self.scale_dref_none)
        self.btn_apply_dref = QPushButton("Apply Dref")
        # Remove refs/help per optimization
        # (buttons kept not created)
        self.btn_show_refs = None
        self.btn_help = None

        # Layout: numeric grid
        grid = QGridLayout(); r = 0
        grid.addWidget(QLabel("τ (µs)"), r, 0); grid.addWidget(self.tau_us, r, 1); r+=1
        grid.addWidget(QLabel("D (µm²/s)"), r, 0); grid.addWidget(self.D_um2_s, r, 1); r+=1
        grid.addWidget(QLabel("rₕ (nm)"), r, 0); grid.addWidget(self.rh_nm, r, 1); r+=1
        grid.addWidget(QLabel("S (wz/wxy)"), r, 0); grid.addWidget(self.S, r, 1); r+=1
        grid.addWidget(QLabel("Veff (fL)"), r, 0); grid.addWidget(self.veff_fL, r, 1); r+=1
        grid.addWidget(QLabel("T (°C)"), r, 0); grid.addWidget(self.temp_C, r, 1); r+=1
        grid.addWidget(QLabel("η (mPa·s)"), r, 0); grid.addWidget(self.eta_mPa_s, r, 1); r+=1
        grid.addWidget(self.use_water_eta, r, 0, 1, 2); r+=1
        # 1/N and N
        self.invN = QDoubleSpinBox(); self._cfg(self.invN, 0.0, 1.0, 9, 0.0)
        grid.addWidget(QLabel("1/N"), r, 0); grid.addWidget(self.invN, r, 1); r+=1
        grid.addWidget(QLabel("N"), r, 0); grid.addWidget(self.num_mols, r, 1); r+=1
        grid.addWidget(QLabel("Conc (nM)"), r, 0); grid.addWidget(self.conc_nM, r, 1); r+=1

        # Constraint group (exclusive)
        self.rb_fix_D = QRadioButton("Fix D")
        self.rb_fix_rh = QRadioButton("Fix rₕ")
        self.rb_fix_V = QRadioButton("Fix Veff")
        self.rb_fix_D.setChecked(True)
        self.fix_group = QButtonGroup(self)
        for rb in (self.rb_fix_D, self.rb_fix_rh, self.rb_fix_V): self.fix_group.addButton(rb)
        fix_box = QGroupBox("Constraint (choose one)")
        fix_layout = QHBoxLayout();
        for rb in (self.rb_fix_D, self.rb_fix_rh, self.rb_fix_V): fix_layout.addWidget(rb)
        fix_layout.addItem(QSpacerItem(10,10,QSizePolicy.Expanding,QSizePolicy.Minimum))
        fix_box.setLayout(fix_layout)

        # Dye box
        dye_box = QGroupBox("Reference dye (D @ 25 °C, water)")
        v = QVBoxLayout()
        # Place combo and apply button on one line
        combo_row = QHBoxLayout()
        combo_row.addWidget(self.dye_combo)
        combo_row.addWidget(self.btn_apply_dref)
        combo_row.addItem(QSpacerItem(10,10,QSizePolicy.Expanding,QSizePolicy.Minimum))
        v.addLayout(combo_row)
        v.addWidget(self.scale_dref)
        v.addWidget(self.scale_dref_none)
        dye_box.setLayout(v)

        # Place constraint group box at top of widget
        root = QVBoxLayout(); root.addWidget(fix_box); root.addLayout(grid); root.addWidget(dye_box)
        self.setLayout(root)
        self._update_field_enable()
        # Initialize water viscosity mode default
        self._on_use_water_eta(self.use_water_eta.isChecked())

    def _cfg(self, sb: QDoubleSpinBox, lo: float, hi: float, dec: int, val: float):
        sb.setRange(lo, hi); sb.setDecimals(dec); sb.setValue(val)

    def _connect_signals(self):
        # Core fields trigger recompute
        for w in (self.tau_us, self.D_um2_s, self.rh_nm, self.S, self.veff_fL, self.temp_C, self.eta_mPa_s):
            w.valueChanged.connect(self._recompute)
        # Bi-directional coupling (user edits drive the other)
        self.conc_nM.valueChanged.connect(self._conc_changed)
        self.num_mols.valueChanged.connect(self._N_changed)
        self.invN.valueChanged.connect(self._invN_changed)

        # Constraint selection & UI logic
        self.fix_group.buttonToggled.connect(self._on_constraint_changed)
        self.use_water_eta.toggled.connect(self._on_use_water_eta)

        # Dye actions
        self.btn_apply_dref.clicked.connect(self._apply_dref_to_D)

    # ---------- Constraint / UI state ----------
    def _on_constraint_changed(self, _btn, _state):
        self._update_field_enable()
        self._recompute()

    def _update_field_enable(self):
        # Outputs are disabled according to the chosen constraint
        fixD = self.rb_fix_D.isChecked()
        fixR = self.rb_fix_rh.isChecked()
        fixV = self.rb_fix_V.isChecked()
        self.D_um2_s.setReadOnly(fixD is False)
        self.rh_nm.setReadOnly(fixR is False)
        self.veff_fL.setReadOnly(fixV is False)
        # Make read-only fields visually distinct
        for sb in (self.D_um2_s, self.rh_nm, self.veff_fL):
            pal = sb.palette()
            if sb.isReadOnly():
                pal.setColor(sb.backgroundRole(), pal.base().color())
            sb.setPalette(pal)

    def _on_use_water_eta(self, checked: bool):
        self.eta_mPa_s.setEnabled(not checked)
        self._recompute()

    # ---------- Coupling helpers ----------
    def _conc_changed(self, val: float):
        if self._in_update: return
        self._last_edited = 'conc'
        V_fL = self.veff_fL.value()
        if V_fL > 0:
            N = val * V_fL * N_PER_nM_fL
            self._set_spin(self.num_mols, N)
            if N > 0:
                self._set_spin(self.invN, 1.0/N)
            else:
                self._set_spin(self.invN, 0.0)

    def _N_changed(self, val: float):
        if self._in_update: return
        self._last_edited = 'N'
        V_fL = self.veff_fL.value()
        if V_fL > 0:
            conc = val / (V_fL * N_PER_nM_fL)
            self._set_spin(self.conc_nM, conc)
            if val > 0:
                self._set_spin(self.invN, 1.0/val)
            else:
                self._set_spin(self.invN, 0.0)

    def _invN_changed(self, val: float):
        if self._in_update: return
        self._last_edited = 'invN'
        V_fL = self.veff_fL.value()
        N = 1.0/val if val > 0 else 0.0
        self._set_spin(self.num_mols, N)
        if V_fL > 0:
            conc = (N / (V_fL * N_PER_nM_fL)) if V_fL > 0 else 0.0
            self._set_spin(self.conc_nM, conc)

    # ---------- Dye actions ----------
    def _apply_dref_to_D(self):
        name = self.dye_combo.currentText()
        info = DYE_DATA.get(name)
        if not info:
            return
        D_use = float(info.get("D25_um2_s", float("nan")))
        if self.scale_dref.isChecked():
            T_K = self.temp_C.value() + 273.15
            eta = self._current_eta_Pa_s()
            D_use = scale_D_from_25C(D_use, T_K, eta)
        # Switch to Fix D so D is authoritative
        self.rb_fix_D.setChecked(True)
        self._update_field_enable()
        self._set_spin(self.D_um2_s, D_use)
        self._recompute()
        self.D_um2_s.setFocus()

    # References/help removed in optimized UI

    # ---------- Utility ----------
    def _current_eta_Pa_s(self) -> float:
        if self.use_water_eta.isChecked():
            T_K = self.temp_C.value() + 273.15
            eta = water_viscosity_Pa_s(T_K)
            # mirror into UI (disabled when checked)
            self._set_spin(self.eta_mPa_s, Pa_s_to_mPa_s(eta))
            return eta
        return mPa_s_to_Pa_s(self.eta_mPa_s.value())

    def _set_spin(self, spin: QDoubleSpinBox, value: float):
        self._in_update = True
        spin.setValue(value)
        self._in_update = False

    # ---------- Core recomputation ----------
    def _recompute(self):
        if self._in_update: return
        self._in_update = True
        try:
            tau_s = us_to_s(self.tau_us.value())
            S = self.S.value()
            T_K = self.temp_C.value() + 273.15
            eta = self._current_eta_Pa_s()
            D_m2_s = um2s_to_m2s(self.D_um2_s.value())
            rh_m = nm_to_m(self.rh_nm.value())
            Veff_m3 = fL_to_m3(self.veff_fL.value())

            if self.rb_fix_D.isChecked():
                if tau_s > 0 and S > 0 and D_m2_s > 0:
                    Veff_new = veff_from_tau_D_S(tau_s, D_m2_s, S)
                    self._set_spin(self.veff_fL, m3_to_fL(Veff_new))
                if D_m2_s > 0:
                    rh_new = stokes_einstein_rh(T_K, eta, D_m2_s)
                    self._set_spin(self.rh_nm, m_to_nm(rh_new))
            elif self.rb_fix_rh.isChecked():
                if rh_m > 0:
                    D_new = stokes_einstein_D(T_K, eta, rh_m)
                    self._set_spin(self.D_um2_s, m2s_to_um2s(D_new))
                    if tau_s > 0 and S > 0:
                        Veff_new = veff_from_tau_D_S(tau_s, D_new, S)
                        self._set_spin(self.veff_fL, m3_to_fL(Veff_new))
            elif self.rb_fix_V.isChecked():
                if tau_s > 0 and S > 0 and Veff_m3 > 0:
                    D_new = D_from_tau_Veff_S(tau_s, Veff_m3, S)
                    if math.isfinite(D_new) and D_new > 0:
                        self._set_spin(self.D_um2_s, m2s_to_um2s(D_new))
                        rh_new = stokes_einstein_rh(T_K, eta, D_new)
                        self._set_spin(self.rh_nm, m_to_nm(rh_new))

            # Keep N and concentration consistent when Veff changes
            V_fL = self.veff_fL.value()
            if V_fL > 0:
                if self._last_edited == 'N':
                    N = self.num_mols.value()
                    conc = N / (V_fL * N_PER_nM_fL)
                    self._set_spin(self.conc_nM, conc)
                    if N > 0:
                        self._set_spin(self.invN, 1.0/N)
                    else:
                        self._set_spin(self.invN, 0.0)
                elif self._last_edited == 'invN':
                    invN = self.invN.value()
                    N = 1.0/invN if invN > 0 else 0.0
                    self._set_spin(self.num_mols, N)
                    conc = (N / (V_fL * N_PER_nM_fL)) if V_fL > 0 else 0.0
                    self._set_spin(self.conc_nM, conc)
                else:
                    conc = self.conc_nM.value()
                    N = conc * V_fL * N_PER_nM_fL
                    self._set_spin(self.num_mols, N)
                    if N > 0:
                        self._set_spin(self.invN, 1.0/N)
                    else:
                        self._set_spin(self.invN, 0.0)
        finally:
            self._in_update = False


def main():
    app = QApplication(sys.argv)
    w = ConfocalCalcWidget()
    w.resize(820, 660)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

if __name__ == 'plugin':
    plugin = ConfocalCalcWidget()
    plugin.show()
