import sys
import math
import json
from typing import Dict, List

from qtpy.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel, QDoubleSpinBox, QRadioButton,
    QGroupBox, QHBoxLayout, QVBoxLayout, QSpacerItem, QSizePolicy, QComboBox,
    QTextEdit, QPushButton, QDialog, QDialogButtonBox, QButtonGroup, QCheckBox,
    QFileDialog,
)
from qtpy.QtCore import Qt

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c

from .core.algorithms import *  # noqa: F401,F403
from .core.algorithms import compute_confocal, DYE_DATA, N_PER_nM_fL


# ========= GUI =========
import pathlib

_VIEW_JSON = pathlib.Path(__file__).parent / "fcs_calculator.view.json"


class _ConfocalModel:
    """Backing model for the declarative editor (fields in fcs_calculator.view.json).

    AutoForm reads the initial values from these attributes; the calculator's
    compute logic operates on the grabbed editor widgets, so the model only
    provides defaults and the view-spec.
    """

    def __init__(self) -> None:
        self.tau_us = 70.0
        self.D_um2_s = 400.0
        self.rh_nm = 0.5
        self.S = 5.0
        self.veff_fL = 0.4
        self.temp_C = 20.0
        self.eta_mPa_s = 0.890
        self.use_water_eta = True
        self.invN = 0.0
        self.num_mols = 0.602214 * 0.4
        self.conc_nM = 1.0

    def view_spec(self):
        from chisurf.core.dataspec import load_view_spec
        return load_view_spec(_VIEW_JSON)


class _ConstraintSection(QWidget):
    """Exclusive Fix-D / Fix-rₕ / Fix-Veff radios (a declarative custom section)."""

    def __init__(self, model, target=None, parent=None, **kwargs):
        super().__init__(parent)
        self.rb_fix_D = QRadioButton("Fix D")
        self.rb_fix_rh = QRadioButton("Fix rₕ")
        self.rb_fix_V = QRadioButton("Fix Veff")
        self.rb_fix_D.setChecked(True)
        self.fix_group = QButtonGroup(self)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        for rb in (self.rb_fix_D, self.rb_fix_rh, self.rb_fix_V):
            self.fix_group.addButton(rb)
            lay.addWidget(rb)
        lay.addStretch(1)


class _DyeSection(QWidget):
    """Reference-dye combo + apply button + scaling radios."""

    def __init__(self, model, target=None, parent=None, **kwargs):
        super().__init__(parent)
        self.dye_combo = QComboBox()
        self.dye_combo.addItems(list(DYE_DATA.keys()))
        self.btn_apply_dref = QPushButton("Apply Dref")
        self.scale_dref = QRadioButton("Apply with Temp/η scaling")
        self.scale_dref.setChecked(True)
        self.scale_dref_none = QRadioButton("Apply at 25 °C (no scaling)")
        self.scale_group = QButtonGroup(self)
        self.scale_group.addButton(self.scale_dref)
        self.scale_group.addButton(self.scale_dref_none)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 2, 0, 2)
        row = QHBoxLayout()
        row.addWidget(self.dye_combo)
        row.addWidget(self.btn_apply_dref)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(self.scale_dref)
        v.addWidget(self.scale_dref_none)


class _ShapeSection(QWidget):
    """Molecular-shape estimator (type / size / aspect + apply→D)."""

    def __init__(self, model, target=None, parent=None, **kwargs):
        super().__init__(parent)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Sphere", "Ellipsoid", "Cylinder"])
        self.shape_size_nm = QDoubleSpinBox()
        self.shape_size_nm.setRange(0.1, 1e9)
        self.shape_size_nm.setDecimals(3)
        self.shape_size_nm.setValue(5.0)
        self.shape_aspect = QDoubleSpinBox()
        self.shape_aspect.setRange(0.1, 1e3)
        self.shape_aspect.setDecimals(3)
        self.shape_aspect.setValue(1.0)
        self.btn_apply_shape = QPushButton("Apply shape→D")
        g = QGridLayout(self)
        g.setContentsMargins(0, 2, 0, 2)
        g.addWidget(QLabel("Type"), 0, 0)
        g.addWidget(self.shape_combo, 0, 1)
        g.addWidget(QLabel("Size (nm)"), 1, 0)
        g.addWidget(self.shape_size_nm, 1, 1)
        g.addWidget(QLabel("Aspect"), 2, 0)
        g.addWidget(self.shape_aspect, 2, 1)
        g.addWidget(self.btn_apply_shape, 3, 0, 1, 2)


class _JsonSection(QWidget):
    """Export / import settings buttons."""

    def __init__(self, model, target=None, parent=None, **kwargs):
        super().__init__(parent)
        self.btn_export_json = QPushButton("Export JSON")
        self.btn_import_json = QPushButton("Import JSON")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.addWidget(self.btn_export_json)
        row.addWidget(self.btn_import_json)
        row.addStretch(1)


def _register_confocal_sections() -> None:
    from chisurf.gui.autoform.sections.registry import register_section
    register_section("confocal_constraint")(_ConstraintSection)
    register_section("confocal_dye")(_DyeSection)
    register_section("confocal_shape")(_ShapeSection)
    register_section("confocal_json")(_JsonSection)


@persist_plugin_state("fcs_calculator")
class ConfocalCalcWidget(QWidget):
    """Interactive FCS confocal diffusion/volume calculator.

    Links FCS fit parameters (τ, D, r_h, Veff, N, concentration) with
    temperature-dependent viscosity, reference dyes (D @ 25 °C, water) and
    basic molecular-shape models (sphere, ellipsoid, cylinder).
    """

    def __init__(self, parent=None):
        """Initialize the calculator widget, internal state, and default values.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("FCS Confocal Calculator — τ, D, rₕ, Veff, Concentration")
        self._in_update = False
        self._last_edited = 'conc'  # driver for N↔c coupling
        from .gui.client import ConfocalCalcClient
        self._client = ConfocalCalcClient()
        self._setup_ui()
        self._connect_signals()
        self._recompute()

    def _setup_ui(self):
        """Render the editor from fcs_calculator.view.json via AutoForm.

        The numeric fields, the water-η toggle, and the constraint / dye / shape
        / JSON blocks are declared in the view-spec; here we grab the live editor
        widgets (and the custom-section children) into ``self.*`` under the same
        names the compute logic already uses, so that logic is unchanged.
        """
        _register_confocal_sections()
        self._model = _ConfocalModel()

        from chisurf.gui.autoform import AutoForm
        from chisurf.gui.autoform.sections.builtin import ToggleWidget, ValueWidget

        form = AutoForm(self._model, parent=self)
        self._form = form

        editors = {}
        for vw in form.findChildren(ValueWidget):
            attr = getattr(getattr(vw, "_section", None), "attr", None)
            if attr:
                editors[attr] = vw.editor
        self.tau_us = editors["tau_us"]
        self.D_um2_s = editors["D_um2_s"]
        self.rh_nm = editors["rh_nm"]
        self.S = editors["S"]
        self.veff_fL = editors["veff_fL"]
        self.temp_C = editors["temp_C"]
        self.eta_mPa_s = editors["eta_mPa_s"]
        self.invN = editors["invN"]
        self.num_mols = editors["num_mols"]
        self.conc_nM = editors["conc_nM"]

        for tw in form.findChildren(ToggleWidget):
            if getattr(getattr(tw, "_section", None), "attr", None) == "use_water_eta":
                self.use_water_eta = tw.checkbox

        cw = form.findChildren(_ConstraintSection)[0]
        self.rb_fix_D, self.rb_fix_rh, self.rb_fix_V = cw.rb_fix_D, cw.rb_fix_rh, cw.rb_fix_V
        self.fix_group = cw.fix_group

        dw = form.findChildren(_DyeSection)[0]
        self.dye_combo = dw.dye_combo
        self.btn_apply_dref = dw.btn_apply_dref
        self.scale_dref, self.scale_dref_none = dw.scale_dref, dw.scale_dref_none
        self.scale_group = dw.scale_group

        sw = form.findChildren(_ShapeSection)[0]
        self.shape_combo = sw.shape_combo
        self.shape_size_nm, self.shape_aspect = sw.shape_size_nm, sw.shape_aspect
        self.btn_apply_shape = sw.btn_apply_shape

        jw = form.findChildren(_JsonSection)[0]
        self.btn_export_json, self.btn_import_json = jw.btn_export_json, jw.btn_import_json

        # Removed buttons that the handlers tolerate being absent.
        self.btn_show_refs = None
        self.btn_help = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(form)
        self.setLayout(root)

        self._update_field_enable()
        # Initialize water viscosity mode default
        self._on_use_water_eta(self.use_water_eta.isChecked())

    def _cfg(self, sb: QDoubleSpinBox, lo: float, hi: float, dec: int, val: float):
        """Configure a ``QDoubleSpinBox`` with range, decimals, and value.

        Parameters
        ----------
        sb : QDoubleSpinBox
            The spin box to configure.
        lo : float
            Minimum value.
        hi : float
            Maximum value.
        dec : int
            Number of decimals.
        val : float
            Initial value.
        """
        sb.setRange(lo, hi); sb.setDecimals(dec); sb.setValue(val)

    def _connect_signals(self):
        """Wire Qt signals to the widget's update and recompute slots."""
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

        # Shape actions
        self.shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        self.btn_apply_shape.clicked.connect(self._apply_shape_to_D)

        # JSON actions
        self.btn_export_json.clicked.connect(self._export_json)
        self.btn_import_json.clicked.connect(self._import_json)

    # ---------- Constraint / UI state ----------
    def _on_constraint_changed(self, _btn, _state):
        """Handle the user switching the active constraint radio button.

        Parameters
        ----------
        _btn : QRadioButton
            The toggled button (unused).
        _state : bool
            The new checked state (unused).
        """
        self._update_field_enable()
        self._recompute()

    def _update_field_enable(self):
        """Set the read-only state of the D, rₕ and Veff fields based on the active constraint."""
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
        """Enable/disable the manual viscosity spin box and recompute.

        Parameters
        ----------
        checked : bool
            Whether the "Use water η(T)" checkbox is now checked.
        """
        self.eta_mPa_s.setEnabled(not checked)
        self._recompute()

    # ---------- Coupling helpers ----------
    def _conc_changed(self, val: float):
        """Update ``N`` and ``1/N`` when the concentration spin box changes.

        Parameters
        ----------
        val : float
            New concentration in nM.
        """
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
        """Update the concentration and ``1/N`` spin boxes when ``N`` changes.

        Parameters
        ----------
        val : float
            New number of molecules.
        """
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
        """Update ``N`` and the concentration when the ``1/N`` field changes.

        Parameters
        ----------
        val : float
            New value of 1/N.
        """
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
        """Apply the selected reference dye's ``D_25`` (with optional T/η scaling) to the ``D`` field."""
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

    def _on_shape_changed(self, _index: int):
        """Enable the aspect-ratio input only when the shape is not a sphere.

        Parameters
        ----------
        _index : int
            Unused current index of the shape combo.
        """
        shape = self.shape_combo.currentText()
        self.shape_aspect.setEnabled(shape != "Sphere")

    def _apply_shape_to_D(self):
        """Compute ``D`` from the selected shape (Sphere/Ellipsoid/Cylinder) and apply it to the ``D`` field."""
        if self._in_update:
            return
        shape = self.shape_combo.currentText()
        try:
            T_K = self.temp_C.value() + 273.15
            eta = self._current_eta_Pa_s()
            size_nm = self.shape_size_nm.value()
            aspect = self.shape_aspect.value()
            if size_nm <= 0:
                return
            D_use = None
            if shape == "Sphere":
                r_m = nm_to_m(size_nm) / 2.0
                D_m2_s = stokes_einstein_D(T_K, eta, r_m)
                D_use = m2s_to_um2s(D_m2_s)
            elif shape == "Ellipsoid":
                if aspect <= 0:
                    return
                b_m = nm_to_m(size_nm) / 2.0
                a_m = b_m * aspect
                D_use = diffusion_ellipsoid(T_K, eta, a_m, b_m)
            elif shape == "Cylinder":
                if aspect <= 0:
                    return
                d_m = nm_to_m(size_nm)
                L_m = d_m * aspect
                D_use = diffusion_cylinder(T_K, eta, L_m, d_m)
            if D_use is None or not math.isfinite(D_use) or D_use <= 0:
                return
            self.rb_fix_D.setChecked(True)
            self._update_field_enable()
            self._set_spin(self.D_um2_s, D_use)
            self._recompute()
            self.D_um2_s.setFocus()
        except Exception:
            pass

    def _collect_settings(self) -> Dict:
        """Return a dictionary snapshot of all current settings (for JSON export).

        Returns
        -------
        dict
            Mapping from setting name to its current value.
        """
        shape = self.shape_combo.currentText()
        fix_mode = "D" if self.rb_fix_D.isChecked() else ("rh" if self.rb_fix_rh.isChecked() else "V")
        return {
            "tau_us": self.tau_us.value(),
            "D_um2_s": self.D_um2_s.value(),
            "rh_nm": self.rh_nm.value(),
            "S": self.S.value(),
            "veff_fL": self.veff_fL.value(),
            "temp_C": self.temp_C.value(),
            "eta_mPa_s": self.eta_mPa_s.value(),
            "conc_nM": self.conc_nM.value(),
            "num_mols": self.num_mols.value(),
            "use_water_eta": self.use_water_eta.isChecked(),
            "invN": self.invN.value(),
            "fix_mode": fix_mode,
            "dye": self.dye_combo.currentText(),
            "scale_dref": self.scale_dref.isChecked(),
            "shape_type": shape,
            "shape_size_nm": self.shape_size_nm.value(),
            "shape_aspect": self.shape_aspect.value(),
        }

    def _apply_settings(self, data: Dict):
        """Apply a settings dict (as produced by :meth:`_collect_settings`) to the widget.

        Parameters
        ----------
        data : dict
            Mapping from setting name to its value.
        """
        self._in_update = True
        try:
            if "tau_us" in data:
                self.tau_us.setValue(float(data["tau_us"]))
            if "D_um2_s" in data:
                self.D_um2_s.setValue(float(data["D_um2_s"]))
            if "rh_nm" in data:
                self.rh_nm.setValue(float(data["rh_nm"]))
            if "S" in data:
                self.S.setValue(float(data["S"]))
            if "veff_fL" in data:
                self.veff_fL.setValue(float(data["veff_fL"]))
            if "temp_C" in data:
                self.temp_C.setValue(float(data["temp_C"]))
            if "eta_mPa_s" in data:
                self.eta_mPa_s.setValue(float(data["eta_mPa_s"]))
            if "conc_nM" in data:
                self.conc_nM.setValue(float(data["conc_nM"]))
            if "num_mols" in data:
                self.num_mols.setValue(float(data["num_mols"]))
            if "use_water_eta" in data:
                self.use_water_eta.setChecked(bool(data["use_water_eta"]))
            if "invN" in data:
                self.invN.setValue(float(data["invN"]))
            fix_mode = data.get("fix_mode")
            if fix_mode == "D":
                self.rb_fix_D.setChecked(True)
            elif fix_mode == "rh":
                self.rb_fix_rh.setChecked(True)
            elif fix_mode == "V":
                self.rb_fix_V.setChecked(True)
            dye = data.get("dye")
            if dye and dye in DYE_DATA:
                idx = self.dye_combo.findText(dye)
                if idx >= 0:
                    self.dye_combo.setCurrentIndex(idx)
            if "scale_dref" in data:
                if data["scale_dref"]:
                    self.scale_dref.setChecked(True)
                else:
                    self.scale_dref_none.setChecked(True)
            shape = data.get("shape_type")
            if shape:
                idx = self.shape_combo.findText(shape)
                if idx >= 0:
                    self.shape_combo.setCurrentIndex(idx)
            if "shape_size_nm" in data:
                self.shape_size_nm.setValue(float(data["shape_size_nm"]))
            if "shape_aspect" in data:
                self.shape_aspect.setValue(float(data["shape_aspect"]))
        finally:
            self._in_update = False
            self._update_field_enable()
            self._recompute()

    def _export_json(self):
        """Open a save dialog and write the current settings to a JSON file."""
        data = self._collect_settings()
        text = json.dumps(data, indent=2, sort_keys=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save FCS Calculator Settings",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except Exception:
            # Fail silently; caller can retry or ignore
            return

    def _import_json(self):
        """Open a load dialog and apply the selected JSON file's settings to the widget."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load FCS Calculator Settings",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            # Fail silently; caller can retry or ignore
            return
        if isinstance(data, dict):
            self._apply_settings(data)

    # References/help removed in optimized UI

    # ---------- Utility ----------
    def _current_eta_Pa_s(self) -> float:
        """Return the current viscosity in Pa·s, using water's model when enabled.

        Returns
        -------
        float
            Viscosity in Pa·s.
        """
        if self.use_water_eta.isChecked():
            T_K = self.temp_C.value() + 273.15
            eta = water_viscosity_Pa_s(T_K)
            # mirror into UI (disabled when checked)
            self._set_spin(self.eta_mPa_s, Pa_s_to_mPa_s(eta))
            return eta
        return mPa_s_to_Pa_s(self.eta_mPa_s.value())

    def _set_spin(self, spin: QDoubleSpinBox, value: float):
        """Set ``spin``'s value while suppressing update signals.

        Parameters
        ----------
        spin : QDoubleSpinBox
            The spin box to update.
        value : float
            New value.
        """
        self._in_update = True
        spin.setValue(value)
        self._in_update = False

    # ---------- Core recomputation ----------
    def _recompute(self):
        """Recompute the linked fields via the backend (RPC) confocal solver."""
        if self._in_update:
            return
        self._in_update = True
        try:
            if self.rb_fix_rh.isChecked():
                constraint = "rh"
            elif self.rb_fix_V.isChecked():
                constraint = "V"
            else:
                constraint = "D"

            r = self._client.compute(
                tau_us=self.tau_us.value(), S=self.S.value(),
                temp_C=self.temp_C.value(), eta_mPa_s=self.eta_mPa_s.value(),
                use_water_eta=bool(self.use_water_eta.isChecked()),
                constraint=constraint,
                D_um2_s=self.D_um2_s.value(), rh_nm=self.rh_nm.value(),
                veff_fL=self.veff_fL.value(), conc_nM=self.conc_nM.value(),
                num_mols=self.num_mols.value(), invN=self.invN.value(),
                last_edited=self._last_edited or "conc",
            )
            if not r.get("ok"):
                return
            d = r["result"]
            self._set_spin(self.D_um2_s, d["D_um2_s"])
            self._set_spin(self.rh_nm, d["rh_nm"])
            self._set_spin(self.veff_fL, d["veff_fL"])
            if self.use_water_eta.isChecked():
                self._set_spin(self.eta_mPa_s, d["eta_mPa_s"])
            self._set_spin(self.conc_nM, d["conc_nM"])
            self._set_spin(self.num_mols, d["num_mols"])
            self._set_spin(self.invN, d["invN"])
        finally:
            self._in_update = False


def main():
    """Run the FCS confocal calculator as a standalone application."""
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
