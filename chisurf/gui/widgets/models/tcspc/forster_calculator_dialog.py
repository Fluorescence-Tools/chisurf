"""Förster radius calculator widget backed by MFDB spectra."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
from qtpy import QtCore, QtWidgets

from chisurf.core.fluorescence.fret.forster import forster_radius_from_spectra
from chisurf.core.mfdb.database_resolver import resolve_database_path
from chisurf.core.mfdb.repository import MFDatabase

_EXT_COEFF_ALIASES = frozenset({
    "ext_coeff", "molar_extinction", "molar_ec", "epsilon",
    "extinction coefficient", "ext_coeff_max", "ec_max",
})

_QY_ALIASES = frozenset({
    "qy", "quantum yield", "phi", "phi_d", "qy_d", "quantum_yield",
    "fluorescence quantum yield", "q_fluor",
})

_VIEW_JSON = pathlib.Path(__file__).parent / "forster_calculator.view.json"


class _SearchableCombo(QtWidgets.QComboBox):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.completer().setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        self.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)


def _query_probes_with_spectrum(
    db: MFDatabase,
    spectrum_type: str,
) -> list[dict[str, Any]]:
    rows = db.conn.execute(
        """SELECT DISTINCT p.probe_id, p.chromophore_name
           FROM probes p
           JOIN spectra s ON s.probe_id = p.probe_id
           WHERE s.spectrum_type = ?
             AND p.deleted_at IS NULL
             AND s.deleted_at IS NULL
           ORDER BY p.chromophore_name""",
        (spectrum_type,),
    ).fetchall()
    return [dict(r) for r in rows]


def _get_spectra_for_probe(
    db: MFDatabase,
    probe_id: int,
) -> dict[str, dict[str, np.ndarray]]:
    rows = db.conn.execute(
        """SELECT spectrum_type, wavelengths, intensity_values
           FROM spectra
           WHERE probe_id = ?
             AND deleted_at IS NULL""",
        (probe_id,),
    ).fetchall()
    result: dict[str, dict[str, np.ndarray]] = {}
    for r in rows:
        stype = r["spectrum_type"]
        result[stype] = {
            "wavelengths": np.frombuffer(r["wavelengths"], dtype=np.float64),
            "intensity": np.frombuffer(r["intensity_values"], dtype=np.float64),
        }
    return result


def _lookup_optical_property(
    db: MFDatabase, probe_id: int, name_aliases: frozenset[str]
) -> float | None:
    for prop in db.get_optical_properties(probe_id):
        if prop["property_name"] in name_aliases:
            try:
                val = float(prop["property_value"])
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
    return None


def _try_lookup_cached_r0(
    db: MFDatabase, donor_name: str, acceptor_name: str
) -> float | None:
    row = db.conn.execute(
        """SELECT fr.forster_radius
           FROM flr_fret_forster_radius fr
           JOIN probes d ON d.probe_id = fr.donor_probe_id
           JOIN probes a ON a.probe_id = fr.acceptor_probe_id
           WHERE d.chromophore_name = ?
             AND a.chromophore_name = ?
             AND d.deleted_at IS NULL
             AND a.deleted_at IS NULL
           LIMIT 1""",
        (donor_name, acceptor_name),
    ).fetchone()
    return float(row[0]) if row else None


def _spectrum_tooltip(
    kind: str, data: dict[str, np.ndarray]
) -> str:
    wl = data["wavelengths"]
    return (
        f"{kind} · {len(wl)} pts · "
        f"{wl[0]:.0f}–{wl[-1]:.0f} nm"
    )


class ForsterCalculatorModel:
    """Backing model for the autoform parameter panel.

    Attributes mirror the ``attr`` keys in ``forster_calculator.view.json``.
    """

    def __init__(self) -> None:
        self.donor_qy = 0.9
        self.acceptor_emax = 100_000.0
        self.refractive_index = 1.33

    def view_spec(self):
        from chisurf.core.dataspec import load_view_spec
        return load_view_spec(_VIEW_JSON)


class ForsterCalculatorWidget(QtWidgets.QWidget):
    """A widget to calculate Förster radius from MFDB spectra.

    Queries the MFDB for probes with emission / absorption spectra, lets
    the user pick donor and acceptor probes via searchable combo boxes,
    and computes R₀ via the canonical overlap integral.  Emits
    ``forster_radius_calculated`` with the result in Ångström.

    The orientation factor κ² is fixed at 2/3 (dynamic isotropic averaging)
    — orientation corrections are handled elsewhere in the FRET model.
    """

    forster_radius_calculated = QtCore.Signal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Förster Radius Calculator")
        self._probe_names: dict[int, str] = {}
        self._probe_spectra: dict[int, dict[str, Any]] = {}
        self._last_r0: float | None = None
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._compute)
        self._setup_ui()
        self._load_probes()

    # ---- UI construction ------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Donor & Acceptor probes (collapsible)
        probe_group = QtWidgets.QGroupBox("Donor / Acceptor")
        probe_group.setCheckable(False)
        probe_layout = QtWidgets.QFormLayout(probe_group)
        self.donor_combo = _SearchableCombo()
        probe_layout.addRow("Donor emission:", self.donor_combo)
        self.acceptor_combo = _SearchableCombo()
        probe_layout.addRow("Acceptor absorption:", self.acceptor_combo)
        layout.addWidget(probe_group)

        # Parameter section (declarative — rendered from view.json by AutoForm)
        self._model = ForsterCalculatorModel()
        from chisurf.gui.autoform import AutoForm
        from chisurf.gui.autoform.sections.builtin import ValueWidget

        self._form = AutoForm(self._model, parent=self)
        layout.addWidget(self._form)

        # Extract editor widgets from the autoform
        self.qy_spin: QtWidgets.QDoubleSpinBox | None = None
        self.ext_spin: QtWidgets.QDoubleSpinBox | None = None
        self.n_spin: QtWidgets.QDoubleSpinBox | None = None
        for vw in self._form.findChildren(ValueWidget):
            attr = getattr(getattr(vw, "_section", None), "attr", None)
            if attr == "donor_qy":
                self.qy_spin = vw.editor
            elif attr == "acceptor_emax":
                self.ext_spin = vw.editor
            elif attr == "refractive_index":
                self.n_spin = vw.editor

        # Result section
        result_layout = QtWidgets.QHBoxLayout()
        self.result_label = QtWidgets.QLabel("")
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 14pt; font-weight: bold; padding: 8px;"
        )
        result_layout.addWidget(self.result_label, 1)

        self.apply_btn = QtWidgets.QPushButton("Apply R₀ to model")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply)
        result_layout.addWidget(self.apply_btn)
        layout.addLayout(result_layout)

        # Wire up auto-calculation triggers
        self.donor_combo.currentIndexChanged.connect(self._on_donor_changed)
        self.acceptor_combo.currentIndexChanged.connect(self._on_acceptor_changed)
        if self.qy_spin is not None:
            self.qy_spin.valueChanged.connect(self._schedule)
        if self.ext_spin is not None:
            self.ext_spin.valueChanged.connect(self._schedule)
        if self.n_spin is not None:
            self.n_spin.valueChanged.connect(self._schedule)

    # ---- data loading ---------------------------------------------------

    def _load_probes(self) -> None:
        try:
            db_path = resolve_database_path()
            db = MFDatabase(str(db_path), readonly=True)
        except Exception:
            self.result_label.setText("Could not open MFDB database.")
            self.apply_btn.setEnabled(False)
            return

        try:
            donors = _query_probes_with_spectrum(db, "emission")
            acceptors = _query_probes_with_spectrum(db, "absorption")

            all_ids: set[int] = set()
            for p in donors:
                all_ids.add(p["probe_id"])
                self._probe_names[p["probe_id"]] = p["chromophore_name"]
            for p in acceptors:
                all_ids.add(p["probe_id"])
                self._probe_names[p["probe_id"]] = p["chromophore_name"]

            self._probe_spectra = {}
            for pid in all_ids:
                spectra = _get_spectra_for_probe(db, pid)
                if spectra:
                    self._probe_spectra[pid] = spectra

            self.donor_combo.blockSignals(True)
            self.donor_combo.clear()
            self.donor_combo.addItem("", None)
            for p in donors:
                pid = p["probe_id"]
                spec = self._probe_spectra.get(pid, {})
                em = spec.get("emission")
                if em is not None:
                    self.donor_combo.addItem(p["chromophore_name"], pid)
                    idx = self.donor_combo.count() - 1
                    self.donor_combo.setItemData(
                        idx, _spectrum_tooltip("emission", em),
                        QtCore.Qt.ToolTipRole
                    )
            self.donor_combo.blockSignals(False)

            self.acceptor_combo.blockSignals(True)
            self.acceptor_combo.clear()
            self.acceptor_combo.addItem("", None)
            for p in acceptors:
                pid = p["probe_id"]
                spec = self._probe_spectra.get(pid, {})
                ab = spec.get("absorption")
                if ab is not None:
                    self.acceptor_combo.addItem(p["chromophore_name"], pid)
                    idx = self.acceptor_combo.count() - 1
                    self.acceptor_combo.setItemData(
                        idx, _spectrum_tooltip("absorption", ab),
                        QtCore.Qt.ToolTipRole
                    )
            self.acceptor_combo.blockSignals(False)
        finally:
            db.close()

    # ---- helpers --------------------------------------------------------

    def _db_for_read(self) -> MFDatabase | None:
        try:
            return MFDatabase(str(resolve_database_path()), readonly=True)
        except Exception:
            return None

    def _current_probe_id(self, combo: _SearchableCombo) -> int | None:
        idx = combo.currentIndex()
        pid = combo.itemData(idx)
        if pid is not None:
            return pid
        text = combo.currentText().strip()
        if not text:
            return None
        for i in range(combo.count()):
            if combo.itemText(i) == text:
                return combo.itemData(i)
        return None

    def _probe_name(self, pid: int | None) -> str:
        if pid is None:
            return "?"
        return self._probe_names.get(pid, "?")

    def _auto_fill_property(
        self,
        probe_id: int,
        aliases: frozenset[str],
        setter,
    ) -> None:
        db = self._db_for_read()
        if db is None:
            return
        try:
            val = _lookup_optical_property(db, probe_id, aliases)
            if val is not None:
                setter(val)
        finally:
            db.close()

    def _schedule(self) -> None:
        self._debounce.start()

    # ---- computation ----------------------------------------------------

    def _compute(self) -> None:
        donor_pid = self._current_probe_id(self.donor_combo)
        acceptor_pid = self._current_probe_id(self.acceptor_combo)

        if donor_pid is None or acceptor_pid is None:
            self.result_label.setText("")
            self._last_r0 = None
            self.apply_btn.setEnabled(False)
            return

        donor_spectra = self._probe_spectra.get(donor_pid, {})
        acceptor_spectra = self._probe_spectra.get(acceptor_pid, {})

        if "emission" not in donor_spectra or "absorption" not in acceptor_spectra:
            self.result_label.setText("")
            self._last_r0 = None
            self.apply_btn.setEnabled(False)
            return

        # Try MFDB cached R0 first
        donor_name = self._probe_name(donor_pid)
        acceptor_name = self._probe_name(acceptor_pid)
        db = self._db_for_read()
        if db is not None:
            try:
                cached = _try_lookup_cached_r0(db, donor_name, acceptor_name)
                if cached is not None and cached > 0:
                    self._last_r0 = cached
                    self.result_label.setText(f"R₀ = {cached:.1f} Å  (cached)")
                    self.apply_btn.setEnabled(True)
                    return
            finally:
                db.close()

        # Compute from spectra
        donor_em = donor_spectra["emission"]
        acceptor_abs = acceptor_spectra["absorption"]

        ext_coeff = self._model.acceptor_emax
        acceptor_ext = np.asarray(acceptor_abs["intensity"], dtype=float)
        peak = float(acceptor_ext.max())
        if peak > 0:
            acceptor_ext = acceptor_ext / peak * ext_coeff

        wl_min = max(
            float(np.min(donor_em["wavelengths"])),
            float(np.min(acceptor_abs["wavelengths"])),
        )
        wl_max = min(
            float(np.max(donor_em["wavelengths"])),
            float(np.max(acceptor_abs["wavelengths"])),
        )

        if wl_min >= wl_max:
            self._last_r0 = 0.0
            self.result_label.setText("R₀ = 0.0 Å  (no spectral overlap)")
            self.apply_btn.setEnabled(True)
            return

        from scipy.interpolate import interp1d

        common_wl = np.linspace(wl_min, wl_max, 1000)
        donor_interp = interp1d(
            donor_em["wavelengths"],
            donor_em["intensity"],
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        acceptor_interp = interp1d(
            acceptor_abs["wavelengths"],
            acceptor_ext,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        donor_int = donor_interp(common_wl)
        acceptor_ext_interp = acceptor_interp(common_wl)

        donor_qy = self._model.donor_qy
        n = self._model.refractive_index

        try:
            r0_angstrom, overlap_J = forster_radius_from_spectra(
                common_wl,
                donor_int,
                acceptor_ext_interp,
                donor_quantum_yield=donor_qy,
                kappa2=2.0 / 3.0,
                refractive_index=n,
            )
            self._last_r0 = r0_angstrom
            self.result_label.setText(
                f"R₀ = {r0_angstrom:.1f} Å    J = {overlap_J:.3e} M⁻¹ cm⁻¹ nm⁴"
            )
            self.apply_btn.setEnabled(True)
        except (ValueError, ZeroDivisionError) as exc:
            self._last_r0 = None
            self.result_label.setText(f"Error: {exc}")
            self.apply_btn.setEnabled(False)

    # ---- signal handlers -----------------------------------------------

    def _on_donor_changed(self) -> None:
        pid = self._current_probe_id(self.donor_combo)
        if pid is not None and self.qy_spin is not None:
            self._auto_fill_property(
                pid, _QY_ALIASES,
                lambda v: self.qy_spin.setValue(v),
            )
        self._schedule()

    def _on_acceptor_changed(self) -> None:
        pid = self._current_probe_id(self.acceptor_combo)
        if pid is not None and self.ext_spin is not None:
            self._auto_fill_property(
                pid, _EXT_COEFF_ALIASES,
                lambda v: self.ext_spin.setValue(v),
            )
        self._schedule()

    def _apply(self) -> None:
        if self._last_r0 is not None:
            self.forster_radius_calculated.emit(self._last_r0)
