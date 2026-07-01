"""View-model backing the time-resolved anisotropy wizard.

Holds the wizard state (the four polarised file paths, the loaded curves, the
IRF background region, the instrument corrections and the lifetime/rotation
spectra) and exposes the ``source`` HTML methods, the ``complete_when`` booleans
and the ``action`` callbacks referenced by ``anisotropy.view.json``. The pure
numerics live in :mod:`...core` (IRF correction, spectrum I/O, the VV/VH link
plan); this class orchestrates data loading and fit construction, which are
inherently coupled to the process-global ChiSurf session and its GUI fit models
(imported lazily). Interactive pieces (the IRF region selector, the component
tables) are embedded Qt widgets that read/write this model.
"""

from __future__ import annotations

import json
import logging
import os.path
import pathlib
import typing
from collections.abc import Callable

import numpy as np

from ..core import fits as core_fits
from ..core import irf as core_irf
from ..core import spectra as core_spectra

logger = logging.getLogger(__name__)

_VIEW_JSON = pathlib.Path(__file__).parent.parent / "anisotropy.view.json"

_WELCOME_HTML = """
<h3>Time-Resolved Anisotropy</h3>
<p>This wizard builds a linked VV/VH global anisotropy fit from polarisation-resolved
decays.</p>
<ol>
<li><b>Data</b> — point to the VV/VH IRF and sample decays.</li>
<li><b>IRF</b> — pick a background region; the IRFs are background-subtracted and
intensity-matched.</li>
<li><b>Corrections</b> — set the g-factor and l1/l2 channel-mixing factors.</li>
<li><b>Components</b> — define the lifetime and rotation spectra.</li>
<li><b>Finish</b> — create the VV, VH and global fits with all parameters linked.</li>
</ol>
"""

_FINISH_HTML = """
<h3>Create the anisotropy fit</h3>
<p>Two lifetime fits (VV, VH) are created and wrapped in a global fit. The VH
channel is linked to VV for the photon count, rotation components, lifetimes and
instrument corrections, so the anisotropy is fit consistently across both
polarisations.</p>
<p>Click <b>Create fits</b> to add them to ChiSurf.</p>
"""


class AnisotropyViewModel:
    """State + view wiring for the anisotropy wizard (numerics live in ``core``)."""

    def view_spec(self):
        """Resolve AutoForm's view spec from ``anisotropy.view.json``."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_VIEW_JSON)

    def __init__(self) -> None:
        self._observers: list[Callable[[str], None]] = []
        # file paths (bound to value/file sections)
        self.irf_vv_path = ""
        self.irf_vh_path = ""
        self.data_vv_path = ""
        self.data_vh_path = ""
        # loaded curves + corrected IRFs
        self.data: dict[str, typing.Any] = {
            "irf_vv": None,
            "irf_vh": None,
            "data_vv": None,
            "data_vh": None,
            "irf_vv_bg_norm": None,
            "irf_vh_bg_norm": None,
        }
        # background region (channel indices)
        self.region_lb = 0
        self.region_ub = 100
        # instrument corrections (loaded from / saved to the settings JSON)
        self._corrections = {"g_factor": 1.0, "l1": 0.0, "l2": 0.0}
        self._load_corrections()
        # spectra (lists of [amplitude, value] pairs)
        self.spk_path = ""
        self.lifetime_spectrum: list[list[float]] = list(
            core_spectra.DEFAULT_SPECTRA["lifetime_spectrum"]
        )
        self.rotation_spectrum: list[list[float]] = list(
            core_spectra.DEFAULT_SPECTRA["rotation_spectrum"]
        )
        self._load_default_spectra()
        self._status_html = ""

    # ── observer hook ──────────────────────────────────────────────────
    def add_observer(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with an event name on every change."""
        self._observers.append(cb)

    def notify(self, event: str = "changed") -> None:
        """Notify observers (host refreshes info panels / ✓ marks)."""
        for cb in list(self._observers):
            try:
                cb(event)
            except Exception:
                logger.debug("anisotropy observer failed", exc_info=True)

    def update(self) -> None:
        """Refresh hook used by bound sections/widgets after editing state."""
        self.notify("refresh")

    # ── corrections (settings JSON) ─────────────────────────────────────
    def _corrections_path(self) -> pathlib.Path:
        import chisurf.core.settings

        return chisurf.core.settings.chisurf_settings_path / "anisotropy_corrections.json"

    def _load_corrections(self) -> None:
        try:
            path = self._corrections_path()
            if path.exists():
                self._corrections.update(json.loads(path.read_text()))
        except Exception:
            logger.debug("anisotropy: could not load corrections", exc_info=True)

    def _save_corrections(self) -> None:
        try:
            self._corrections_path().write_text(json.dumps(self._corrections))
        except Exception:
            logger.debug("anisotropy: could not save corrections", exc_info=True)

    @property
    def g_factor(self) -> float:
        """The polarisation g-factor (VV/VH detection-efficiency ratio)."""
        return float(self._corrections.get("g_factor", 1.0))

    @g_factor.setter
    def g_factor(self, value: float) -> None:
        self._corrections["g_factor"] = float(value)
        self._save_corrections()

    @property
    def l1(self) -> float:
        """Channel-mixing correction l1."""
        return float(self._corrections.get("l1", 0.0))

    @l1.setter
    def l1(self, value: float) -> None:
        self._corrections["l1"] = float(value)
        self._save_corrections()

    @property
    def l2(self) -> float:
        """Channel-mixing correction l2."""
        return float(self._corrections.get("l2", 0.0))

    @l2.setter
    def l2(self, value: float) -> None:
        self._corrections["l2"] = float(value)
        self._save_corrections()

    # ── spectra ─────────────────────────────────────────────────────────
    def _load_default_spectra(self) -> None:
        try:
            path = core_spectra.spk_json_path()
            spectra = core_spectra.load_spectra(path)
            if spectra["lifetime_spectrum"]:
                self.lifetime_spectrum = spectra["lifetime_spectrum"]
            if spectra["rotation_spectrum"]:
                self.rotation_spectrum = spectra["rotation_spectrum"]
            self.spk_path = str(path)
        except Exception:
            logger.debug("anisotropy: could not load default spectra", exc_info=True)

    def save_spectra(self) -> None:
        """Persist the current spectra to the last (or default) ``*.spk.json``."""
        path = self.spk_path or str(core_spectra.spk_json_path())
        core_spectra.save_spectra(path, self.lifetime_spectrum, self.rotation_spectrum)
        self.spk_path = path
        self.notify("refresh")

    def load_spectra(self, path: str) -> None:
        """Load both spectra from *path* into the model."""
        spectra = core_spectra.load_spectra(path)
        self.lifetime_spectrum = spectra["lifetime_spectrum"]
        self.rotation_spectrum = spectra["rotation_spectrum"]
        self.spk_path = str(path)
        self.notify("refresh")

    # ── data loading ────────────────────────────────────────────────────
    def _pairs(self) -> list[tuple[str, str, str]]:
        """Return ``(key, polarization, path)`` for the four polarised inputs.

        For a Jordi-format setup a single IRF file and a single data file hold
        both polarisations, so each is loaded twice.
        """
        import chisurf as cs

        setup = getattr(getattr(cs, "cs", object()), "current_setup", None)
        if getattr(setup, "is_jordi", False):
            return [
                ("irf_vv", "vv", self.irf_vv_path),
                ("irf_vh", "vh", self.irf_vv_path),
                ("data_vv", "vv", self.data_vv_path),
                ("data_vh", "vh", self.data_vv_path),
            ]
        return [
            ("irf_vv", "vv", self.irf_vv_path),
            ("irf_vh", "vh", self.irf_vh_path),
            ("data_vv", "vv", self.data_vv_path),
            ("data_vh", "vh", self.data_vh_path),
        ]

    def files_ready(self) -> bool:
        """Whether every required input file exists."""
        return all(pathlib.Path(p).is_file() for _, _, p in self._pairs())

    def load_data(self) -> bool:
        """Load the polarised IRF/data curves through the current ChiSurf reader.

        Returns ``True`` on success. Populates ``self.data`` and computes an
        initial IRF background region.
        """
        import chisurf as cs

        cs_mod = getattr(cs, "cs", object())
        setup = getattr(cs_mod, "current_setup", None)
        for key, suffix, filename in self._pairs():
            if setup is not None:
                setup.polarization = suffix
            reader = getattr(setup, "experiment_reader", None) or getattr(
                cs_mod, "current_experiment_reader", None
            )
            if reader is None:
                logger.warning("anisotropy: no experiment reader available")
                return False
            name = os.path.splitext(filename)[0] + suffix
            dataset = reader.get_data(filename=filename, name=name)[0]
            base, _ = os.path.splitext(dataset.name)
            dataset.name = base + "_" + suffix
            self.data[key] = dataset

        irf_vv = self.data.get("irf_vv")
        if irf_vv is not None:
            self.region_lb, self.region_ub = core_irf.initial_region(len(irf_vv.x))
        self.apply_region(self.region_lb, self.region_ub)
        self.notify("refresh")
        return True

    def apply_region(self, lb: int, ub: int) -> None:
        """Recompute the corrected IRFs for background region ``[lb, ub)``."""
        self.region_lb, self.region_ub = int(lb), int(ub)
        irf_vv, irf_vh = self.data.get("irf_vv"), self.data.get("irf_vh")
        if irf_vv is None or irf_vh is None:
            return
        vv, vh = core_irf.correct_irfs(irf_vv.y, irf_vh.y, lb, ub)
        self.data["irf_vv_bg_norm"] = self._make_curve(irf_vv, vv, "_vv")
        self.data["irf_vh_bg_norm"] = self._make_curve(irf_vh, vh, "_vh")

    def _make_curve(self, template, y, suffix):
        import chisurf as cs
        import chisurf.core.data
        import chisurf.core.experiments

        experiment = getattr(getattr(cs, "cs", object()), "current_experiment", None)
        return chisurf.core.data.DataCurve(
            x=template.x,
            y=y,
            ey=template.ey,
            experiment=experiment,
            setup=chisurf.core.experiments.tcspc.TCSPCReader,
            name=os.path.splitext(template.name)[0] + suffix,
        )

    def plot_series(self) -> list[dict]:
        """Return raw + corrected IRF series for the interactive plot."""
        series = []
        specs = [
            ("irf_vv", "VV (raw)", "#3b78ff", 0.4, 2),
            ("irf_vh", "VH (raw)", "#ff5b5b", 0.4, 2),
            ("irf_vv_bg_norm", "VV (corrected)", "#0040ff", 1.0, 3),
            ("irf_vh_bg_norm", "VH (corrected)", "#d00000", 1.0, 3),
        ]
        for key, label, color, _alpha, width in specs:
            curve = self.data.get(key)
            if curve is None:
                continue
            y = np.asarray(curve.y, dtype=float)
            series.append(
                {
                    "x": np.arange(len(y)).tolist(),
                    "y": y.tolist(),
                    "name": label,
                    "color": color,
                    "width": width,
                }
            )
        return series

    # ── info sources ────────────────────────────────────────────────────
    def welcome_html(self) -> str:
        """Return the welcome-step introduction (HTML)."""
        return _WELCOME_HTML

    def data_html(self) -> str:
        """Return a live checklist of the four input files (HTML)."""
        rows = []
        for label, path in (
            ("IRF VV", self.irf_vv_path),
            ("IRF VH", self.irf_vh_path),
            ("Data VV", self.data_vv_path),
            ("Data VH", self.data_vh_path),
        ):
            ok = bool(path) and pathlib.Path(path).is_file()
            mark = "✅" if ok else "—"
            rows.append(f"<tr><td>{label}</td><td>{mark}</td><td>{path or ''}</td></tr>")
        return "<table cellpadding='3'>" + "".join(rows) + "</table>"

    def finish_html(self) -> str:
        """Return the finish-step explanation (HTML)."""
        return _FINISH_HTML + (self._status_html or "")

    # ── completion booleans ─────────────────────────────────────────────
    @property
    def data_ready(self) -> bool:
        """Whether all four input files exist."""
        return self.files_ready()

    @property
    def components_ready(self) -> bool:
        """Whether both spectra have at least one component."""
        return len(self.lifetime_spectrum) >= 1 and len(self.rotation_spectrum) >= 1

    # ── fit creation ────────────────────────────────────────────────────
    def create_fits(self) -> None:
        """Create the VV, VH and global fits and wire the VV↔VH links.

        Delegates the parameter link/constraint sequence to
        :func:`...core.fits.build_link_plan` / :func:`...core.fits.apply_link_plan`.
        """
        import chisurf as cs
        import chisurf.core.actions
        from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client

        if not (self.data_ready and self.components_ready):
            self._set_status(False, "Data or components missing.")
            return
        if self.data.get("irf_vv_bg_norm") is None:
            self._set_status(False, "Load data and set the IRF region first.")
            return

        # make the corrected IRFs and sample decays available as datasets
        datasets = getattr(cs, "imported_datasets", None)
        if datasets is not None:
            for key in ("irf_vv_bg_norm", "irf_vh_bg_norm", "data_vv", "data_vh"):
                if self.data.get(key) is not None:
                    datasets.append(self.data[key])
        selector = getattr(getattr(cs, "cs", object()), "dataset_selector", None)
        if selector is not None:
            selector.update()

        n = len(getattr(cs, "imported_datasets", []))
        fc = get_fitting_client()

        chisurf.core.actions.dispatch(
            name="fit.add",
            payload={
                "model_name": "Lifetime fit",
                "dataset_indices": [n - 2, n - 1],
                "model_kw": dict(self._corrections),
            },
        )
        fit_vv = fc.get_fit_objects()[-2]
        fit_vh = fc.get_fit_objects()[-1]
        vv_idx = len(fc.get_fit_objects()) - 2
        vh_idx = len(fc.get_fit_objects()) - 1

        # lifetimes
        lt = core_spectra.flatten(self.lifetime_spectrum)
        fit_vv.model.lifetimes.pop()
        fit_vh.model.lifetimes.pop()
        for i in range(0, len(lt), 2):
            fit_vv.model.lifetimes.append(lt[i], lt[i + 1])
            fit_vh.model.lifetimes.append(lt[i], lt[i + 1])

        # rotations
        rs = core_spectra.flatten(self.rotation_spectrum)
        fit_vv.model.anisotropy.radioButtonVV.setChecked(True)
        fit_vv.model.anisotropy.hide_roation_parameters()
        fit_vh.model.anisotropy.radioButtonVH.setChecked(True)
        fit_vh.model.anisotropy.hide_roation_parameters()
        fit_vv.model.anisotropy.remove_rotation()
        fit_vh.model.anisotropy.remove_rotation()
        for i in range(0, len(rs), 2):
            fit_vv.model.anisotropy.add_rotation(b=rs[i], rho=rs[i + 1])
            fit_vh.model.anisotropy.add_rotation(b=rs[i], rho=rs[i + 1])

        # IRF
        for fit, key in ((fit_vv, "irf_vv_bg_norm"), (fit_vh, "irf_vh_bg_norm")):
            fit.model.convolve._irf = self.data[key]
            fit.model.convolve.lineEdit.setText(self.data[key].name)
            fit.update()

        # global fit wrapping VV + VH
        chisurf.core.actions.dispatch(
            name="fit.add", payload={"model_name": "Global fit", "dataset_indices": [0]}
        )
        global_fit = fc.get_fit_objects()[-1]
        global_fit.model.append_fit(fit_vv)
        global_fit.model.append_fit(fit_vh)

        fit_vv.model.anisotropy.polarization_type = "vv"
        fit_vh.model.anisotropy.polarization_type = "vh"

        plan = core_fits.build_link_plan(
            n_lifetime=len(lt) // 2, n_rotation=len(rs) // 2, corrections=self._corrections
        )
        core_fits.apply_link_plan(fc, plan, vv_idx, vh_idx)

        fit_vv.update()
        fit_vh.update()
        self._set_status(True, "Created VV, VH and global anisotropy fits.")

    def _set_status(self, ok: bool, msg: str) -> None:
        color = "#2e7d32" if ok else "#c62828"
        self._status_html = f"<p style='color:{color};font-weight:600'>{msg}</p>"
        self.notify("refresh")


__all__ = ["AnisotropyViewModel"]
