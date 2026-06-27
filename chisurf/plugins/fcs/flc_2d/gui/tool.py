"""Dockable 2D-FLCS GUI backed by the plugin RPC API."""

from __future__ import annotations

import html
import json
import logging
import pathlib
from typing import Any

import numpy as np
from qtpy import QtCore, QtWidgets

from chisurf.core.math.regularization import LCurveData

from .client import FlcClient

_GUI_DIR = pathlib.Path(__file__).parent
logger = logging.getLogger(__name__)


class FlcHelpDialog(QtWidgets.QDialog):
    """Modal help dialog for the 2D-FLCS plugin."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create the help dialog."""
        super().__init__(parent)
        self.setWindowTitle("2D-FLCS Help")
        self.resize(760, 620)
        layout = QtWidgets.QVBoxLayout(self)
        text = QtWidgets.QTextBrowser(self)
        text.setOpenExternalLinks(True)
        text.setHtml(self._html())
        layout.addWidget(text, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    @staticmethod
    def _cli_help() -> str:
        """Return command-line help text."""
        try:
            from click.testing import CliRunner

            from ..cli import cli

            result = CliRunner().invoke(cli, ["--help"])
            return result.output or "CLI help unavailable."
        except Exception as exc:  # noqa: BLE001
            return f"CLI help unavailable: {exc}"

    @classmethod
    def _html(cls) -> str:
        """Return HTML help text."""
        cli_help = html.escape(cls._cli_help())
        return f"""
        <h2>2D-FLCS</h2>
        <p>Builds 2D fluorescence-decay correlation maps from TTTR photon
        streams, resolves lifetime species, and estimates exchange dynamics
        from lifetime-filtered species correlations.</p>
        <h3>Workflow</h3>
        <ol>
          <li>Open a TTTR file.</li>
          <li>Optionally load an IRF file or use synthetic/detected IRF mode.</li>
          <li>Set the lag window, lifetime grid, and inversion method.</li>
          <li>Run the analysis and inspect the docked maps and plots.</li>
        </ol>
        <h3>RPC</h3>
        <p>The backend exposes <code>flc2d.load_tttr</code>,
        <code>flc2d.correlate</code>, <code>flc2d.fit</code>,
        <code>flc2d.lifetime_spectrum</code>, and
        <code>flc2d.lifetime_lcurve</code>.</p>
        <h3>CLI</h3>
        <pre>{cli_help}</pre>
        """


class _FlcModel:
    """Settings model consumed by the declarative AutoForm view."""

    def __init__(self) -> None:
        """Initialize default 2D-FLCS settings and result buffers."""
        self.dT_ms = 1.0
        self.ddT_ms = 2.0
        self.tmin_ns = 0.0
        self.tmax_ns = 12.5
        self.max_bins = 80
        self.fit_mode = "nnls"
        self.n_components = 40
        self.tau_min_ns = 0.3
        self.tau_max_ns = 8.0
        self.log10_reg = 0.0
        self.irf_mode = "synthetic"
        self.irf_center_ns = 0.5
        self.irf_fwhm_ns = 0.15
        self.irf_shape = 0.0
        self.compute_dynamics = True
        self.n_casc = 25
        self.run_1d_mem = False
        self.mem_reg = 1000.0
        self.mem_mi_type = 0
        self.gaussian_components = 2
        self.irf_rise_scan = False
        self.run_global_mem = False
        self.n_lags = 6
        self.colormap = "viridis"
        self.sim_tau1_ns = 1.0
        self.sim_tau2_ns = 3.0
        self.sim_k12 = 30.0
        self.sim_k21 = 10.0
        self.sim_intensity_cps = 20000.0
        self.sim_time_s = 60.0
        self._lifetime: dict[str, np.ndarray] | None = None
        self._correlation: list[dict[str, Any]] | None = None
        self._mem1d: dict[str, np.ndarray] | None = None
        self._spectrum_img: np.ndarray | None = None
        self._residual_img: np.ndarray | None = None
        self._lcurve_data: LCurveData | None = None
        self._irf_preview: dict[str, np.ndarray] | None = None

    def view_spec(self):
        """Return the declarative AutoForm view spec."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_GUI_DIR / "flc_2d.view.json")

    def spectrum_image(self) -> np.ndarray | None:
        """Return the 2D lifetime-lifetime map."""
        return self._spectrum_img

    def residual_image(self) -> np.ndarray | None:
        """Return the 2D fit residual map."""
        return self._residual_img

    def irf_series(self) -> list[dict[str, Any]]:
        """Return decay and active IRF preview plot series."""
        if not self._irf_preview:
            return []
        series = [
            {
                "x": self._irf_preview["t"],
                "y": self._irf_preview["decay"],
                "color": "y",
                "name": "decay",
            }
        ]
        if "irf" in self._irf_preview:
            series.append(
                {
                    "x": self._irf_preview["t"],
                    "y": self._irf_preview["irf"],
                    "color": "c",
                    "name": "IRF",
                }
            )
        return series

    def lcurve_data(self) -> LCurveData | None:
        """Return 1D inversion L-curve diagnostics."""
        return self._lcurve_data

    def lifetime_series(self) -> list[dict[str, Any]]:
        """Return lifetime-distribution plot series."""
        series: list[dict[str, Any]] = []
        if self._lifetime:
            series.append(
                {
                    "x": self._lifetime["tau"],
                    "y": self._lifetime["amp"],
                    "color": "y",
                    "width": 2,
                    "name": "lifetime distribution",
                }
            )
        if self._mem1d:
            series.append(
                {
                    "x": self._mem1d["tau"],
                    "y": self._mem1d["amp"],
                    "color": "g",
                    "width": 2,
                    "name": "1D-MEM",
                }
            )
            if "gauss" in self._mem1d:
                series.append(
                    {
                        "x": self._mem1d["tau"],
                        "y": self._mem1d["gauss"],
                        "color": "r",
                        "width": 1,
                        "name": "Gaussian fit",
                    }
                )
        return series

    def correlation_series(self) -> list[dict[str, Any]]:
        """Return species-correlation plot series."""
        return self._correlation or []


class FlcTwoDTool(QtWidgets.QMainWindow):
    """Modern 2D-FLCS tool with dockable AutoForm panels."""

    tool_settings_name = "FlcTwoDTool"

    def __init__(self, parent=None) -> None:
        """Initialize the 2D-FLCS GUI."""
        super().__init__(parent)
        self.setWindowTitle("2D-FLCS")
        self.resize(1040, 660)
        self._client = FlcClient()
        self._model = _FlcModel()
        self._tttr = None
        self._tttr_path: str | None = None
        self._irf = None
        self._irf_time_ns = None
        self._irf_file = None
        self._irf_file_time_ns = None
        self._irf_path: str | None = None
        self._spectrum2d = None
        self._kinetics = None
        self.dock_area = None
        self._build_toolbar()
        self._build_central()

    def _build_toolbar(self) -> None:
        """Build the compact top toolbar."""
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("flcTwoDMainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)

        act_open = QtWidgets.QAction("Open", self)
        act_open.setToolTip("Open TTTR photon stream file.")
        act_open.triggered.connect(self._on_open)
        toolbar.addAction(act_open)

        act_irf = QtWidgets.QAction("IRF", self)
        act_irf.setToolTip("Load an instrument-response TTTR file.")
        act_irf.triggered.connect(self._on_open_irf)
        toolbar.addAction(act_irf)

        act_sim = QtWidgets.QAction("Sim", self)
        act_sim.setToolTip("Generate a synthetic two-state exchange photon stream.")
        act_sim.triggered.connect(self._on_simulate)
        toolbar.addAction(act_sim)

        toolbar.addSeparator()
        self.act_run = QtWidgets.QAction("Run", self)
        self.act_run.setToolTip("Build 2D-FDC and resolve lifetimes.")
        self.act_run.triggered.connect(self._on_run)
        self.act_run.setEnabled(False)
        toolbar.addAction(self.act_run)

        spacer = QtWidgets.QWidget(self)
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        act_help = QtWidgets.QAction("?", self)
        act_help.setToolTip("Show plugin help and CLI/RPC reference.")
        act_help.triggered.connect(self._show_help)
        toolbar.addAction(act_help)

    def _build_central(self) -> None:
        """Build the declarative docked AutoForm UI."""
        from chisurf.gui.autoform import AutoForm

        self._form = AutoForm(self._model, parent=self)
        self.setCentralWidget(self._form)
        self._settings_form = self._form
        self._plots_form = self._form
        self._configure_dock_area()
        self._restore_window_geometry()
        self.statusBar().showMessage("Open a TTTR file to begin.")

    def _configure_dock_area(self) -> None:
        """Enable dock visibility context menus and layout persistence."""
        areas = getattr(self._form, "_dock_areas", [])
        self.dock_area = areas[0] if areas else None
        if self.dock_area is None:
            return
        self.dock_area.setContextMenuEnabled(True)
        self.dock_area.setContextMenuMode("basic")
        self.dock_area.setTabsClosable(True)
        self.dock_area.layoutChanged.connect(self._save_dock_layout)
        self._restore_dock_layout()

    def _refresh_results(self) -> None:
        """Refresh all result docks."""
        self._form.refresh_plots()

    def _show_help(self) -> None:
        """Show modal plugin help."""
        FlcHelpDialog(self).exec_()

    def _settings(self) -> QtCore.QSettings:
        """Return persistent settings for the plugin shell."""
        return QtCore.QSettings("chisurf", self.tool_settings_name)

    def _save_window_geometry(self) -> None:
        """Persist the main window geometry."""
        try:
            settings = self._settings()
            settings.setValue("geometry", self.saveGeometry())
            settings.sync()
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Failed to save window geometry: {exc}")

    def _restore_window_geometry(self) -> None:
        """Restore the main window geometry."""
        try:
            geometry = self._settings().value("geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Failed to restore window geometry: {exc}")

    def _save_dock_layout(self) -> None:
        """Persist the dock layout."""
        if self.dock_area is None:
            return
        try:
            settings = self._settings()
            settings.setValue("dock_layout", json.dumps(self.dock_area.get_layout_state(), sort_keys=True))
            settings.sync()
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Failed to save dock layout: {exc}")

    def _restore_dock_layout(self) -> None:
        """Restore the saved dock layout."""
        if self.dock_area is None:
            return
        try:
            value = self._settings().value("dock_layout")
            if isinstance(value, str):
                layout_state = json.loads(value)
            elif isinstance(value, dict):
                layout_state = value
            else:
                return
            self.dock_area.set_layout_state(layout_state, emit_change=False)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Failed to restore dock layout: {exc}")

    def closeEvent(self, event) -> None:  # noqa: N802
        """Save state before closing."""
        self._save_window_geometry()
        self._save_dock_layout()
        super().closeEvent(event)

    def _tttr_from_payload(self, payload: dict[str, Any]):
        """Reconstruct ``TttrData`` from an RPC payload."""
        from .. import api

        return api.TttrData(
            macro_times=np.asarray(payload.get("macro_times", []), dtype=np.int64),
            micro_times=np.asarray(payload.get("micro_times", []), dtype=np.int64),
            routing_channels=np.asarray(payload.get("routing_channels", []), dtype=np.int64),
            macro_time_resolution_s=float(payload["macro_time_resolution_s"]),
            micro_time_resolution_ns=float(payload["micro_time_resolution_ns"]),
            n_microtime_channels=int(payload["n_microtime_channels"]),
        )

    def _on_open(self) -> None:
        """Open a TTTR file through the RPC client."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open TTTR file",
            "",
            "TTTR (*.ptu *.ht3 *.pt3 *.spc *.h5 *.hdf5);;All files (*)",
        )
        if not path:
            return
        try:
            self._tttr = self._tttr_from_payload(self._client.load_tttr(path, include_arrays=True))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._tttr_path = path
        data = self._tttr
        if data.micro_time_resolution_ns:
            self._model.tmax_ns = round(data.n_microtime_channels * data.micro_time_resolution_ns, 3)
            self._settings_form.sync_fields()
        self.act_run.setEnabled(True)
        self.statusBar().showMessage(
            f"Loaded {pathlib.Path(path).name}: {data.n_photons:,} photons, "
            f"{data.n_microtime_channels} TCSPC channels @ {data.micro_time_resolution_ns:.4g} ns"
        )

    def _on_open_irf(self) -> None:
        """Open an IRF TTTR file through the RPC client."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open IRF (TTTR) file",
            "",
            "TTTR (*.ptu *.ht3 *.pt3 *.spc *.h5 *.hdf5);;All files (*)",
        )
        if not path:
            return
        try:
            irf_data = self._tttr_from_payload(self._client.load_tttr(path, include_arrays=True))
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Load failed", str(exc))
            return
        n_channels = irf_data.n_microtime_channels
        hist = np.bincount(
            np.clip(np.asarray(irf_data.micro_times).astype(np.int64), 0, n_channels - 1),
            minlength=n_channels,
        )[:n_channels].astype(float)
        self._irf_file = hist
        self._irf_file_time_ns = np.arange(n_channels) * (irf_data.micro_time_resolution_ns or 1.0)
        self._irf_path = path
        self._model.irf_mode = "file"
        self._settings_form.sync_fields()
        self.statusBar().showMessage(f"Loaded IRF from {pathlib.Path(path).name} ({n_channels} channels)")

    def set_irf(self, irf: np.ndarray, irf_time_ns: np.ndarray) -> None:
        """Set the IRF directly for tests and scripts."""
        self._irf_file = np.asarray(irf, dtype=float)
        self._irf_file_time_ns = np.asarray(irf_time_ns, dtype=float)
        self._model.irf_mode = "file"

    def _resolve_irf(self, resolution_ns: float, n_channels: int, micro_times: np.ndarray) -> None:
        """Resolve the active IRF according to the selected mode."""
        from .. import api

        model = self._model
        time_ns = np.arange(int(n_channels)) * resolution_ns
        if model.irf_mode == "file":
            self._irf = self._irf_file
            self._irf_time_ns = self._irf_file_time_ns
        elif model.irf_mode == "synthetic":
            self._irf = api.make_synthetic_irf(time_ns, model.irf_center_ns, model.irf_fwhm_ns, shape=model.irf_shape)
            self._irf_time_ns = time_ns
        elif model.irf_mode == "detect":
            idx = np.asarray(micro_times).astype(np.int64)
            idx = idx[(idx >= 0) & (idx < n_channels)]
            decay = np.bincount(idx, minlength=n_channels)[:n_channels].astype(float)
            self._irf = api.detect_irf(decay, time_ns, fwhm_ns=model.irf_fwhm_ns, shape=model.irf_shape)
            self._irf_time_ns = time_ns
        else:
            self._irf = None
            self._irf_time_ns = None

    def _on_simulate(self) -> None:
        """Generate a synthetic photon stream and load it as the active dataset."""
        from .. import api

        model = self._model
        rate_matrix = np.array([[0.0, model.sim_k12], [model.sim_k21, 0.0]], dtype=float)
        self.statusBar().showMessage("Simulating photon stream...")
        QtWidgets.QApplication.processEvents()
        t_step_ns, n_channels = 0.004, 3127
        sim_time = np.arange(n_channels) * t_step_ns
        sim_irf = api.make_synthetic_irf(sim_time, model.irf_center_ns, model.irf_fwhm_ns, shape=model.irf_shape)
        try:
            stream = api.simulate_stream(
                rate_matrix,
                [model.sim_tau1_ns, model.sim_tau2_ns],
                [model.sim_intensity_cps, model.sim_intensity_cps],
                total_time_s=model.sim_time_s,
                irf=sim_irf,
                irf_time_ns=sim_time,
                tstep_ns=t_step_ns,
                n_microtime_channels=n_channels,
                seed=1,
            )
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Simulation failed", str(exc))
            return
        self._tttr = api.TttrData(
            macro_times=stream.macro_times,
            micro_times=stream.micro_times,
            routing_channels=np.zeros(stream.macro_times.size, dtype=np.int64),
            macro_time_resolution_s=stream.macro_time_resolution_s,
            micro_time_resolution_ns=stream.micro_time_resolution_ns,
            n_microtime_channels=stream.n_microtime_channels,
        )
        self._tttr_path = None
        self._model.tmax_ns = round(stream.n_microtime_channels * stream.micro_time_resolution_ns, 3)
        self._settings_form.sync_fields()
        self.act_run.setEnabled(True)
        self.statusBar().showMessage(
            f"Simulated {self._tttr.n_photons:,} photons "
            f"(tau={model.sim_tau1_ns:g}/{model.sim_tau2_ns:g} ns, "
            f"k12={model.sim_k12:g} k21={model.sim_k21:g}/s). Press Run."
        )

    @staticmethod
    def _estimate_populations(spec: dict[str, Any], peaks: np.ndarray) -> np.ndarray:
        """Estimate equilibrium populations from a lifetime distribution."""
        tau = np.asarray(spec["tau_grid"], dtype=float)
        amp = np.clip(np.asarray(spec["amplitudes"], dtype=float), 0.0, None)
        split = float(np.sqrt(peaks[0] * peaks[1]))
        area = amp * tau
        p_short = float(area[tau < split].sum())
        p_long = float(area[tau >= split].sum())
        total = p_short + p_long
        if total <= 0:
            return np.array([0.5, 0.5])
        return np.array([p_short, p_long]) / total

    def _run_lcurve(self, micro: np.ndarray, data: Any, method: str) -> None:
        """Compute and store L-curve diagnostics through RPC."""
        model = self._model
        payload = self._client.lifetime_lcurve(
            micro,
            n_microtime_bins=data.n_microtime_channels,
            micro_time_resolution_ns=data.micro_time_resolution_ns or 1.0,
            tau_range=(model.tau_min_ns, model.tau_max_ns),
            n_components=model.n_components,
            irf=self._irf,
            irf_time_ns=self._irf_time_ns,
            method=method,
        )
        self._model._lcurve_data = LCurveData(
            reg=np.asarray(payload["reg"], dtype=float),
            residual_norm=np.asarray(payload["residual_norm"], dtype=float),
            solution_norm=np.asarray(payload["solution_norm"], dtype=float),
            corner_index=int(payload["corner_index"]),
        )

    def _update_irf_preview(self, micro: np.ndarray, data: Any, resolution_ns: float) -> None:
        """Update normalized decay and IRF preview data."""
        idx = micro[(micro >= 0) & (micro < data.n_microtime_channels)]
        decay_hist = np.bincount(idx, minlength=data.n_microtime_channels).astype(float)
        t_full = np.arange(data.n_microtime_channels) * resolution_ns
        preview = {"t": t_full, "decay": decay_hist / (decay_hist.max() or 1.0)}
        if self._irf is not None:
            irf_on_t = np.interp(
                t_full,
                np.asarray(self._irf_time_ns, dtype=float),
                np.asarray(self._irf, dtype=float),
                left=0.0,
                right=0.0,
            )
            preview["irf"] = irf_on_t / (irf_on_t.max() or 1.0)
        self._model._irf_preview = preview

    def _on_run(self) -> None:
        """Run the current 2D-FLCS analysis."""
        if self._tttr is None:
            return
        from .. import api

        data = self._tttr
        model = self._model
        macro = np.ascontiguousarray(data.macro_times)
        order = np.argsort(macro, kind="stable")
        macro = macro[order].astype(np.int64)
        micro = np.ascontiguousarray(data.micro_times)[order].astype(np.int64)
        resolution_ns = data.micro_time_resolution_ns or 1.0
        t_min = max(1, int(round(model.tmin_ns / resolution_ns)))
        t_max = max(t_min + 1, int(round(model.tmax_ns / resolution_ns)))
        lag_ticks = max(1, int(round(model.dT_ms * 1e-3 / data.macro_time_resolution_s)))
        window_ticks = max(1, int(round(model.ddT_ms * 1e-3 / data.macro_time_resolution_s)))
        reg = None if model.log10_reg == 0.0 else 10.0**model.log10_reg

        self._resolve_irf(resolution_ns, data.n_microtime_channels, micro)

        self.statusBar().showMessage("Building 2D-FDC...")
        QtWidgets.QApplication.processEvents()
        out = self._client.correlate(
            macro,
            micro,
            dT=lag_ticks,
            ddT=window_ticks,
            tMin=t_min,
            tMax=t_max,
            logt_imax=60,
            max_bins=256,
        )
        mat = np.asarray(out["mat_lin"], dtype=float)
        time_ns = (np.asarray(out["mat_lin_t"], dtype=float) + 1) * resolution_ns

        self.statusBar().showMessage("Inverting 2D spectrum...")
        QtWidgets.QApplication.processEvents()
        try:
            spec2d = self._client.fit(
                mat,
                time_ns,
                mode=("mem" if model.fit_mode == "mem" else "tikhonov"),
                tau_range=(model.tau_min_ns, model.tau_max_ns),
                n_components=min(model.n_components, 32),
                irf=self._irf,
                irf_time_ns=self._irf_time_ns,
                reg=reg,
                max_bins=model.max_bins,
            )
            self._spectrum2d = spec2d
            self._model._spectrum_img = np.asarray(spec2d["spectrum"], dtype=float)
            residual = np.asarray(spec2d.get("residual", []), dtype=float)
            self._model._residual_img = residual if residual.size else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("2D spectrum failed: %s", exc)
            self._spectrum2d = None
            self._model._spectrum_img = np.log1p(mat) if mat.size else None
            self._model._residual_img = None

        self.statusBar().showMessage("Resolving lifetimes...")
        QtWidgets.QApplication.processEvents()
        method = "tikhonov" if model.fit_mode == "tikhonov" else "nnls"
        spec = self._client.lifetime_spectrum(
            micro,
            n_microtime_bins=data.n_microtime_channels,
            micro_time_resolution_ns=resolution_ns,
            tau_range=(model.tau_min_ns, model.tau_max_ns),
            n_components=model.n_components,
            irf=self._irf,
            irf_time_ns=self._irf_time_ns,
            method=method,
            reg=reg,
        )
        self._model._lifetime = {
            "tau": np.asarray(spec["tau_grid"], dtype=float),
            "amp": np.asarray(spec["amplitudes"], dtype=float),
        }
        peaks = np.sort(np.asarray(spec.get("peak_lifetimes", []), dtype=float))

        try:
            self._run_lcurve(micro, data, method)
        except Exception as exc:  # noqa: BLE001
            logger.warning("L-curve failed: %s", exc)
            self._model._lcurve_data = None

        self._update_irf_preview(micro, data, resolution_ns)
        self._model._correlation = []

        if model.compute_dynamics and peaks.size >= 2:
            self._run_dynamics(api, macro, micro, data, resolution_ns, peaks, spec)

        if not self._model._correlation:
            peaks_txt = ", ".join(f"{peak:.2f}" for peak in peaks)
            self.statusBar().showMessage(f"Resolved lifetimes: tau = {peaks_txt} ns")

        self._run_optional_mem(api, macro, micro, t_min, t_max, resolution_ns)
        self._refresh_results()

    def _run_dynamics(
        self,
        api,
        macro: np.ndarray,
        micro: np.ndarray,
        data: Any,
        resolution_ns: float,
        peaks: np.ndarray,
        spec: dict[str, Any],
    ) -> None:
        """Compute species-resolved correlations and kinetics."""
        model = self._model
        self.statusBar().showMessage("Computing species correlation...")
        QtWidgets.QApplication.processEvents()
        try:
            patterns = api.species_decay_patterns(
                peaks[:2],
                data.n_microtime_channels,
                resolution_ns,
                irf=self._irf,
                irf_time_ns=self._irf_time_ns,
            )
            dyn = api.species_correlation(
                macro,
                micro,
                patterns,
                None,
                data.macro_time_resolution_s,
                n_microtime_bins=data.n_microtime_channels,
                n_casc=int(model.n_casc),
            )
            series = []
            for index, values in dyn.correlation.auto.items():
                series.append({"x": dyn.correlation.lag_s, "y": values, "color": "c", "name": f"auto {index}"})
            for (i, j), values in dyn.correlation.cross.items():
                series.append({"x": dyn.correlation.lag_s, "y": values, "color": "m", "name": f"cross {i}-{j}"})
            self._model._correlation = series
            message = "tau = " + ", ".join(f"{peak:.2f}" for peak in peaks) + " ns"
            populations = self._estimate_populations(spec, peaks[:2])
            try:
                kinetics = api.rate_matrix_kinetics(dyn.correlation, n_states=2, populations=populations)
                self._kinetics = kinetics
                rate_sum = float(kinetics.relaxation_rates[0])
                message += f"; relaxation {kinetics.relaxation_times_s[0] * 1e3:.1f} ms"
                message += f" (k12+k21={rate_sum:.1f}/s)"
            except Exception as exc:  # noqa: BLE001
                logger.warning("rate-matrix fit failed: %s", exc)
                relax = dyn.relaxation.get("relaxation_time_s")
                if relax:
                    message += f"; relaxation {relax * 1e3:.1f} ms"
            self.statusBar().showMessage(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dynamics failed: %s", exc)

    def _run_optional_mem(
        self,
        api,
        macro: np.ndarray,
        micro: np.ndarray,
        t_min: int,
        t_max: int,
        resolution_ns: float,
    ) -> None:
        """Run optional 1D-MEM analysis and Gaussian decomposition."""
        model = self._model
        self._model._mem1d = None
        if not model.run_1d_mem:
            return
        self.statusBar().showMessage("Building 1D-FDC + 1D-MEM...")
        QtWidgets.QApplication.processEvents()
        try:
            fdc1d = api.one_d_fdc(macro, micro, tMin=t_min, tMax=t_max, max_bins=400)
            t1d_ns = fdc1d["lin_t"].astype(float) * resolution_ns
            if model.irf_rise_scan and self._irf is not None:
                rise = api.search_irf_rise(
                    fdc1d["lin"],
                    t1d_ns,
                    self._irf,
                    self._irf_time_ns,
                    tau_range=(model.tau_min_ns, model.tau_max_ns),
                    n_components=model.n_components,
                    mem_kwargs={"reg": model.mem_reg, "mi_type": int(model.mem_mi_type)},
                )
                tau_grid, amplitudes = rise.tau_grid, rise.averaged
            else:
                mem = api.lifetime_spectrum_mem(
                    fdc1d["lin"],
                    t1d_ns,
                    tau_range=(model.tau_min_ns, model.tau_max_ns),
                    n_components=model.n_components,
                    irf=self._irf,
                    irf_time_ns=self._irf_time_ns,
                    reg=model.mem_reg,
                    mi_type=int(model.mem_mi_type),
                )
                tau_grid, amplitudes = mem.tau_grid, mem.amplitudes
            self._model._mem1d = {"tau": tau_grid, "amp": amplitudes}
            gaussian = api.fit_gaussian_components(tau_grid, amplitudes, int(model.gaussian_components))
            self._model._mem1d["gauss"] = gaussian.model
        except Exception as exc:  # noqa: BLE001
            logger.warning("1D-MEM failed: %s", exc)
