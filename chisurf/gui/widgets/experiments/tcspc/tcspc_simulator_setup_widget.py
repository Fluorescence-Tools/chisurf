from __future__ import annotations

from chisurf.gui import QtWidgets

import pathlib

import numpy as np
import pyqtgraph as pg

import chisurf as cs
class TCSPCSimulatorSetupWidget(QtWidgets.QWidget):
    """Controller widget for the TCSPC simulator reader.

    This widget exposes the simulator parameters (sample name, lifetime
    spectrum, number of TAC channels, peak count, dt) and a small preview
    plot. It also provides:

    - An IRF selector with Gaussian fallback.
    - A **Simulate** button to generate a Poisson-noisy decay.
    - An **Add** button to append the simulated decay to
      ``cs.imported_datasets``.
    - A tool button next to the *lifetime spectrum* field that loads an
      interleaved (amplitude, lifetime, ...) spectrum from a CSV/text file
      into the line edit.
    """

    @cs.gui.decorators.init_with_ui("tcspc_simulator.ui")
    def __init__(self, *args, **kwargs):
        # Internal reference to the currently selected IRF dataset (if any)
        self._irf_dataset = None

        # Lifetime spectrum loader button next to the line edit.
        # We wrap the existing lineEdit_2 into a small row layout together
        # with a tool button so the .ui file does not need to change.
        try:
            le_lt = self.lineEdit_2
        except Exception:
            le_lt = None
        if le_lt is not None:
            try:
                grid = self.gridLayout
            except Exception:
                grid = None
            if isinstance(grid, QtWidgets.QGridLayout):
                try:
                    idx = grid.indexOf(le_lt)
                except Exception:
                    idx = -1
                if idx >= 0:
                    try:
                        row, col, row_span, col_span = grid.getItemPosition(idx)
                    except Exception:
                        row, col, row_span, col_span = 1, 1, 1, 1

                    # Remove from grid and put into a small container with a button.
                    grid.removeWidget(le_lt)
                    container = QtWidgets.QWidget(self)
                    hl = QtWidgets.QHBoxLayout(container)
                    hl.setContentsMargins(0, 0, 0, 0)
                    hl.setSpacing(2)
                    hl.addWidget(le_lt, 1)

                    self.toolButton_load_lifetime = QtWidgets.QToolButton(container)
                    self.toolButton_load_lifetime.setText("...")
                    self.toolButton_load_lifetime.setToolTip(
                        "Load lifetime spectrum from CSV/text file"
                    )
                    hl.addWidget(self.toolButton_load_lifetime, 0)

                    grid.addWidget(container, row, col, row_span, col_span)

                    try:
                        self.toolButton_load_lifetime.clicked.connect(
                            self._on_load_lifetime_clicked
                        )
                    except Exception:
                        pass

        # Hidden IRF selector window (shown on button click, similar to ConvolveWidget)
        # Do not pass a concrete Experiment instance as filter here; this would
        # break isinstance checks in ExperimentalDataSelector. Leaving
        # "experiment" unset mirrors ConvolveWidget behaviour and shows the
        # same imported datasets.
        self.irf_selector = cs.gui.widgets.experiments.ExperimentalDataSelector(
            click_close=True,
            parent=None,
            context_menu_enabled=False,
            change_event=self._on_irf_selection_changed,
        )

        # IRF selection row: label + "Select IRF" button
        irf_select_layout = QtWidgets.QHBoxLayout()
        irf_select_layout.setContentsMargins(0, 0, 0, 0)
        irf_select_layout.setSpacing(2)

        self.label_irf_source = QtWidgets.QLabel(
            "IRF: Gaussian (no dataset selected)",
            self,
        )
        irf_select_layout.addWidget(self.label_irf_source, 1)

        self.toolButton_select_irf = QtWidgets.QToolButton(self)
        self.toolButton_select_irf.setText("Select IRF")
        self.toolButton_select_irf.setToolTip(
            "Select IRF dataset from imported TCSPC curves"
        )
        irf_select_layout.addWidget(self.toolButton_select_irf)

        self.toolButton_unload_irf = QtWidgets.QToolButton(self)
        self.toolButton_unload_irf.setText("X")
        self.toolButton_unload_irf.setToolTip("Unload IRF and use Gaussian")
        irf_select_layout.addWidget(self.toolButton_unload_irf)

        self.verticalLayout_2.addLayout(irf_select_layout)

        # Gaussian fallback IRF parameters (used when no IRF dataset is selected)
        irf_gauss_layout = QtWidgets.QHBoxLayout()
        irf_gauss_layout.setContentsMargins(0, 0, 0, 0)
        irf_gauss_layout.setSpacing(2)
        irf_gauss_layout.addWidget(QtWidgets.QLabel("Gaussian IRF mean [ns]:"))
        self.doubleSpinBox_irf_mean = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox_irf_mean.setDecimals(3)
        self.doubleSpinBox_irf_mean.setRange(-1000.0, 1000.0)
        self.doubleSpinBox_irf_mean.setValue(5.0)
        irf_gauss_layout.addWidget(self.doubleSpinBox_irf_mean)
        irf_gauss_layout.addWidget(QtWidgets.QLabel("sigma [ns]:"))
        self.doubleSpinBox_irf_sigma = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox_irf_sigma.setDecimals(3)
        self.doubleSpinBox_irf_sigma.setRange(1e-4, 1000.0)
        self.doubleSpinBox_irf_sigma.setValue(0.2)
        irf_gauss_layout.addWidget(self.doubleSpinBox_irf_sigma)
        irf_gauss_layout.addStretch(1)
        self.verticalLayout_2.addLayout(irf_gauss_layout)

        # Simulation preview (simulate button + add button + plot)
        preview_group = QtWidgets.QGroupBox("Simulation preview", self)
        preview_group.setMaximumHeight(250)
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(0)
        header.addStretch(1)

        self.toolButton_simulate = QtWidgets.QToolButton(preview_group)
        self.toolButton_simulate.setText("Simulate")
        self.toolButton_simulate.setToolTip("Simulate TCSPC decay using current parameters")
        header.addWidget(self.toolButton_simulate)

        self.toolButton_add = QtWidgets.QToolButton(preview_group)
        self.toolButton_add.setText("Add")
        self.toolButton_add.setToolTip("Add simulated decay as dataset")
        header.addWidget(self.toolButton_add)

        preview_layout.addLayout(header)

        self.simulation_plot = pg.PlotWidget(title="Simulated TCSPC decay")
        self.simulation_plot.setLabel("bottom", "Time (ns)")
        self.simulation_plot.setLabel("left", "Counts")
        try:
            self.simulation_plot.setLogMode(y=True)
        except Exception:
            pass
        preview_layout.addWidget(self.simulation_plot)

        self.verticalLayout.addWidget(preview_group)

        # Internal state for last simulation
        self._sim_t = None
        self._sim_y = None

        self.actionParametersChanged.triggered.connect(self.onParametersChanged)
        try:
            self.toolButton_select_irf.clicked.connect(self._on_irf_button_clicked)
        except Exception:
            pass
        try:
            self.toolButton_unload_irf.clicked.connect(self._on_irf_unload_clicked)
        except Exception:
            pass
        try:
            self.toolButton_simulate.clicked.connect(self._on_simulate_clicked)
        except Exception:
            pass
        try:
            self.toolButton_add.clicked.connect(self._on_add_clicked)
        except Exception:
            pass

        self.onParametersChanged()

    def get_filename(self) -> pathlib.Path:
        return pathlib.Path(self.lineEdit.text())

    def updateUI(self):
        """Update UI elements based on current_setup properties."""
        # Get the current setup
        setup = cs.cs.current_setup

        # Update sample_name line edit
        if hasattr(setup, 'sample_name'):
            self.lineEdit.setText(setup.sample_name)

        # Update dt spin box
        if hasattr(setup, 'dt'):
            self.doubleSpinBox.setValue(setup.dt)

        # Update n_tac spin box
        if hasattr(setup, 'n_tac'):
            self.spinBox.setValue(setup.n_tac)

        # Update p0 spin box
        if hasattr(setup, 'p0'):
            self.spinBox_2.setValue(setup.p0)

        # Update lifetime_spectrum line edit
        if hasattr(setup, 'lifetime_spectrum'):
            self.lineEdit_2.setText(', '.join(map(str, setup.lifetime_spectrum)) if setup.lifetime_spectrum.size > 0 else '')

    def onParametersChanged(self):
        dt = self.doubleSpinBox.value()
        n_tac = self.spinBox.value()
        p0 = self.spinBox_2.value()
        sample_name = str(self.lineEdit.text())
        lt_text = self.lineEdit_2.text()
        cs.run(
            "\n".join(
                [
                    f"gui.current_setup.sample_name = '{sample_name}'",
                    f"gui.current_setup.dt = {dt}",
                    f"gui.current_setup.lifetime_spectrum = np.array([{lt_text}], dtype=np.float64)",
                    f"gui.current_setup.n_tac = {n_tac}",
                    f"gui.current_setup.p0 = {p0}"
                ]
            )
        )

    # ------------------------------------------------------------------
    # Simulation & IRF handling
    # ------------------------------------------------------------------

    def _on_load_lifetime_clicked(self) -> None:
        """Load a lifetime spectrum from a CSV/text file into the line edit.

        The loader accepts either a flat 1D list of values or a
        two-column (amplitude, lifetime) table. In both cases the values
        are converted into the interleaved ``a1, tau1, a2, tau2, ...``
        string used by the simulator. The line edit is updated and
        :meth:`onParametersChanged` is called so ``gui.current_setup``
        reflects the new spectrum.
        """

        from qtpy import QtWidgets as _QtWidgets
        import numpy as _np

        try:
            start_dir = getattr(cs, "working_path", "") or ""
        except Exception:
            start_dir = ""

        fn, _ = _QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open lifetime spectrum (CSV/text)",
            str(start_dir),
            "Data files (*.csv *.txt *.dat);;All files (*)",
        )
        if not fn:
            return

        try:
            data = _np.loadtxt(fn, dtype=float, ndmin=1)
        except Exception:
            return

        values: _np.ndarray
        try:
            arr = _np.asarray(data, dtype=float)
        except Exception:
            return

        if arr.ndim == 1:
            values = arr
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            # Use the first two columns as (amplitude, lifetime) pairs
            a = arr[:, 0].ravel()
            tau = arr[:, 1].ravel()
            n = min(a.size, tau.size)
            if n <= 0:
                return
            inter = _np.empty(2 * n, dtype=float)
            inter[0::2] = a[:n]
            inter[1::2] = tau[:n]
            values = inter
        else:
            return

        try:
            txt = ", ".join(f"{float(v):g}" for v in values)
        except Exception:
            return

        try:
            self.lineEdit_2.setText(txt)
        except Exception:
            return

        try:
            self.onParametersChanged()
        except Exception:
            pass

    def _parse_lifetime_spectrum(self) -> np.ndarray:
        """Parse the lifetime spectrum text into an interleaved numpy array."""

        text = self.lineEdit_2.text()
        parts = [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]
        values = []
        for p in parts:
            try:
                values.append(float(p))
            except Exception:
                continue
        return np.asarray(values, dtype=float)

    def _update_irf_label(self) -> None:
        """Update the IRF source label based on the currently selected dataset."""

        try:
            ds = getattr(self, "_irf_dataset", None)
        except Exception:
            ds = None

        if ds is None:
            text = "IRF: Gaussian (no dataset selected)"
        else:
            try:
                name = getattr(ds, "name", None) or getattr(ds, "filename", None)
            except Exception:
                name = None
            if not name:
                name = "IRF dataset"
            text = f"IRF: {name}"

        try:
            self.label_irf_source.setText(text)
        except Exception:
            pass

    def _on_irf_button_clicked(self) -> None:
        """Show the IRF selector window when the user clicks the select button."""

        try:
            self.irf_selector.show()
        except Exception:
            pass

    def _on_irf_unload_clicked(self) -> None:
        """Unload the currently selected IRF and revert to Gaussian mode."""

        try:
            self._irf_dataset = None
        except Exception:
            pass
        self._update_irf_label()

    def _on_irf_selection_changed(self):
        """Callback used by the hidden IRF selector when the selection changes."""

        ds = None
        try:
            ds = self.irf_selector.selected_dataset
        except Exception:
            ds = None

        try:
            self._irf_dataset = ds
        except Exception:
            pass

        self._update_irf_label()

    def _build_irf(self, time_axis: np.ndarray) -> np.ndarray:
        """Return an IRF on the given time axis.

        Preference order:
        1. Use the currently selected TCSPC dataset in the selector.
        2. Fall back to a Gaussian IRF defined by mean/std spin boxes.
        """

        import numpy as _np

        # Try to use the explicitly selected IRF dataset (if any)
        try:
            ds = getattr(self, "_irf_dataset", None)
        except Exception:
            ds = None

        if ds is not None:
            try:
                x_irf = _np.asarray(getattr(ds, 'x', []), dtype=float)
                y_irf = _np.asarray(getattr(ds, 'y', []), dtype=float)
            except Exception:
                x_irf = _np.zeros(0, dtype=float)
                y_irf = _np.zeros(0, dtype=float)

            if x_irf.size > 1 and y_irf.size == x_irf.size:
                order = _np.argsort(x_irf)
                x_sorted = x_irf[order]
                y_sorted = y_irf[order]
                irf_interp = _np.interp(time_axis, x_sorted, y_sorted, left=0.0, right=0.0)
                try:
                    max_val = float(_np.max(irf_interp))
                except Exception:
                    max_val = 0.0
                if max_val > 0.0:
                    irf_interp /= max_val
                return irf_interp

        # Fallback: Gaussian IRF defined by mean/std spin boxes
        try:
            mean = float(self.doubleSpinBox_irf_mean.value())
        except Exception:
            mean = 0.0
        try:
            sigma = float(self.doubleSpinBox_irf_sigma.value())
        except Exception:
            sigma = 0.2
        if sigma <= 0.0:
            sigma = 1e-3
        irf = _np.exp(-0.5 * ((time_axis - mean) / sigma) ** 2)
        try:
            max_val = float(_np.max(irf))
        except Exception:
            max_val = 0.0
        if max_val > 0.0:
            irf /= max_val
        return irf

    def _simulate_decay(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Compute simulated TCSPC decay for current parameters and IRF.

        The deterministic decay is computed from the lifetime spectrum and IRF,
        scaled to the requested peak count, and then Poisson noise is applied
        to obtain a realistic simulated trace. The IRF used for the simulation
        is stored (scaled) for plotting.
        """

        import numpy as _np

        lifetime_spectrum = self._parse_lifetime_spectrum()
        if lifetime_spectrum.size == 0:
            return None, None

        try:
            n_tac = int(self.spinBox.value())
        except Exception:
            n_tac = 4096
        try:
            dt = float(self.doubleSpinBox.value())
        except Exception:
            dt = 0.0141
        try:
            p0 = float(self.spinBox_2.value())
        except Exception:
            p0 = 0.0

        time_axis = _np.arange(int(max(1, n_tac)), dtype=float) * dt
        irf = self._build_irf(time_axis)

        # Deterministic expectation of the decay
        decay_expectation = _np.zeros_like(time_axis, dtype=float)
        try:
            from chisurf.core.fluorescence.tcspc import convolve as _tcspc_convolve

            _tcspc_convolve.convolve_lifetime_spectrum(
                output_decay=decay_expectation,
                lifetime_spectrum=lifetime_spectrum,
                instrument_response_function=irf,
                convolution_stop=time_axis.size,
                time_axis=time_axis,
            )
        except Exception:
            # Fallback: no convolution, plain multi-exponential decay
            from chisurf.core.fluorescence import general as _fl_general

            _, decay_expectation = _fl_general.calculate_fluorescence_decay(
                lifetime_spectrum=lifetime_spectrum,
                time_axis=time_axis,
                normalize=True,
            )

        # Scale deterministic decay to requested peak count
        try:
            y_max_det = float(_np.max(decay_expectation))
        except Exception:
            y_max_det = 0.0
        if y_max_det > 0.0 and p0 > 0.0:
            decay_expectation = decay_expectation * (p0 / y_max_det)

        # Apply Poisson noise to obtain simulated counts
        lam = _np.clip(decay_expectation, 0.0, None)
        try:
            decay_counts = _np.random.poisson(lam).astype(float)
        except Exception:
            # If Poisson fails for any reason, fall back to deterministic decay
            decay_counts = decay_expectation.astype(float)

        # Prepare IRF for plotting (scaled to the decay amplitude when possible)
        irf_plot = _np.asarray(irf, dtype=float)
        try:
            max_irf = float(_np.max(irf_plot))
        except Exception:
            max_irf = 0.0
        try:
            max_decay = float(_np.max(decay_counts))
        except Exception:
            max_decay = 0.0
        if max_irf > 0.0 and max_decay > 0.0:
            irf_plot = irf_plot * (max_decay / max_irf)
        self._sim_irf = irf_plot

        return time_axis, decay_counts

    def _refresh_simulation_plot(self) -> None:
        t = getattr(self, "_sim_t", None)
        y = getattr(self, "_sim_y", None)
        irf = getattr(self, "_sim_irf", None)
        if t is None or y is None:
            return
        if np.size(t) == 0 or np.size(y) == 0:
            return
        try:
            self.simulation_plot.clear()
            # Simulated decay
            self.simulation_plot.plot(t, y, pen='y')
            # IRF overlay (if available and matching length)
            if irf is not None and np.size(irf) == np.size(t):
                self.simulation_plot.plot(t, irf, pen='r')
            try:
                self.simulation_plot.setLogMode(y=True)
            except Exception:
                pass

            # Enforce a minimum visible y-value of 0.1 on the log-scaled axis
            try:
                y_main = np.asarray(y, dtype=float)
                if irf is not None and np.size(irf) == np.size(y_main):
                    y_irf = np.asarray(irf, dtype=float)
                    y_all = np.maximum(y_main, y_irf)
                else:
                    y_all = y_main
                y_pos = y_all[y_all > 0]
                if y_pos.size:
                    y_max = float(y_pos.max())
                    y_min = 0.1
                    if y_max <= y_min:
                        y_max = y_min * 10.0
                    self.simulation_plot.setYRange(np.log10(y_min), np.log10(y_max))
            except Exception:
                pass
        except Exception:
            pass

    def _on_simulate_clicked(self) -> None:
        t, y = self._simulate_decay()
        if t is None or y is None:
            return
        self._sim_t = t
        self._sim_y = y
        self._refresh_simulation_plot()

    def _on_add_clicked(self) -> None:
        """Add the last simulated decay as a dataset to imported_datasets.

        The dataset is wrapped into an ExperimentDataCurveGroup with
        experiment, setup and data_reader metadata so that it behaves like
        TCSPC datasets loaded via CSV/TTTR readers.
        """

        import numpy as _np
        from chisurf.macros import core_data as _core_data

        t = getattr(self, "_sim_t", None)
        y = getattr(self, "_sim_y", None)
        if t is None or y is None or _np.size(t) == 0 or _np.size(y) == 0:
            # If nothing simulated yet, try to simulate now
            t, y = self._simulate_decay()
            if t is None or y is None:
                return
            self._sim_t = t
            self._sim_y = y

        try:
            name = str(self.lineEdit.text()) or "TCSPC-Simulated"
        except Exception:
            name = "TCSPC-Simulated"

        # Resolve experiment, setup and reader from the global ChiSurf state
        try:
            gui = cs.cs
        except Exception:
            gui = None

        try:
            experiment = getattr(gui, 'current_experiment', None) if gui is not None else None
        except Exception:
            experiment = None
        try:
            setup = getattr(gui, 'current_setup', None) if gui is not None else None
        except Exception:
            setup = None
        try:
            experiment_reader = getattr(gui, 'current_experiment_reader', None) if gui is not None else None
        except Exception:
            experiment_reader = None

        try:
            from chisurf.core.fluorescence import tcspc as _tcspc_mod

            ey = _tcspc_mod.counting_noise(y)
        except Exception:
            ey = None

        # Create an experimental curve with proper metadata so selectors and
        # fits see it like any other TCSPC dataset.
        data_set = cs.core.data.DataCurve(
            x=_np.asarray(t, dtype=float),
            y=_np.asarray(y, dtype=float),
            ey=ey,
            name=name,
            experiment=experiment,
            data_reader=experiment_reader,
            setup=setup,
        )

        # Wrap into an ExperimentDataCurveGroup to mirror grouped imports.
        try:
            dataset_group = cs.core.data.ExperimentDataCurveGroup([data_set])
        except Exception:
            dataset_group = None

        if dataset_group is not None and gui is not None:
            try:
                cs.imported_datasets.append(dataset_group)
                cs.gui.run_on_gui_thread(gui.update)
                return
            except Exception:
                pass

        # Fallback to the generic add_dataset helper if grouping fails
        _core_data.add_dataset(experiment_reader=experiment_reader, dataset=data_set)
