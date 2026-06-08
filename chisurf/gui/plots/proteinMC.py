import json
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

import chisurf as cs
import chisurf.core.settings
from chisurf.gui.plots.plotbase import Plot

try:
    from chisurf.plugins.chimol.chimol.renderer.view import MolView as ChimolView
except Exception:  # pragma: no cover - optional GUI backend
    ChimolView = None

colors = cs.core.settings.gui['plot']['colors']
color_scheme = cs.core.settings.colors


_REPRESENTATIONS = ("cartoon", "ca_trace", "atoms")


class ProteinMCPlotControl(QtWidgets.QWidget):
    """Control panel for the ProteinMC trajectory plot."""

    def __init__(self, parent=None, plot: "Plot" = None, **kwargs):
        super().__init__(parent)
        self._plot = plot
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QtWidgets.QLabel("Trajectory curves"))

        self.show_rmsd = QtWidgets.QCheckBox("RMSD")
        self.show_drmsd = QtWidgets.QCheckBox("dRMSD")
        self.show_energy_box = QtWidgets.QCheckBox("Energy")
        self.show_fret_box = QtWidgets.QCheckBox("Labeling (FRET)")

        for cb, default in (
            (self.show_rmsd, True),
            (self.show_drmsd, True),
            (self.show_energy_box, True),
            (self.show_fret_box, True),
        ):
            cb.setChecked(default)
            layout.addWidget(cb)

        self.autoscale_btn = QtWidgets.QPushButton("Auto-scale")
        layout.addWidget(self.autoscale_btn)

        layout.addStretch(1)

        if plot is not None:
            self.show_rmsd.toggled.connect(self._toggle_rmsd)
            self.show_drmsd.toggled.connect(self._toggle_drmsd)
            self.show_energy_box.toggled.connect(self._toggle_energy)
            self.show_fret_box.toggled.connect(self._toggle_fret)
            self.autoscale_btn.clicked.connect(self._autoscale)

    def _toggle_rmsd(self, checked: bool) -> None:
        if self._plot is not None and getattr(self._plot, "rmsd_plot", None) is not None:
            self._plot.rmsd_plot.setVisible(checked)

    def _toggle_drmsd(self, checked: bool) -> None:
        if self._plot is not None and getattr(self._plot, "drmsd_plot", None) is not None:
            self._plot.drmsd_plot.setVisible(checked)

    def _toggle_energy(self, checked: bool) -> None:
        if self._plot is not None and getattr(self._plot, "energy_plot", None) is not None:
            self._plot.energy_plot.setVisible(checked)

    def _toggle_fret(self, checked: bool) -> None:
        if self._plot is not None and getattr(self._plot, "fret_plot", None) is not None:
            self._plot.fret_plot.setVisible(checked)

    def _autoscale(self) -> None:
        if self._plot is None:
            return
        for plot in (
            getattr(self._plot, attr, None)
            for attr in ("rmsd_plot", "drmsd_plot", "energy_plot", "fret_plot")
        ):
            if plot is not None:
                try:
                    plot.getViewBox().autoRange()
                except Exception:
                    pass


class ProteinMCPlot(Plot):

    name = "Trajectory-Plot"

    def __init__(self, fit, *args, **kwargs):
        super().__init__(fit=fit, *args, **kwargs)
        self.trajectory = fit.model
        self.source = fit.model

        # The base Plot.__init__ already created self.layout as a QVBoxLayout
        # on this widget. Build a 2x2 grid of pyqtgraph plots inside it so
        # each trajectory series is shown in its own subplot. (Using a
        # pyqtgraph DockArea here proved fragile in the embedded tab layout.)
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)

        p1 = pg.PlotWidget()
        p2 = pg.PlotWidget()
        p3 = pg.PlotWidget()
        p4 = pg.PlotWidget()
        for w in (p1, p2, p3, p4):
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        grid.addWidget(p1, 0, 0)
        grid.addWidget(p2, 0, 1)
        grid.addWidget(p3, 1, 0)
        grid.addWidget(p4, 1, 1)
        for row in range(2):
            grid.setRowStretch(row, 1)
        for col in range(2):
            grid.setColumnStretch(col, 1)

        self.layout.addLayout(grid, stretch=1)

        # RMSD - Curves
        self.rmsd_plot = p1.getPlotItem()
        self.drmsd_plot = p2.getPlotItem()
        self.energy_plot = p3.getPlotItem()
        self.fret_plot = p4.getPlotItem()

        self.rmsd_plot.setTitle("RMSD")
        self.drmsd_plot.setTitle("dRMSD")
        self.energy_plot.setTitle("Energy")
        self.fret_plot.setTitle("FRET")

        lw = cs.core.settings.gui['plot']['line_width']
        self.rmsd_curve = self.rmsd_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['irf'], width=lw), name='rmsd')
        self.drmsd_curve = self.drmsd_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['data'], width=lw), name='drmsd')
        self.energy_curve = self.energy_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['model'], width=lw), name='energy')
        self.fret_curve = self.fret_plot.plot(x=[0.0], y=[0.0], pen=pg.mkPen(colors['model'], width=lw), name='fret')
        self.frame_lines = []
        for plot_item in (self.rmsd_plot, self.drmsd_plot, self.energy_plot, self.fret_plot):
            line = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen((255, 255, 0, 180), width=1))
            plot_item.addItem(line, ignoreBounds=True)
            self.frame_lines.append(line)

        # Build the controller now that the plot items exist so it can wire
        # to them.
        self.plot_controller = ProteinMCPlotControl(self, plot=self)

        try:
            cs.logging.info(
                "ProteinMCPlot: initialized for fit '%s' with model '%s'",
                getattr(fit, 'name', 'unknown'),
                getattr(fit.model.__class__, 'name', fit.model.__class__.__name__)
            )
        except Exception:
            pass

    def update_all(self, *args, **kwargs):

        try:
            rmsd = np.array(self.trajectory.rmsd)
            drmsd = np.array(self.trajectory.drmsd)
            energy = np.array(self.trajectory.energy)
            energy_fret = np.array(self.trajectory.chi2r)
        except Exception as e:
            cs.logging.warning(f"ProteinMCPlot.update_all: failed to read trajectory arrays: {e}")
            return

        x = list(range(len(rmsd))) if rmsd.size else []

        self.rmsd_curve.setData(x=x, y=rmsd)
        self.drmsd_curve.setData(x=x, y=drmsd)
        self.energy_curve.setData(x=x, y=energy)
        self.fret_curve.setData(x=x, y=energy_fret)
        frame_index = int(getattr(self.trajectory, "current_frame_index", 0))
        for line in self.frame_lines:
            line.setValue(frame_index)

        try:
            cs.logging.info(
                "ProteinMCPlot.update_all: updated trajectory curves with %d points",
                len(x)
            )
        except Exception:
            pass

    def update(self, *args, **kwargs):
        """Refresh ProteinMC trajectory curves."""
        super().update(*args, **kwargs)
        self.update_all(*args, **kwargs)


class ProteinMCStructureControl(QtWidgets.QWidget):
    """Control panel for the ProteinMC structure viewer.

    Provides frame navigation (first / prev / current / next / last) and
    representation switching (cartoon / ca_trace / atoms). The control
    operates on a parent :class:`ProteinMCStructurePlot` and reflects
    changes in the live Chimol viewer.
    """

    frameChanged = QtCore.Signal(int)

    def __init__(self, parent=None, plot: "Plot" = None, **kwargs):
        super().__init__(parent)
        self._plot = plot
        self._viewer = getattr(plot, "viewer", None) if plot is not None else None
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(120)
        self._play_timer.timeout.connect(self._goto_next)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QtWidgets.QLabel("Trajectory playback"))

        # Frame navigation
        nav_row = QtWidgets.QHBoxLayout()
        nav_row.setSpacing(3)
        self.first_btn = QtWidgets.QPushButton("|<")
        self.prev_btn = QtWidgets.QPushButton("<")
        self.play_btn = QtWidgets.QPushButton("▶")
        self.pause_btn = QtWidgets.QPushButton("Ⅱ")
        self.stop_btn = QtWidgets.QPushButton("■")
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setValue(0)
        self.frame_label = QtWidgets.QLabel("/ 0")
        self.next_btn = QtWidgets.QPushButton(">")
        self.last_btn = QtWidgets.QPushButton(">|")
        for w in (
            self.first_btn,
            self.prev_btn,
            self.play_btn,
            self.pause_btn,
            self.stop_btn,
            self.frame_spin,
            self.frame_label,
            self.next_btn,
            self.last_btn,
        ):
            if isinstance(w, QtWidgets.QPushButton):
                w.setMaximumWidth(48)
            nav_row.addWidget(w)
        layout.addLayout(nav_row)

        step_row = QtWidgets.QHBoxLayout()
        step_row.setSpacing(3)
        step_row.addWidget(QtWidgets.QLabel("Step size"))
        self.step_spin = QtWidgets.QSpinBox()
        self.step_spin.setRange(1, 1000000)
        self.step_spin.setValue(1)
        self.step_spin.setToolTip("Number of frames to jump per playback step.")
        step_row.addWidget(self.step_spin)
        layout.addLayout(step_row)

        # Representation selector
        layout.addWidget(QtWidgets.QLabel("Representation"))
        self.representation_combo = QtWidgets.QComboBox()
        self.representation_combo.addItems(list(_REPRESENTATIONS))
        self.representation_combo.setCurrentText("atoms")
        layout.addWidget(self.representation_combo)

        layout.addStretch(1)

        if self._viewer is not None:
            self.first_btn.clicked.connect(self._goto_first)
            self.prev_btn.clicked.connect(self._goto_prev)
            self.play_btn.clicked.connect(self._play)
            self.pause_btn.clicked.connect(self._pause)
            self.stop_btn.clicked.connect(self._stop)
            self.next_btn.clicked.connect(self._goto_next)
            self.last_btn.clicked.connect(self._goto_last)
            self.frame_spin.valueChanged.connect(self._on_spin)
            self.representation_combo.currentTextChanged.connect(self._on_representation)

    def _object_id(self):
        """Return the Chimol object controlled by this panel."""

        return getattr(self._plot, "object_id", None)

    def _on_spin(self, value: int) -> None:
        model = getattr(self._plot, "model", None)
        if model is not None and hasattr(model, "set_current_frame"):
            model.set_current_frame(int(value))
        if self._viewer is not None:
            try:
                self._viewer.set_active_frame(int(value), object_id=self._object_id())
            except Exception:
                try:
                    self._viewer.set_current_frame(int(value))
                except Exception:
                    pass

    def _goto_first(self) -> None:
        if self._viewer is None:
            return
        self.frame_spin.setValue(0)

    def _play(self) -> None:
        """Start local trajectory playback."""

        if self._viewer is None:
            return
        if not self._play_timer.isActive():
            self._play_timer.start()

    def _pause(self) -> None:
        """Pause local trajectory playback at the current frame."""

        self._play_timer.stop()

    def _stop(self) -> None:
        """Stop playback and return to the first frame."""

        self._play_timer.stop()
        self._goto_first()

    def _goto_prev(self) -> None:
        model = getattr(self._plot, "model", None)
        if model is not None:
            current = int(getattr(model, "current_frame_index", 0))
        else:
            if self._viewer is None:
                return
            try:
                current = int(self._viewer.get_active_frame_index(self._object_id()))
            except Exception:
                try:
                    current = int(self._viewer.get_current_frame())
                except Exception:
                    current = 0
        step = max(1, int(self.step_spin.value()))
        self.frame_spin.setValue(max(0, current - step))

    def _goto_next(self) -> None:
        model = getattr(self._plot, "model", None)
        if model is not None:
            current = int(getattr(model, "current_frame_index", 0))
            total = int(getattr(model, "frame_count", 0))
        else:
            if self._viewer is None:
                return
            try:
                current = int(self._viewer.get_active_frame_index(self._object_id()))
            except Exception:
                try:
                    current = int(self._viewer.get_current_frame())
                except Exception:
                    current = 0
            try:
                total = int(self._viewer.get_frame_count(self._object_id()))
            except Exception:
                try:
                    total = int(self._viewer.get_total_frames())
                except Exception:
                    total = 0
        step = max(1, int(self.step_spin.value()))
        if total > 0 and current + step > total - 1:
            self.frame_spin.setValue(0)
        else:
            self.frame_spin.setValue(current + step)

    def _goto_last(self) -> None:
        model = getattr(self._plot, "model", None)
        if model is not None:
            total = int(getattr(model, "frame_count", 0))
        else:
            if self._viewer is None:
                return
            try:
                total = int(self._viewer.get_frame_count(self._object_id()))
            except Exception:
                try:
                    total = int(self._viewer.get_total_frames())
                except Exception:
                    total = 0
        self.frame_spin.setValue(max(0, total - 1))

    def _on_representation(self, mode: str) -> None:
        if self._viewer is None:
            return
        try:
            self._viewer.set_representation(mode, object_id=self._object_id())
        except Exception:
            try:
                self._viewer.set_representation(mode)
            except Exception:
                pass

    def refresh_from_viewer(self) -> None:
        """Sync the spin box and label to the current viewer state."""
        if self._viewer is None:
            return
        try:
            model = getattr(self._plot, "model", None)
            total = int(getattr(model, "frame_count", 0)) if model is not None else int(self._viewer.get_frame_count(self._object_id()))
            current = int(getattr(model, "current_frame_index", 0)) if model is not None else int(self._viewer.get_active_frame_index(self._object_id()))
        except Exception:
            try:
                total = int(self._viewer.get_total_frames())
                current = int(self._viewer.get_current_frame())
            except Exception:
                return
        self.frame_spin.blockSignals(True)
        try:
            self.frame_spin.setRange(0, max(0, total - 1))
            self.frame_spin.setValue(max(0, current))
        finally:
            self.frame_spin.blockSignals(False)
        self.frame_label.setText(f"/ {max(0, total - 1)}")


class ProteinMCStructurePlot(Plot):
    """Chimol structure plot for live ProteinMC trajectories."""

    name = "Structure"

    def __init__(self, fit, *args, **kwargs):
        """Create a Chimol-backed ProteinMC structure plot."""
        super().__init__(fit=fit, *args, **kwargs)
        self.model = fit.model
        self.object_id = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # The base Plot.__init__ already created self.layout as a QVBoxLayout
        # on this widget. Reuse it so the viewer fills the tab.
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        if ChimolView is None:
            self.viewer = None
            placeholder = QtWidgets.QLabel("Chimol viewer unavailable", self)
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            self.layout.addWidget(placeholder, stretch=1)
        else:
            self.viewer = ChimolView(parent=self, representation_mode="atoms")
            self.viewer.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            self.viewer.setMinimumSize(320, 240)
            self.layout.addWidget(self.viewer, stretch=1)
        # Build the controller after the viewer exists so it can drive it.
        self.plot_controller = ProteinMCStructureControl(self, plot=self)
        self.update_all()

    def update_all(self, *args, **kwargs):
        """Refresh the structure display from the ProteinMC model widget."""
        if self.viewer is None:
            return
        structure = getattr(self.model, "proteinmc_structure", None)
        frames = getattr(self.model, "trajectory_frames", None)
        if structure is None and not frames:
            return
        if self.object_id is None:
            if structure is not None:
                self.object_id = self.viewer.add_structure(structure, name="ProteinMC")
            else:
                self.object_id = self.viewer.add_coordinates(np.asarray(frames[0], dtype=float), name="ProteinMC")
            try:
                self.viewer.set_representation("atoms", object_id=self.object_id)
            except Exception:
                pass
        if frames:
            arr = np.asarray(frames, dtype=float)
            active_frame = min(max(0, int(getattr(self.model, "current_frame_index", len(arr) - 1))), len(arr) - 1)
            try:
                self.viewer.set_frames(
                    arr,
                    object_id=self.object_id,
                    active_frame=active_frame,
                )
            except TypeError:
                self.viewer.set_frames(arr, object_id=self.object_id)
                set_active_frame = getattr(self.viewer, "set_active_frame", None)
                if set_active_frame is not None:
                    set_active_frame(active_frame, object_id=self.object_id)
            try:
                self.viewer.set_representation("atoms", object_id=self.object_id)
            except Exception:
                pass
        if getattr(self, "plot_controller", None) is not None:
            self.plot_controller.refresh_from_viewer()

    def update(self, *args, **kwargs):
        """Refresh the Chimol structure plot."""
        super().update(*args, **kwargs)
        self.update_all(*args, **kwargs)


class ProteinMCDistanceNetworkPlot(Plot):
    """Circular FPS distance-network plot for the selected ProteinMC frame."""

    name = "Distance Network"

    def __init__(self, fit, *args, **kwargs):
        """Create a circular distance-agreement plot."""
        super().__init__(fit=fit, *args, **kwargs)
        self.model = fit.model
        self._network_cache_key = None
        self._network_edges = []
        self._network_edge_items = []
        self._network_node_positions = {}
        self._network_static_items = []
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setBackground((20, 20, 20))
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("bottom")
        self.layout.addWidget(self.plot_widget, stretch=1)
        self.plot_controller = ProteinMCDistanceNetworkControl(self, plot=self)
        self.update_all()

    def update_all(self, *args, **kwargs):
        """Redraw network agreement for the shared current frame."""
        controller = getattr(self, "plot_controller", None)
        if controller is not None and hasattr(controller, "refresh_from_model"):
            controller.refresh_from_model()
        structure = getattr(self.model, "proteinmc_structure", None)
        frames = getattr(self.model, "trajectory_frames", []) or []
        if not frames and structure is not None:
            xyz = getattr(structure, "xyz", None)
            if xyz is not None:
                frames = [xyz]
        labeling_file = ""
        try:
            labeling_file = self.model.labeling_edit.text().strip()
        except Exception:
            pass
        if not labeling_file:
            try:
                for pot in getattr(self.model, "_potential_settings", lambda: [])():
                    if pot.get("name") == "dye":
                        labeling_file = pot.get("settings", {}).get("labeling_file", "")
                        if labeling_file:
                            break
            except Exception:
                pass
        if structure is None or not frames or not labeling_file:
            self.plot_widget.clear()
            self._network_cache_key = None
            self._draw_message("No FPS network data")
            return
        cache_key = (id(structure), str(labeling_file))
        if cache_key != self._network_cache_key:
            self._build_network_cache(structure, labeling_file, cache_key)
        if not self._network_edges or not self._network_edge_items:
            return
        frame_idx = max(0, min(int(getattr(self.model, "current_frame_index", 0)), len(frames) - 1))
        xyz = np.asarray(frames[frame_idx], dtype=float)
        for edge, item in zip(self._network_edges, self._network_edge_items):
            try:
                model_distance = float(np.linalg.norm(xyz[edge["i1"]] - xyz[edge["i2"]]))
            except Exception:
                continue
            target = float(edge["target"])
            error = max(float(edge["error_neg"] if model_distance < target else edge["error_pos"]), 1e-12)
            wres = (model_distance - target) / error
            item.setPen(pg.mkPen(_agreement_color(wres), width=1.0 + min(abs(wres), 3.0) * 0.8))

    def _build_network_cache(self, structure, labeling_file: str, cache_key) -> None:
        """Build static network graphics once for responsive playback."""
        self.plot_widget.clear()
        self._network_cache_key = cache_key
        self._network_edges = []
        self._network_edge_items = []
        self._network_node_positions = {}
        self._network_static_items = []
        try:
            payload = _load_labeling_payload(labeling_file)
            nodes, edges = _network_from_labeling(structure, payload)
        except Exception as exc:
            self._draw_message(f"Cannot load network: {exc}")
            return
        if not nodes or not edges:
            self._draw_message("No distances in FPS file")
            return
        self._network_node_positions = _circle_positions(nodes)
        for edge in edges:
            p1 = self._network_node_positions.get(edge["p1"])
            p2 = self._network_node_positions.get(edge["p2"])
            if p1 is None or p2 is None:
                continue
            item = self.plot_widget.plot([p1[0], p2[0]], [p1[1], p2[1]], pen=pg.mkPen((80, 80, 80, 120), width=1.0))
            self._network_edges.append(edge)
            self._network_edge_items.append(item)
        scatter = pg.ScatterPlotItem(
            x=[self._network_node_positions[name][0] for name in nodes],
            y=[self._network_node_positions[name][1] for name in nodes],
            size=8,
            brush=pg.mkBrush(230, 230, 230),
            pen=pg.mkPen(30, 30, 30),
        )
        self.plot_widget.addItem(scatter)
        self._network_static_items.append(scatter)
        for name in nodes:
            pos = self._network_node_positions[name]
            label = pg.TextItem(str(name), color=(230, 230, 230), anchor=(0.5, 0.5))
            label.setPos(float(pos[0] * 1.12), float(pos[1] * 1.12))
            self.plot_widget.addItem(label)
            self._network_static_items.append(label)
        self.plot_widget.setRange(xRange=(-1.25, 1.25), yRange=(-1.25, 1.25), padding=0.02)

    def _draw_message(self, message: str) -> None:
        label = pg.TextItem(str(message), color=(230, 230, 230), anchor=(0.5, 0.5))
        label.setPos(0.0, 0.0)
        self.plot_widget.addItem(label)
        self.plot_widget.setRange(xRange=(-1, 1), yRange=(-1, 1), padding=0.02)

    def update(self, *args, **kwargs):
        """Refresh the circular network plot."""
        super().update(*args, **kwargs)
        self.update_all(*args, **kwargs)


class ProteinMCDistanceNetworkControl(QtWidgets.QWidget):
    """Plot-controller panel for the circular ProteinMC distance network."""

    def __init__(self, parent=None, plot: ProteinMCDistanceNetworkPlot = None, **kwargs):
        """Create frame playback controls for the distance-network plot."""
        super().__init__(parent)
        self._plot = plot
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(120)
        self._play_timer.timeout.connect(self._next_frame)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QtWidgets.QLabel("Distance network"))

        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(3)
        self.start_btn = QtWidgets.QPushButton("|<", self)
        self.prev_btn = QtWidgets.QPushButton("<", self)
        self.play_btn = QtWidgets.QPushButton("▶", self)
        self.stop_btn = QtWidgets.QPushButton("■", self)
        self.frame_spin = QtWidgets.QSpinBox(self)
        self.frame_spin.setRange(0, 0)
        self.frame_label = QtWidgets.QLabel("/ 0", self)
        self.next_btn = QtWidgets.QPushButton(">", self)
        self.last_btn = QtWidgets.QPushButton(">|", self)
        for widget in (
            self.start_btn,
            self.prev_btn,
            self.play_btn,
            self.stop_btn,
            self.frame_spin,
            self.frame_label,
            self.next_btn,
            self.last_btn,
        ):
            if isinstance(widget, QtWidgets.QPushButton):
                widget.setMaximumWidth(48)
            nav.addWidget(widget)
        layout.addLayout(nav)

        step_row = QtWidgets.QHBoxLayout()
        step_row.addWidget(QtWidgets.QLabel("Step size", self))
        self.step_spin = QtWidgets.QSpinBox(self)
        self.step_spin.setRange(1, 1000000)
        self.step_spin.setValue(1)
        self.step_spin.setToolTip("Number of frames to jump per Play/Next step.")
        step_row.addWidget(self.step_spin)
        layout.addLayout(step_row)

        layout.addStretch(1)

        self.start_btn.clicked.connect(self._start)
        self.play_btn.clicked.connect(self._play)
        self.stop_btn.clicked.connect(self._stop)
        self.prev_btn.clicked.connect(self._prev_frame)
        self.next_btn.clicked.connect(self._next_frame)
        self.last_btn.clicked.connect(lambda: self.frame_spin.setValue(self.frame_spin.maximum()))
        self.frame_spin.valueChanged.connect(self._set_frame)
        self.refresh_from_model()

    def _model(self):
        """Return the ProteinMC model backing this controller."""
        return getattr(self._plot, "model", None)

    def _set_frame(self, value: int) -> None:
        """Set the shared ProteinMC frame from this controller."""
        model = self._model()
        if model is not None and hasattr(model, "set_current_frame"):
            model.set_current_frame(int(value))

    def _start(self) -> None:
        """Jump to the first frame."""
        self._play_timer.stop()
        self.frame_spin.setValue(0)

    def _play(self) -> None:
        """Play through frames until stopped."""
        if not self._play_timer.isActive():
            self._play_timer.start()

    def _stop(self) -> None:
        """Stop playback and return to the first frame."""
        self._play_timer.stop()
        self.frame_spin.setValue(0)

    def _prev_frame(self) -> None:
        """Move one frame backward."""
        step = max(1, int(self.step_spin.value()))
        self.frame_spin.setValue(max(0, self.frame_spin.value() - step))

    def _next_frame(self) -> None:
        """Move one frame forward, wrapping at the end."""
        current = self.frame_spin.value()
        maximum = self.frame_spin.maximum()
        step = max(1, int(self.step_spin.value()))
        if maximum <= 0:
            self.frame_spin.setValue(0)
            return
        self.frame_spin.setValue((current + step) % (maximum + 1))

    def refresh_from_model(self) -> None:
        """Sync the frame selector to the shared ProteinMC frame state."""
        model = self._model()
        total = int(getattr(model, "frame_count", 0)) if model is not None else 0
        current = int(getattr(model, "current_frame_index", 0)) if model is not None else 0
        maximum = max(0, total - 1)
        self.frame_spin.blockSignals(True)
        try:
            self.frame_spin.setRange(0, maximum)
            self.frame_spin.setValue(max(0, min(current, maximum)))
        finally:
            self.frame_spin.blockSignals(False)
        self.frame_label.setText(f"/ {maximum}")


def _load_labeling_payload(filename: str) -> dict:
    """Load an FPS JSON labeling file."""
    with open(filename, "r") as fp:
        return json.load(fp)


def _network_from_labeling(structure, payload: dict) -> tuple[list[str], list[dict]]:
    """Return sorted network nodes and resolved edges from FPS JSON."""
    positions = payload.get("Positions", {}) or {}
    distances = payload.get("Distances", {}) or {}
    used = []
    edges = []
    for distance in distances.values():
        p1_name = str(distance.get("position1_name", ""))
        p2_name = str(distance.get("position2_name", ""))
        if not p1_name or not p2_name or p1_name not in positions or p2_name not in positions:
            continue
        try:
            i1 = _resolve_position_index(structure, positions[p1_name])
            i2 = _resolve_position_index(structure, positions[p2_name])
        except Exception:
            continue
        used.extend([p1_name, p2_name])
        edges.append(
            {
                "p1": p1_name,
                "p2": p2_name,
                "i1": i1,
                "i2": i2,
                "target": float(distance.get("distance", 0.0)),
                "error_neg": float(distance.get("error_neg", 1.0)),
                "error_pos": float(distance.get("error_pos", 1.0)),
            }
        )
    nodes = sorted(set(used), key=_node_sort_key)
    return nodes, edges


def _resolve_position_index(structure, position: dict) -> int:
    """Resolve one FPS position to a structure atom index."""
    atoms = structure.atoms
    if "chain_identifier" in position and "residue_seq_number" in position:
        chain = str(position["chain_identifier"])
        residue = int(position["residue_seq_number"])
        atom_name = str(position.get("atom_name", "CA"))
        mask = np.ones(len(atoms), dtype=bool)
        if "chain" in atoms.dtype.names:
            mask &= np.array([_as_text(v) == chain for v in atoms["chain"]])
        if "res_id" in atoms.dtype.names:
            mask &= atoms["res_id"] == residue
        if "atom_name" in atoms.dtype.names:
            atom_mask = np.array([_as_text(v) == atom_name for v in atoms["atom_name"]])
            ca_mask = np.array([_as_text(v) == "CA" for v in atoms["atom_name"]])
            selected = np.where(mask & atom_mask)[0]
            if selected.size == 0:
                selected = np.where(mask & ca_mask)[0]
        else:
            selected = np.where(mask)[0]
        if selected.size:
            return int(selected[0])
    if "attachment_atom_index" in position:
        index = int(position["attachment_atom_index"])
        if 0 <= index < len(atoms):
            return index
    raise ValueError(f"Cannot resolve labeling position: {position!r}")


def _circle_positions(nodes: list[str]) -> dict[str, np.ndarray]:
    """Return unit-circle coordinates for network nodes."""
    n = max(len(nodes), 1)
    return {
        name: np.array([np.cos(2.0 * np.pi * i / n), np.sin(2.0 * np.pi * i / n)], dtype=float)
        for i, name in enumerate(nodes)
    }


def _agreement_color(wres: float) -> tuple[int, int, int, int]:
    """Map signed weighted residual to blue-agree through magenta-disagree."""
    t = float(np.clip((abs(float(wres)) - 1.0) / 2.0, 0.0, 1.0))
    blue = np.array([45, 130, 255, 180], dtype=float)
    magenta = np.array([255, 0, 255, 230], dtype=float)
    color = blue * (1.0 - t) + magenta * t
    return tuple(int(v) for v in color)


def _node_sort_key(name: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return (int(digits) if digits else 10**9, str(name))


def _as_text(value) -> str:
    """Convert numpy string scalar values to plain text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()



# class ProteinMCPlot_Old(Plot):
#
#     name = "Trajectory-Plot"
#
#     def __init__(self, fit):
#         Plot.__init__(self, fit)
#         self.layout = QtGui.QVBoxLayout(self)
#
#         self.trajectory = fit.models
#         self.source = fit.models
#
#         # RMSD - Curves
#         top_left = QtGui.QFrame(self)
#         top_left.setFrameShape(QtGui.QFrame.StyledPanel)
#         l = QtGui.QVBoxLayout(top_left)
#
#         top_right = QtGui.QFrame(self)
#         top_right.setFrameShape(QtGui.QFrame.StyledPanel)
#         r = QtGui.QVBoxLayout(top_right)
#
#         splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
#         splitter.addWidget(top_left)
#         splitter.addWidget(top_right)
#
#         self.layout.addWidget(splitter)
#
#         win = CurveDialog()
#         self.rmsd_plot = win.get_plot()
#         self.rmsd_plot.set_titles(ylabel='RMSD')
#         self.rmsd_curve = make.curve([],  [], color="m", linewidth=1)
#         self.rmsd_plot.add_item(self.rmsd_curve)
#         l.addWidget(self.rmsd_plot)
#
#         win = CurveDialog()
#         self.drmsd_plot = win.get_plot()
#         self.drmsd_plot.set_titles(ylabel='dRMSD')
#         self.drmsd_curve = make.curve([],  [], color="r", linewidth=1)
#         self.drmsd_plot.add_item(self.drmsd_curve)
#         r.addWidget(self.drmsd_plot)
#
#         # Energy - Curves
#         top_left = QtGui.QFrame(self)
#         top_left.setFrameShape(QtGui.QFrame.StyledPanel)
#         l = QtGui.QVBoxLayout(top_left)
#
#         top_right = QtGui.QFrame(self)
#         top_right.setFrameShape(QtGui.QFrame.StyledPanel)
#         r = QtGui.QVBoxLayout(top_right)
#
#         splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
#         splitter.addWidget(top_left)
#         splitter.addWidget(top_right)
#
#         self.layout.addWidget(splitter)
#
#         win = CurveDialog()
#         self.fret_plot = win.get_plot()
#         self.fret_plot.set_titles(ylabel='FRET-Energy')
#         self.fret_curve = make.curve([],  [], color="m", linewidth=1)
#         self.fret_plot.add_item(self.fret_curve)
#         l.addWidget(self.fret_plot)
#
#         win = CurveDialog()
#         self.energy_plot = win.get_plot()
#         self.energy_plot.set_titles(ylabel='System-Energy')
#         self.energy_curve = make.curve([],  [], color="g", linewidth=1)
#         self.energy_plot.add_item(self.energy_curve)
#         r.addWidget(self.energy_plot)
#
#     def update_all(self, *args, **kwargs):
#
#         rmsd = np.array(self.trajectory.rmsd)
#         drmsd = np.array(self.trajectory.drmsd)
#         energy = np.array(self.trajectory.energy)
#         energy_fret = np.array(self.trajectory.chi2r)
#         x = list(range(len(rmsd)))
#
#         self.rmsd_curve.set_data(x, rmsd)
#         self.drmsd_curve.set_data(x, drmsd)
#         self.energy_curve.set_data(x, energy)
#         self.fret_curve.set_data(x, energy_fret)
#
#         self.energy_plot.do_autoscale()
#         self.fret_plot.do_autoscale()
#         self.rmsd_plot.do_autoscale()
#         self.drmsd_plot.do_autoscale()
