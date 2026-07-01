from __future__ import annotations
import chisurf as cs

import json
import os
import pathlib
import textwrap
import typing

import numpy as np
from qtpy import QtWidgets, uic, QtCore, QtGui
import matplotlib.colors as mcolors

import chisurf.core.data
import chisurf.core.fitting
import chisurf.core.decorators
import chisurf.gui.decorators
import chisurf.core.settings

import chisurf.gui.widgets
import chisurf.gui.widgets.experiments.widgets
from chisurf.gui.widgets.mdi_custom_titlebar import CustomMdiSubWindow
from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client
from chisurf.core.math.optimization.leastsqbound import OptimizationCancelled
from chisurf.gui.widgets.dock_area import DockArea


class FitSubWindow(CustomMdiSubWindow):

    def update(self, *args):
        super().update(self, *args)
        self.plot_tab_widget.update(*args)
        self.refresh_current_plot()

    def refresh_current_plot(self) -> None:
        """Recompute and redraw the currently visible plot from the model.

        The DockArea's own ``update()`` only schedules a Qt repaint; it does
        **not** re-pull the model curve. The actual redraw is performed by the
        individual :class:`Plot` widget via its ``update_all``/``update`` hook
        (the same path :meth:`on_change_plot` uses). Call it here so that a
        parameter-value edit or a finished fit (delivered as ``fit.updated`` /
        ``fit.ran`` events) is reflected in the trace without switching tabs.
        """
        try:
            idx = self.plot_tab_widget.currentIndex()
            plot = self.ensure_plot_created(idx)
            if plot is None:
                return
            update_all = getattr(plot, "update_all", None)
            if callable(update_all):
                update_all()
            elif hasattr(plot, "update"):
                plot.update()
        except Exception:
            pass

    def __init__(
            self,
            fit: cs.core.fitting.fit.FitGroup,
            control_layout: QtWidgets.QLayout,
            fit_widget: object = None,
            *args,
            **kwargs
    ):
        # Initialize with fit name as title
        title = getattr(fit, 'name', 'Fit Window')
        super().__init__(title=title, *args,  **kwargs)

        self.fit = fit
        self.fit_widget = fit_widget

        # Use the content_layout from CustomMdiSubWindow instead of creating new widget
        # Set the focus policy of the subwindow
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.content_widget.setFocusPolicy(QtCore.Qt.ClickFocus)

        # Use the existing content_layout from CustomMdiSubWindow
        layout = self.content_layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Stacked widget for front and back faces
        self.stack = QtWidgets.QStackedWidget(self)
        layout.addWidget(self.stack)

        # Front face (Plots and Controls)
        self.front_widget = QtWidgets.QWidget()
        self.front_layout = QtWidgets.QVBoxLayout()
        self.front_layout.setContentsMargins(0, 0, 0, 0)
        self.front_layout.setSpacing(0)
        self.front_widget.setLayout(self.front_layout)

        # Create DockArea
        self.plot_tab_widget = DockArea(self)
        self.plot_tab_widget.setNewTabButtonVisible(False)
        self.flip_to_code_btn = QtWidgets.QToolButton()
        self.flip_to_code_btn.setText("Code")
        self.flip_to_code_btn.setFixedSize(50, 20)
        self.flip_to_code_btn.clicked.connect(self.toggle_code_view)
        self.flip_to_code_btn.setAutoRaise(True)
        self.flip_to_code_btn.setCheckable(True)
        self.flip_to_code_btn.setStyleSheet('background: transparent; color: palette(text); font-weight: bold;')
        
        # Add Code button directly to CustomTitleBar to save maximum vertical space
        if hasattr(self, 'title_bar') and self.title_bar.layout():
            layout = self.title_bar.layout()
            # Insert before minimize, maximize, close
            idx = layout.indexOf(self.title_bar.minimize_btn)
            layout.insertWidget(idx, self.flip_to_code_btn)
        
        self.front_layout.addWidget(self.plot_tab_widget)
        self.stack.addWidget(self.front_widget)

        # Back face (Code Editor)
        self.back_widget = QtWidgets.QWidget()
        self.back_layout = QtWidgets.QVBoxLayout()
        self.back_layout.setContentsMargins(0, 0, 0, 0)
        self.back_layout.setSpacing(0)
        self.back_widget.setLayout(self.back_layout)

        self.back_toolbar = QtWidgets.QHBoxLayout()
        self.back_toolbar.setContentsMargins(5, 5, 5, 5)

        self.nav_back_btn = QtWidgets.QToolButton()
        self.nav_back_btn.setText("←")
        self.nav_back_btn.setToolTip("Navigate back to previous cursor position")
        self.nav_back_btn.clicked.connect(self._code_nav_back)

        self.nav_forward_btn = QtWidgets.QToolButton()
        self.nav_forward_btn.setText("→")
        self.nav_forward_btn.setToolTip("Navigate forward to next cursor position")
        self.nav_forward_btn.clicked.connect(self._code_nav_forward)

        self.file_combo = QtWidgets.QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_code_file_selected)

        self.func_combo = QtWidgets.QComboBox()
        self.func_combo.currentIndexChanged.connect(self.on_code_func_selected)

        self.save_code_btn = QtWidgets.QToolButton()
        self.save_code_btn.setText("Save/Apply")
        self.save_code_btn.setToolTip("Save the current editor content to the model")
        self.save_code_btn.clicked.connect(self.save_model_code)

        self.back_toolbar.addWidget(self.nav_back_btn)
        self.back_toolbar.addWidget(self.nav_forward_btn)
        self.back_toolbar.addWidget(QtWidgets.QLabel("File:"))
        self.back_toolbar.addWidget(self.file_combo, 1)
        self.back_toolbar.addWidget(QtWidgets.QLabel("  Jump to:"))
        self.back_toolbar.addWidget(self.func_combo, 1)
        self.back_toolbar.addStretch()
        self.back_toolbar.addWidget(self.save_code_btn)
        
        self.back_layout.addLayout(self.back_toolbar)

        from chisurf.plugins.core.code_editor import CodeEditor
        self.code_editor = CodeEditor(self, language="python", can_load=False)
        # Wire the fit window's own toolbar nav buttons to the current editor
        self.code_editor._on_editor_created = self._on_code_editor_created
        self.code_editor.symbolsChanged.connect(self._sync_code_symbol_combo)
        self.back_layout.addWidget(self.code_editor)

        self.agent_btn = QtWidgets.QToolButton()
        self.agent_btn.setText("🤖")
        self.agent_btn.setToolTip("Toggle AI agent panel")
        self.agent_btn.clicked.connect(self.code_editor._toggle_agent_panel)
        self.back_toolbar.insertWidget(0, self.agent_btn)
        self.stack.addWidget(self.back_widget)

        rect = self.plot_tab_widget.geometry()
        self.setGeometry(rect)

        self.current_plot_controller = QtWidgets.QWidget(self)
        self.current_plot_controller.hide()

        # Lazy plot instantiation: create lightweight tab containers now, build plots on demand
        self._control_layout = control_layout
        from chisurf.gui.widgets.models.model_editor import model_plot_specs
        self._plot_specs = model_plot_specs(fit.model)
        self._plot_containers = []
        self._plots_all = [None] * len(self._plot_specs)      # positional storage
        self._created_plots = []                               # actual created plots (shared)
        # Create empty containers per tab
        for idx, (plot_class, kwargs) in enumerate(self._plot_specs):
            container = QtWidgets.QWidget()
            container.setLayout(QtWidgets.QVBoxLayout())
            container.layout().setContentsMargins(0, 0, 0, 0)
            container.layout().setSpacing(0)
            tab_name = getattr(plot_class, 'name', None)
            if not isinstance(tab_name, str):
                tab_name = getattr(plot_class, '__name__', str(plot_class))
            container.setProperty("fit_plot_index", idx)
            container.setProperty("fit_plot_name", tab_name)
            self._plot_containers.append(container)
            self.plot_tab_widget.addTab(container, tab_name)
        # Share created plot list with FitGroup and its member Fits
        fit.plots = self._created_plots
        for f in fit:
            f.plots = self._created_plots

        # Instantiate the initially visible plot after the event loop returns
        # to avoid re-entrancy issues during fit creation; this may introduce
        # a tiny visual delay but is safer.
        def _ensure_initial_plot():
            idx = self.plot_tab_widget.currentIndex()
            self.ensure_plot_created(idx)
            self._restore_pending_project_plot_state()
            self.on_change_plot()
        QtCore.QTimer.singleShot(0, _ensure_initial_plot)

        self.plot_tab_widget.currentChanged.connect(self.on_change_plot)

        # Use RubberBandResize / RubberBandMove
        self.setOption(
            cs.gui.QtWidgets.QMdiSubWindow.RubberBandResize,
            cs.core.settings.gui['RubberBandResize']
        )
        self.setOption(
            cs.gui.QtWidgets.QMdiSubWindow.RubberBandMove,
            cs.core.settings.gui['RubberBandMove']
        )

        # Set windows icon
        try:
            icon = fit.model.icon
        except AttributeError:
            icon = cs.gui.QtGui.QIcon(":/icons/icons/list-add.png")
        self.setWindowIcon(icon)

        # Set global style sheet
        # window_style = cs.core.settings.gui['fit_window_style']
        # self.setStyleSheet(cs.core.settings.style_sheet)

        self.setAttribute(cs.gui.QtCore.Qt.WA_DeleteOnClose, True)

        # Resize window
        xs, ys = cs.core.settings.gui['fit_windows_size']
        self.resize(xs, ys)

        self.plot_tab_widget.layoutChanged.connect(self.save_fit_dock_layout_state)
        self.restore_fit_dock_layout_state()

    def get_project_plot_state(self) -> dict:
        """Return project-serializable plot layout and controller state.

        Returns
        -------
        dict
            Fit-window plot state suitable for embedding in ``project.json``.
        """
        plots: list[dict] = []
        for idx, plot in enumerate(getattr(self, "_plots_all", []) or []):
            if plot is None:
                continue
            controller = getattr(plot, "plot_controller", None)
            controller_state = {}
            get_controller_state = getattr(controller, "get_state", None)
            if callable(get_controller_state):
                try:
                    controller_state = get_controller_state()
                except Exception:
                    controller_state = {}
            plot_state = {}
            get_plot_state = getattr(plot, "get_state", None)
            if callable(get_plot_state):
                try:
                    plot_state = get_plot_state()
                except Exception:
                    plot_state = {}
            rec: dict[str, object] = {
                "index": idx,
                "name": self._plot_containers[idx].property("fit_plot_name"),
            }
            if controller_state:
                rec["controller"] = controller_state
            if plot_state:
                rec["plot"] = plot_state
            plots.append(rec)

        state: dict[str, object] = {
            "version": 1,
            "current_plot_index": int(self.plot_tab_widget.currentIndex()),
            "plots": plots,
        }
        try:
            state["dock_layout"] = self.get_fit_dock_layout_state()
        except Exception:
            pass
        try:
            geom = self.geometry()
            state["geometry"] = [geom.x(), geom.y(), geom.width(), geom.height()]
        except Exception:
            pass
        try:
            state["stack_index"] = int(self.stack.currentIndex())
        except Exception:
            pass
        return state

    def _restore_pending_project_plot_state(self) -> None:
        """Apply plot state attached to the fit during project loading."""
        state = getattr(self.fit, "_project_plot_state", None)
        if not isinstance(state, dict) or not state:
            return
        try:
            delattr(self.fit, "_project_plot_state")
        except Exception:
            pass
        self.set_project_plot_state(state)

    def set_project_plot_state(self, state: dict) -> bool:
        """Restore project-serialized plot layout and controller state.

        Parameters
        ----------
        state : dict
            State generated by :meth:`get_project_plot_state`.

        Returns
        -------
        bool
            True when at least one state fragment was applied.
        """
        if not isinstance(state, dict):
            return False
        applied = False
        plot_records = state.get("plots")
        if isinstance(plot_records, list):
            for rec in plot_records:
                if not isinstance(rec, dict):
                    continue
                try:
                    idx = int(rec.get("index"))
                except Exception:
                    continue
                plot = self.ensure_plot_created(idx)
                if plot is None:
                    continue
                plot_state = rec.get("plot")
                set_plot_state = getattr(plot, "set_state", None)
                if isinstance(plot_state, dict) and callable(set_plot_state):
                    try:
                        set_plot_state(plot_state)
                        applied = True
                    except Exception:
                        pass
                controller_state = rec.get("controller")
                controller = getattr(plot, "plot_controller", None)
                set_controller_state = getattr(controller, "set_state", None)
                if isinstance(controller_state, dict) and callable(set_controller_state):
                    try:
                        set_controller_state(controller_state)
                        applied = True
                    except Exception:
                        pass

        dock_state = state.get("dock_layout")
        if isinstance(dock_state, dict):
            try:
                restored = self.plot_tab_widget.set_layout_state(
                    dock_state,
                    key_func=self._plot_widget_key,
                    emit_change=False,
                )
                applied = bool(restored) or applied
            except Exception:
                pass

        geom = state.get("geometry")
        if isinstance(geom, list) and len(geom) == 4:
            try:
                self.setGeometry(*(int(v) for v in geom))
                applied = True
            except Exception:
                pass

        stack_index = state.get("stack_index")
        if isinstance(stack_index, int):
            try:
                if 0 <= stack_index < self.stack.count():
                    self.stack.setCurrentIndex(stack_index)
                    applied = True
            except Exception:
                pass

        current_index = state.get("current_plot_index")
        if isinstance(current_index, int):
            try:
                if 0 <= current_index < self.plot_tab_widget.count():
                    self.plot_tab_widget.setCurrentIndex(current_index)
                    applied = True
            except Exception:
                pass
        return applied

    def _fit_model_class_key(self) -> str:
        """Return the persistent layout key for this fit's model class.

        Returns
        -------
        str
            Fully qualified model class name.
        """
        model = getattr(self.fit, "model", None)
        model_cls = model.__class__ if model is not None else self.fit.__class__
        return f"{model_cls.__module__}.{model_cls.__name__}"

    def _plot_widget_key(self, widget: QtWidgets.QWidget) -> str:
        """Return the persistent layout key for a plot widget.

        Parameters
        ----------
        widget : QWidget
            Plot page widget.

        Returns
        -------
        str
            Stable plot key.
        """
        idx = widget.property("fit_plot_index")
        name = widget.property("fit_plot_name")
        try:
            return f"{int(idx)}:{name or ''}"
        except Exception:
            return str(name or "")

    def _fit_dock_layout_settings(self) -> QtCore.QSettings:
        """Return QSettings for fit-window dock layouts in the user folder.

        Returns
        -------
        QSettings
            Settings object backed by ``~/.chisurf/fit_window_dock_layouts.ini``.
        """
        settings_path = cs.core.settings.get_path("settings") / "fit_window_dock_layouts.ini"
        return QtCore.QSettings(str(settings_path), QtCore.QSettings.IniFormat)

    def get_fit_dock_layout_state(self) -> dict:
        """Return the current dock layout state for this fit's model class.

        Returns
        -------
        dict
            Serialized dock layout.
        """
        return self.plot_tab_widget.get_layout_state(key_func=self._plot_widget_key)

    def save_fit_dock_layout_state(self) -> None:
        """Persist the current dock layout for this fit's model class."""
        try:
            if self.plot_tab_widget.count() <= 0:
                return
            state = self.get_fit_dock_layout_state()
            settings = self._fit_dock_layout_settings()
            settings.setValue(self._fit_model_class_key(), json.dumps(state, sort_keys=True))
            settings.sync()
        except Exception as exc:
            try:
                cs.logging.warning(f"Failed to save fit dock layout: {exc}")
            except Exception:
                pass

    def restore_fit_dock_layout_state(self) -> None:
        """Restore the saved dock layout for this fit's model class.

        A saved layout that predates a newly-added plot tab would otherwise drop
        that tab (``set_layout_state`` rebuilds the tab set from the saved keys).
        So the saved layout is ignored when it is missing any plot that exists
        now — the default view-spec tab order is used instead, and the next
        rearrange re-saves the full set.
        """
        try:
            settings = self._fit_dock_layout_settings()
            value = settings.value(self._fit_model_class_key())
            if isinstance(value, str):
                state = json.loads(value)
            elif isinstance(value, dict):
                state = value
            else:
                return

            saved_keys: set[str] = set()

            def _collect(node):
                if isinstance(node, dict):
                    wk = node.get("widget_key")
                    if isinstance(wk, str):
                        saved_keys.add(wk)
                    for v in node.values():
                        _collect(v)
                elif isinstance(node, list):
                    for v in node:
                        _collect(v)

            _collect(state)
            current_keys = {self._plot_widget_key(c) for c in self._plot_containers}
            if current_keys - saved_keys:
                # Saved layout is stale (a plot was added since) — skip restore.
                return

            self.plot_tab_widget.set_layout_state(
                state,
                key_func=self._plot_widget_key,
                emit_change=False,
            )
        except Exception as exc:
            try:
                cs.logging.warning(f"Failed to restore fit dock layout: {exc}")
            except Exception:
                pass

    def ensure_plot_created(self, idx: int):
        # Create plot for given index if not yet created
        if idx < 0 or idx >= len(self._plot_specs):
            return None
        if self._plots_all[idx] is not None:
            return self._plots_all[idx]
        plot_class, kwargs = self._plot_specs[idx]
        try:
            plot = plot_class(self.fit, **kwargs)
        except Exception as e:
            # Provide a fallback widget to avoid breaking the tab UI
            fallback = QtWidgets.QLabel(f"Failed to create plot: {getattr(plot_class, 'name', plot_class.__name__)}\n{e}")
            self._plot_containers[idx].layout().addWidget(fallback)
            self._plots_all[idx] = fallback
            return fallback
        # Attach to container and control layout
        plot.plot_controller.hide()
        self._plot_containers[idx].layout().addWidget(plot, stretch=1)
        self._control_layout.addWidget(plot.plot_controller)
        # Track in storage lists
        self._plots_all[idx] = plot
        self._created_plots.append(plot)
        
        # Connect LinePlot region changes to the Fit widget's range selector
        try:
            region_changed = getattr(plot, 'regionChanged', None)
            if region_changed is not None and hasattr(region_changed, 'connect') and self.fit_widget is not None:
                def _sync_fit_widget_range(xmin: int, xmax: int, fw=self.fit_widget):
                    # Update only the UI of the fit widget to reflect the plot's region
                    # The underlying fit_range is already updated inside the plot via cs.run
                    try:
                        fw.blockSignals(True)
                        fw.xmin = xmin
                        fw.xmax = xmax
                    finally:
                        fw.blockSignals(False)
                region_changed.connect(_sync_fit_widget_range)
        except Exception:
            pass
        
        return plot

    def on_change_plot(self):
        idx = self.plot_tab_widget.currentIndex()
        # Ensure the selected tab's plot exists
        plot = self.ensure_plot_created(idx)
        # Toggle controllers
        try:
            self.current_plot_controller.hide()
        except Exception:
            pass
        if plot is None or not hasattr(plot, 'plot_controller'):
            return
        self.current_plot_controller = plot.plot_controller
        self.current_plot_controller.show()
        # Ensure the newly visible plot refreshes its content; we defer the
        # heavy update to the next event-loop turn to avoid deep re-entrancy
        # during fit creation.
        try:
            update_all = getattr(plot, 'update_all', None)
            if callable(update_all):
                QtCore.QTimer.singleShot(0, update_all)
            elif hasattr(plot, 'update'):
                QtCore.QTimer.singleShot(0, plot.update)
        except Exception:
            try:
                plot.update()
            except Exception:
                pass

    def updateStatusBar(self, msg: str):
        self.statusBar().showMessage(msg)

    def closeEvent(self, event: QtCore.QEvent):
        self.save_fit_dock_layout_state()
        # Honour a per-window opt-out flag (used by macros/app shutdown) as
        # well as the global confirm_close_fit setting.
        if getattr(self, 'close_confirm', True) and cs.core.settings.gui['confirm_close_fit']:
            reply = cs.gui.widgets.MyMessageBox.question(
                self,
                'Message',
                "Are you sure to close this fit?:\n%s" % self.fit.name,
                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                try:
                    fit_idx = getattr(self.fit, "fit_idx", 0)
                    cs.core.actions.dispatch(name="fit.close", payload={"idx": fit_idx})
                except Exception:
                    pass
                cs.gui.widgets.hide_items_in_layout(cs.cs.modelLayout)
                header_layout = getattr(cs.cs, "analysisHeaderLayout", None)
                if header_layout is not None:
                    cs.gui.widgets.hide_items_in_layout(header_layout)
                cs.gui.widgets.hide_items_in_layout(cs.cs.plotOptionsLayout)
            else:
                event.ignore()
        else:
            event.accept()



    def toggle_code_view(self):
        if self.stack.currentIndex() == 0:
            self.flip_to_code_btn.setText("Plots")
            self.flip_to_code_btn.setChecked(True)
            self.show_code_view()
        else:
            self.flip_to_code_btn.setText("Code")
            self.flip_to_code_btn.setChecked(False)
            self.stack.setCurrentIndex(0)

    def _model_view_spec_path(self):
        """Return the model's user-editable ``view.json`` path, or ``None``.

        Resolves the model's ``view_spec_file`` (PRD-38) next to the module that
        defines the model class, so the code view can open it alongside the
        model source.
        """
        try:
            from chisurf.gui.devtools.source_jump import resolve_model_view_spec_path
            target = resolve_model_view_spec_path(self.fit.model)
            return target[0] if target else None
        except Exception:
            return None

    def show_code_view(self):
        import inspect
        import pathlib
        from chisurf.gui.devtools.source_jump import resolve_compute_model_class
        # Resolve the underlying *compute* model class so "Code" opens the pure
        # model source (e.g. core/models/tcspc/lifetime.py) and its co-located
        # view.json — not the GUI widget wrapper that multiply-inherits it
        # (PRD-38).
        model_class = resolve_compute_model_class(self.fit.model) or self.fit.model.__class__
        try:
            source_file = inspect.getsourcefile(model_class)
            if not source_file:
                return
            
            models_dir = pathlib.Path(source_file).parent
            self.file_combo.blockSignals(True)
            self.file_combo.clear()

            py_files = sorted(models_dir.glob("*.py"))
            for p in py_files:
                self.file_combo.addItem(p.name, str(p))
            # Also list the model's user-editable view.json editor specs so the
            # computation and its UI layout are both reachable (PRD-38).
            for p in sorted(models_dir.glob("*.view.json")):
                self.file_combo.addItem(p.name, str(p))

            idx = self.file_combo.findData(str(pathlib.Path(source_file)))
            if idx >= 0:
                self.file_combo.setCurrentIndex(idx)
            self.file_combo.blockSignals(False)

            # Open the model's own view.json as a background tab first (when it
            # has one), then load the model source so the code is the focused
            # tab. Clicking "Code" thus shows both the computation and its JSON
            # editor layout (PRD-38).
            view_json = self._model_view_spec_path()
            if view_json is not None:
                try:
                    self.code_editor.open_file(view_json)
                except Exception as exc:
                    cs.logging.debug(f"Failed to open model view.json: {exc}")

            self.load_code_file(source_file)
            self.stack.setCurrentIndex(1)
        except Exception as e:
            from qtpy import QtWidgets
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load model source: {e}")

    def _get_current_text_editor(self):
        """Return the currently active TextEditor inside the CodeEditor."""
        return self.code_editor._get_current_editor()

    def _code_nav_back(self):
        editor = self._get_current_text_editor()
        if editor is not None:
            editor.navigate_back()

    def _code_nav_forward(self):
        editor = self._get_current_text_editor()
        if editor is not None:
            editor.navigate_forward()

    def _on_code_editor_created(self, editor):
        """Called when a new editor tab is created inside CodeEditor."""
        editor.external_definition_callback = self._open_external_definition
        editor.file_load_callback = self.load_code_file

    def _sync_code_symbol_combo(self, symbols):
        """Populate the function combo from shared editor symbols."""
        self.func_combo.blockSignals(True)
        self.func_combo.clear()
        self.func_combo.addItem("Select...", -1)
        for symbol in symbols:
            line = getattr(symbol, "line", 1)
            kind = getattr(symbol, "kind", "")
            name = getattr(symbol, "display_name", getattr(symbol, "name", ""))
            prefix = "  " if kind == "method" else ""
            self.func_combo.addItem(f"{prefix}{name}", max(0, int(line) - 1))
        self.func_combo.blockSignals(False)

    def _open_external_definition(self, file_path, line_number):
        """Open an external file in the code editor and jump to the given line."""
        self.code_editor.open_file(file_path, line=line_number)

    def load_code_file(self, file_path, line_number: int = 0):
        self.original_source_file = file_path
        self.code_editor.open_file(file_path)
        editor = self._get_current_text_editor()
        if editor is None:
            return
        editor.current_file = file_path

        if line_number > 0:
            doc = editor.document()
            block = doc.findBlockByNumber(line_number)
            if block.isValid():
                cursor = editor.textCursor()
                cursor.setPosition(block.position())
                editor.setTextCursor(cursor)
                editor.centerCursor()

        self._sync_code_symbol_combo(editor.refresh_symbols())
        if not editor._nav_history:
            editor.push_nav_history(file_path, 0)

    def on_code_file_selected(self, idx):
        if idx < 0: return
        file_path = self.file_combo.itemData(idx)
        self.load_code_file(file_path)

    def on_code_func_selected(self, idx):
        if idx < 0:
            return
        line_num = self.func_combo.itemData(idx)
        if line_num >= 0:
            editor = self._get_current_text_editor()
            if editor is None:
                return
            doc = editor.document()
            block = doc.findBlockByNumber(line_num)
            cursor = editor.textCursor()
            cursor.setPosition(block.position())
            editor.setTextCursor(cursor)
            editor.centerCursor()
            editor.setFocus()

    def save_model_code(self):
        editor = self._get_current_text_editor()
        if editor is None:
            return
        code = editor.text()
        if not hasattr(self, 'original_source_file'):
            return

        source_file = self.original_source_file
        import os
        import inspect
        from chisurf.core.settings.path_utils import get_path
        
        target_file = source_file
        if not os.access(source_file, os.W_OK):
            import datetime
            import pathlib
            models_dir = get_path('settings') / 'models'
            models_dir.mkdir(parents=True, exist_ok=True)
            basename = pathlib.Path(source_file).name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            target_file = str(models_dir / f"{basename}_{timestamp}.py")
            
        try:
            with open(target_file, "w") as f:
                f.write(code)
            self.updateStatusBar(f"Saved to {target_file}")
            
            # Check if this is the model file. Resolve against the *compute*
            # model class so an edit to the pure model source (what "Code" now
            # opens, PRD-38) is recognised even when the live instance is a
            # legacy widget that multiply-inherits it.
            from chisurf.gui.devtools.source_jump import resolve_compute_model_class
            instance_class = self.fit.model.__class__
            model_class = resolve_compute_model_class(self.fit.model) or instance_class
            if source_file == inspect.getsourcefile(model_class) or target_file != source_file:
                # dynamically apply the code
                import sys
                module = sys.modules.get(model_class.__module__)
                if module:
                    exec(code, module.__dict__)

                    class_name = model_class.__name__
                    new_class = getattr(module, class_name, None)
                    if new_class:
                        # Only swap the instance class when the live object *is*
                        # the pure compute model. Replacing a legacy widget's
                        # class with the pure model would strip its Qt behaviour;
                        # there the redefined module is enough for fresh fits.
                        if instance_class is model_class:
                            self.fit.model.__class__ = new_class
                        fc = get_fitting_client()
                        if fc is not None:
                            fc.update_fit(fit_index=getattr(self.fit, "fit_idx", None))
                        self.updateStatusBar("Model code applied successfully.")
        except Exception as e:
            from qtpy import QtWidgets
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save and apply code: {e}")
