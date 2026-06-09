from __future__ import annotations
import chisurf as cs

import os
import typing
import pathlib
import textwrap

import numpy as np
import pyqtgraph as pg
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
from chisurf.core.math.optimization.leastsqbound import OptimizationCancelled
from chisurf.gui.widgets.dock_area import DockArea


class FitSubWindow(CustomMdiSubWindow):

    def update(self, *args):
        super().update(self, *args)
        self.plot_tab_widget.update(*args)

    def __init__(
            self,
            fit: cs.core.fitting.fit.FitGroup,
            control_layout: QtWidgets.QLayout,
            fit_widget: 'FittingControllerWidget' = None,
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
        self.nav_back_btn.setText("<")
        self.nav_back_btn.clicked.connect(lambda: self.code_editor.navigate_back())
        
        self.nav_forward_btn = QtWidgets.QToolButton()
        self.nav_forward_btn.setText(">")
        self.nav_forward_btn.clicked.connect(lambda: self.code_editor.navigate_forward())
        
        self.file_combo = QtWidgets.QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_code_file_selected)
        
        self.func_combo = QtWidgets.QComboBox()
        self.func_combo.currentIndexChanged.connect(self.on_code_func_selected)

        self.save_code_btn = QtWidgets.QToolButton()
        self.save_code_btn.setText("Save/Apply")
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

        from chisurf.plugins.core.code_editor.text_editor import TextEditor
        self.code_editor = TextEditor(self, language="python")
        self.code_editor.file_load_callback = self.load_code_file
        self.back_layout.addWidget(self.code_editor)
        self.stack.addWidget(self.back_widget)

        rect = self.plot_tab_widget.geometry()
        self.setGeometry(rect)

        self.current_plot_controller = QtWidgets.QWidget(self)
        self.current_plot_controller.hide()

        # Lazy plot instantiation: create lightweight tab containers now, build plots on demand
        self._control_layout = control_layout
        self._plot_specs = list(fit.model.plot_classes)
        self._plot_containers = []
        self._plots_all = [None] * len(self._plot_specs)      # positional storage
        self._created_plots = []                               # actual created plots (shared)
        # Create empty containers per tab
        for (plot_class, kwargs) in self._plot_specs:
            container = QtWidgets.QWidget()
            container.setLayout(QtWidgets.QVBoxLayout())
            container.layout().setContentsMargins(0, 0, 0, 0)
            container.layout().setSpacing(0)
            self._plot_containers.append(container)
            tab_name = getattr(plot_class, 'name', None)
            if not isinstance(tab_name, str):
                tab_name = getattr(plot_class, '__name__', str(plot_class))
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

    def show_code_view(self):
        import inspect
        import pathlib
        model_class = self.fit.model.__class__
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
            
            idx = self.file_combo.findData(str(pathlib.Path(source_file)))
            if idx >= 0:
                self.file_combo.setCurrentIndex(idx)
            self.file_combo.blockSignals(False)
            
            self.load_code_file(source_file)
            self.stack.setCurrentIndex(1)
        except Exception as e:
            from qtpy import QtWidgets
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load model source: {e}")

    def load_code_file(self, file_path):
        import re
        self.original_source_file = file_path
        self.code_editor.current_file = file_path
        with open(file_path, "r") as f:
            code = f.read()
        self.code_editor.setText(code)
        
        self.func_combo.blockSignals(True)
        self.func_combo.clear()
        self.func_combo.addItem("Select...", -1)
        
        lines = code.split('\n')
        for i, line in enumerate(lines):
            m = re.match(r'^ *(def |class )([a-zA-Z0-9_]+)', line)
            if m:
                indent = len(line) - len(line.lstrip())
                prefix = " " * indent
                self.func_combo.addItem(f"{prefix}{m.group(1)}{m.group(2)}", i)
                
        self.func_combo.blockSignals(False)
        if not self.code_editor._nav_history:
            self.code_editor.push_nav_history(file_path, 0)

    def on_code_file_selected(self, idx):
        if idx < 0: return
        file_path = self.file_combo.itemData(idx)
        self.load_code_file(file_path)

    def on_code_func_selected(self, idx):
        if idx < 0: return
        line_num = self.func_combo.itemData(idx)
        if line_num >= 0:
            doc = self.code_editor.document()
            block = doc.findBlockByNumber(line_num)
            cursor = self.code_editor.textCursor()
            cursor.setPosition(block.position())
            self.code_editor.setTextCursor(cursor)
            self.code_editor.centerCursor()
            self.code_editor.setFocus()

    def save_model_code(self):
        code = self.code_editor.text()
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
            
            # Check if this is the model file
            model_class = self.fit.model.__class__
            if source_file == inspect.getsourcefile(model_class) or target_file != source_file:
                # dynamically apply the code
                import sys
                module = sys.modules.get(model_class.__module__)
                if module:
                    exec(code, module.__dict__)
                    
                    class_name = model_class.__name__
                    new_class = getattr(module, class_name, None)
                    if new_class:
                        self.fit.model.__class__ = new_class
                        self.fit.update()
                        self.updateStatusBar("Model code applied successfully.")
        except Exception as e:
            from qtpy import QtWidgets
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save and apply code: {e}")