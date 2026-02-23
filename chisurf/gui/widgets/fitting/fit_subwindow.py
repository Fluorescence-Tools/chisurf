from __future__ import annotations

import os
import typing
import pathlib
import textwrap

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, uic, QtCore, QtGui
import matplotlib.colors as mcolors

import chisurf.data
import chisurf.fitting
import chisurf.decorators
import chisurf.gui.decorators
import chisurf.settings

import chisurf.gui.widgets
import chisurf.gui.widgets.experiments.widgets
from chisurf.gui.widgets import Controller
from chisurf.gui.widgets.mdi_custom_titlebar import CustomMdiSubWindow
from chisurf.math.optimization.leastsqbound import OptimizationCancelled


class FitSubWindow(CustomMdiSubWindow):

    def update(self, *args):
        super().update(self, *args)
        self.plot_tab_widget.update(*args)

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
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

        self.plot_tab_widget = QtWidgets.QTabWidget(self)
        layout.addWidget(self.plot_tab_widget)
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
            chisurf.gui.QtWidgets.QMdiSubWindow.RubberBandResize,
            chisurf.settings.gui['RubberBandResize']
        )
        self.setOption(
            chisurf.gui.QtWidgets.QMdiSubWindow.RubberBandMove,
            chisurf.settings.gui['RubberBandMove']
        )

        # Set windows icon
        try:
            icon = fit.model.icon
        except AttributeError:
            icon = chisurf.gui.QtGui.QIcon(":/icons/icons/list-add.png")
        self.setWindowIcon(icon)

        # Set global style sheet
        # window_style = chisurf.settings.gui['fit_window_style']
        # self.setStyleSheet(chisurf.settings.style_sheet)

        self.setAttribute(chisurf.gui.QtCore.Qt.WA_DeleteOnClose, True)

        # Resize window
        xs, ys = chisurf.settings.gui['fit_windows_size']
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
        self._plot_containers[idx].layout().addWidget(plot)
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
                    # The underlying fit_range is already updated inside the plot via chisurf.run
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
        if getattr(self, 'close_confirm', True) and chisurf.settings.gui['confirm_close_fit']:
            reply = chisurf.gui.widgets.MyMessageBox.question(
                self,
                'Message',
                "Are you sure to close this fit?:\n%s" % self.fit.name,
                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                try:
                    chisurf.action_controller.execute(name="fit.close", payload={})
                except Exception:
                    pass
                chisurf.gui.widgets.hide_items_in_layout(chisurf.cs.modelLayout)
                header_layout = getattr(chisurf.cs, "analysisHeaderLayout", None)
                if header_layout is not None:
                    chisurf.gui.widgets.hide_items_in_layout(header_layout)
                chisurf.gui.widgets.hide_items_in_layout(chisurf.cs.plotOptionsLayout)
            else:
                event.ignore()
        else:
            event.accept()


