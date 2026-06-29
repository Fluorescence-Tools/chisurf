"""Window classes for SM Acquisition plugin."""

import numpy as np
from qtpy.QtWidgets import (
    QMdiSubWindow,
    QDockWidget,
    QGroupBox,
    QGridLayout,
    QComboBox,
    QCheckBox,
    QToolButton,
    QSpinBox,
    QDoubleSpinBox,
    QProgressBar,
    QLabel,
    QLCDNumber,
    QSizePolicy,
)
from qtpy.QtGui import QFont
from qtpy.QtCore import Qt
import pyqtgraph as pg
import chisurf
from chisurf.gui.widgets.mdi_custom_titlebar import CustomMdiSubWindow
from .controllers import (
    DecayPlotController,
    CorrelationPlotController,
    CountRatePlotController,
    MCSPlotController,
    MacrotimePlotController,
)


class DecayWindow(CustomMdiSubWindow):
    """Window for displaying fluorescence decays."""

    def __init__(self, parent=None):
        """Initialize the decay window."""
        super().__init__(title="Acquisition - Fluorescence Decays", parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        # Create the decay plot widget - single plot instead of stacked
        self.decay_plot_widget = pg.PlotWidget()
        self.decay_plot_widget.setLogMode(y=False)  # Linear scale to show zeros
        self.decay_plot_widget.setLabel('left', 'Counts')
        self.decay_plot_widget.setLabel('bottom', 'Time (ns)')

        # Create curves for each channel (4 channels)
        self.decay_curves = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow
        for i in range(4):
            curve = self.decay_plot_widget.plot(pen=pg.mkPen(color=colors[i], width=2))
            self.decay_curves.append(curve)

        # Set the plot widget as content
        self.set_content(self.decay_plot_widget)

        # Set reasonable default size from settings (same as fit windows)
        xs, ys = chisurf.settings.gui['fit_windows_size']
        self.resize(xs, ys)

        # Add dummy attributes to prevent AttributeError in main window
        self.fit = None
        self.fit_widget = None

        # Create plot controller and parent it to this window so it survives
        # layout changes when the main GUI switches contexts.
        self.plot_controller = DecayPlotController(self)

        # Set current_plot_controller after plot_controller is created
        self.current_plot_controller = self.plot_controller

    def closeEvent(self, event):
        """Handle window close event."""
        # Update checkbox state when window is closed
        import chisurf
        if hasattr(chisurf.cs, '_acquisition_manager'):
            chisurf.cs._acquisition_manager.acquisition_dock.show_decay_checkbox.setChecked(False)
        event.accept()

    def update_decays(self, decay_data):
        """Update the decay plots with new data."""
        for i, decay in enumerate(decay_data):
            if len(decay) > 0:
                self.decay_data[i] = decay

        # Update the plot using controller settings
        self.update_decay_plot()

    def update_decay_plot(self):
        """Update the decay plot based on controller settings."""
        import chisurf
        if hasattr(chisurf.cs, '_acquisition_manager'):
            manager = chisurf.cs._acquisition_manager
            manager.update_decay_plot()


class CorrelationWindow(CustomMdiSubWindow):
    """Window for displaying correlation curves."""

    def __init__(self, parent=None):
        """Initialize the correlation window."""
        super().__init__(title="Acquisition - Correlation Curve", parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        # Create the correlation plot widget
        self.correlation_plot_widget = pg.PlotWidget()
        self.correlation_plot_widget.setLogMode(x=True)
        self.correlation_plot_widget.setLabel('left', 'G(\u03c4)')
        self.correlation_plot_widget.setLabel('bottom', '\u03c4 (ms)')

        # Create curves for each correlation (4 curves)
        self.correlation_curves = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow
        for i in range(4):
            curve = self.correlation_plot_widget.plot(pen=pg.mkPen(color=colors[i], width=2))
            self.correlation_curves.append(curve)

        # Set the plot widget as content
        self.set_content(self.correlation_plot_widget)

        # Set reasonable default size from settings (same as fit windows)
        xs, ys = chisurf.settings.gui['fit_windows_size']
        self.resize(xs, ys)

        # Add dummy attributes to prevent AttributeError in main window
        self.fit = None
        self.fit_widget = None

        # Create plot controller
        self.plot_controller = CorrelationPlotController(self)

        # Set current_plot_controller after plot_controller is created
        self.current_plot_controller = self.plot_controller

    def closeEvent(self, event):
        """Handle window close event."""
        # Update checkbox state when window is closed
        import chisurf
        if hasattr(chisurf.cs, '_acquisition_manager'):
            chisurf.cs._acquisition_manager.acquisition_dock.show_correlation_checkbox.setChecked(False)
        event.accept()

    def update_correlation(self, times, amplitudes):
        """Update the correlation plot with new data."""
        if times is not None and amplitudes is not None:
            self.correlation_times = times
            self.correlation_amplitudes = amplitudes

        # Update the plot using controller settings
        self.update_correlation_plot()

    def update_correlation_plot(self):
        """Update the correlation plot based on controller settings."""
        import chisurf
        if hasattr(chisurf.cs, '_acquisition_manager'):
            manager = chisurf.cs._acquisition_manager
            manager.update_correlation_plot()


class CountRateWindow(CustomMdiSubWindow):
    """Window for displaying count rate traces."""

    def __init__(self, parent=None):
        """Initialize the count rate window."""
        super().__init__(title="Acquisition - Count Rate", parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        # Create the count rate plot widget
        self.count_rate_plot_widget = pg.PlotWidget()
        self.count_rate_plot_widget.setLabel('left', 'Count Rate (cps)')
        self.count_rate_plot_widget.setLabel('bottom', 'Macrotime (s)')
        # Minimize padding/margins inside the plot
        try:
            self.count_rate_plot_widget.setContentsMargins(0, 0, 0, 0)
            plot_item = self.count_rate_plot_widget.getPlotItem()
            if hasattr(plot_item, 'setContentsMargins'):
                plot_item.setContentsMargins(0, 0, 0, 0)
            if hasattr(plot_item, 'layout') and plot_item.layout is not None:
                plot_item.layout.setContentsMargins(0, 0, 0, 0)
                plot_item.layout.setSpacing(0)
            # Hide unused axes and buttons
            try:
                plot_item.showAxis('top', False)
                plot_item.showAxis('right', False)
                if hasattr(plot_item, 'hideButtons'):
                    plot_item.hideButtons()
                if hasattr(plot_item, 'setMenuEnabled'):
                    plot_item.setMenuEnabled(False)
            except Exception:
                pass
            # Reduce axis spacing
            try:
                for ax in ('left', 'bottom', 'right', 'top'):
                    axis = plot_item.getAxis(ax)
                    if axis is not None:
                        axis.setStyle(tickLength=0, autoExpandTextSpace=False, tickTextOffset=0)
                # Shrink tick font on visible axes
                small_font = QFont()
                small_font.setPointSize(8)
                for ax in ('left', 'bottom'):
                    axis = plot_item.getAxis(ax)
                    if axis is not None and hasattr(axis, 'setTickFont'):
                        axis.setTickFont(small_font)
            except Exception:
                pass
            vb = plot_item.getViewBox()
            if hasattr(vb, 'setDefaultPadding'):
                vb.setDefaultPadding(0.0)
        except Exception:
            pass

        # Create curves for each channel (4 channels) plus "All"
        self.count_rate_curves = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 128, 128)]  # Red, Green, Blue, Yellow, Gray for "All"
        for i in range(5):  # 4 channels + 1 for "All"
            curve = self.count_rate_plot_widget.plot(pen=pg.mkPen(color=colors[i], width=2))
            self.count_rate_curves.append(curve)

        # Store horizontal mean lines
        self.mean_lines = []

        # Set the plot widget as content
        self.set_content(self.count_rate_plot_widget)

        # Set reasonable default size from settings (same as fit windows)
        xs, ys = chisurf.settings.gui['fit_windows_size']
        self.resize(xs, ys)

        # Add dummy attributes to prevent AttributeError in main window
        self.fit = None
        self.fit_widget = None

        # Create plot controller
        self.plot_controller = CountRatePlotController(self)

        # Set current_plot_controller after plot_controller is created
        self.current_plot_controller = self.plot_controller

    def closeEvent(self, event):
        """Handle window close event."""
        # Update checkbox state when window is closed
        import chisurf
        if hasattr(chisurf.cs, '_acquisition_manager'):
            chisurf.cs._acquisition_manager.acquisition_dock.show_count_rate_checkbox.setChecked(False)
        event.accept()

    def update_count_rate(self, time_data, count_rate_data):
        """Update the count rate plot with new data."""
        if time_data is not None and count_rate_data is not None:
            self.time_data = time_data
            self.count_rate_data = count_rate_data

    def update_count_rate_plot(self):
        """Update the count rate plot based on controller settings."""
        import chisurf
        if hasattr(chisurf.cs, '_acquisition_manager'):
            manager = chisurf.cs._acquisition_manager
            manager.update_count_rate_plot()


class MCSWindow(CustomMdiSubWindow):
    """Window for displaying MCS trace."""

    def __init__(self, parent=None):
        """Initialize the MCS window."""
        super().__init__(title="Acquisition - MCS Trace", parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        # Create the MCS plot widget
        self.mcs_plot_widget = pg.PlotWidget()
        self.mcs_plot_widget.setLabel('left', 'Intensity (counts)')
        self.mcs_plot_widget.setLabel('bottom', 'Time (ms)')

        self.mcs_curve = self.mcs_plot_widget.plot(pen=pg.mkPen(color=(0, 150, 150), width=2))

        # Set the plot widget as content
        self.set_content(self.mcs_plot_widget)

        # Set reasonable default size
        xs, ys = chisurf.settings.gui['fit_windows_size']
        self.resize(xs, ys)

        # Create plot controller
        self.plot_controller = MCSPlotController(self)

        # Set current_plot_controller after plot_controller is created
        self.current_plot_controller = self.plot_controller

    def closeEvent(self, event):
        """Handle window close event."""
        # Update checkbox state when window is closed
        import chisurf
        if hasattr(chisurf.cs, '_acquisition_manager'):
            chisurf.cs._acquisition_manager.acquisition_dock.show_mcs_checkbox.setChecked(False)
        event.accept()


class MacrotimeWindow(CustomMdiSubWindow):
    """Window for displaying macrotime time series plot."""

    def __init__(self, parent=None):
        """Initialize the macrotime window."""
        super().__init__(title="Acquisition - Macrotime Plot", parent=parent)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        # Create the macrotime plot widget
        self.macrotime_plot_widget = pg.PlotWidget()
        self.macrotime_plot_widget.setLabel('left', 'Macrotime Difference')
        self.macrotime_plot_widget.setLabel('bottom', 'Time (s)')

        self.macrotime_curve = self.macrotime_plot_widget.plot(pen=pg.mkPen(color=(150, 75, 0), width=2))

        # Set the plot widget as content
        self.set_content(self.macrotime_plot_widget)

        # Set reasonable default size from settings (same as fit windows)
        xs, ys = chisurf.settings.gui['fit_windows_size']
        self.resize(xs, ys)

        # Add dummy attributes to prevent AttributeError in main window
        self.fit = None
        self.fit_widget = None

        # Create plot controller
        self.plot_controller = MacrotimePlotController(self)

        # Set current_plot_controller after plot_controller is created
        self.current_plot_controller = self.plot_controller

    def closeEvent(self, event):
        """Handle window close event."""
        # Update checkbox state when window is closed
        import chisurf
        if hasattr(chisurf.cs, '_acquisition_manager'):
            chisurf.cs._acquisition_manager.acquisition_dock.show_macrotime_checkbox.setChecked(False)
        event.accept()

    def update_macrotime(self, histogram_data):
        """Update the macrotime plot with new histogram data."""
        if histogram_data is not None:
            bins, counts = histogram_data
            self.macrotime_bins = bins
            self.macrotime_counts = counts

        # Update the plot using controller settings
        self.update_macrotime_plot()

    def update_macrotime_plot(self):
        """Update the macrotime plot based on controller settings."""
        import chisurf
        if hasattr(chisurf.cs, '_acquisition_manager'):
            manager = chisurf.cs._acquisition_manager
            manager.update_macrotime_plot()
