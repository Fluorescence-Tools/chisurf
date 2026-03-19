"""
Waterfall Plot Widget

A reusable PyQtGraph-based waterfall plot widget for TTTR data visualization.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QVBoxLayout
from qtpy.QtCore import Signal


class WaterfallPlotWidget(QWidget):
    """
    A widget for displaying TTTR waterfall plots with position indicator.
    
    Features:
    - Waterfall image display
    - Horizontal position indicator line
    - Proper axis labeling for TTTR data
    - Signal emission for user interactions
    """
    
    # Signals
    plot_clicked = Signal(float, float)  # x, y coordinates when plot is clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Plot components
        self.plot_widget = pg.PlotWidget(self)
        self.waterfall_img: Optional[pg.ImageItem] = None
        self.position_line: Optional[pg.InfiniteLine] = None
        
        # Data storage
        self.waterfall_data: Optional[np.ndarray] = None
        self.n_macro_bins: int = 0
        self.n_micro_bins: int = 0
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Setup plot widget
        self.plot_widget.setTitle("Microtime Waterfall")
        self.plot_widget.setLabel('left', 'Macrotime (s)')
        self.plot_widget.setLabel('bottom', 'Microtime (bins)')
        
        # Create waterfall image item
        self.waterfall_img = pg.ImageItem()
        self.plot_widget.addItem(self.waterfall_img)
        
        # Create position indicator (horizontal line that moves vertically)
        self.position_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=2))
        self.position_line.setVisible(False)
        self.plot_widget.addItem(self.position_line)
        
        layout.addWidget(self.plot_widget)
    
    def _connect_signals(self):
        """Connect signals."""
        # Emit plot_clicked signal when user clicks on the plot
        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)
    
    def _on_plot_clicked(self, event):
        """Handle mouse clicks on the plot."""
        if event.button() == 1:  # Left click
            # Get the mouse position in plot coordinates
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
            x, y = mouse_point.x(), mouse_point.y()
            self.plot_clicked.emit(x, y)
    
    def set_waterfall_data(
        self,
        rgb_data: np.ndarray,
        macro_t_s: np.ndarray,
        micro_centers: np.ndarray,
        n_macro_bins: int,
        n_micro_bins: int,
    ):
        """
        Set the waterfall data and update the plot.

        Keeps RGB color mixing but uses alpha channel to represent intensity.

        Args:
            rgb_data: RGB image data (shape: n_micro, n_macro, 3)
            macro_t_s: Macrotime values for x-axis (seconds)
            micro_centers: Microtime bin centers for y-axis (bins)
            n_macro_bins: Number of macrotime bins
            n_micro_bins: Number of microtime bins
        """
        if rgb_data.ndim != 3 or rgb_data.shape[2] != 3:
            raise ValueError(f"rgb_data must have shape (n_micro, n_macro, 3), got {rgb_data.shape}")

        self.waterfall_data = rgb_data
        self.n_macro_bins = n_macro_bins
        self.n_micro_bins = n_micro_bins

        # ---- Normalize RGB to float32 in [0, 1] for stable alpha computation ----
        if np.issubdtype(rgb_data.dtype, np.integer):
            rgb = rgb_data.astype(np.float32) / 255.0
        else:
            rgb = rgb_data.astype(np.float32, copy=False)
            # if it looks like [0..255] floats, normalize
            if np.nanmax(rgb) > 1.5:
                rgb = rgb / 255.0

        rgb = np.clip(rgb, 0.0, 1.0)

        # ---- Compute intensity proxy from RGB (luminance) ----
        # You can change weights if your channel coloring has meaning.
        intensity = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])

        # ---- Map intensity -> alpha with log + robust scaling ----
        # This is the critical part that makes bursts pop.
        I = intensity

        # log1p-like compression in [0,1] domain:
        # (scale first so log has effect even for small values)
        # You can tune "log_gain" if needed.
        log_gain = 50.0
        I_log = np.log1p(log_gain * I) / np.log1p(log_gain)

        # robust quantile stretch
        finite = I_log[np.isfinite(I_log)]
        if finite.size:
            lo = float(np.quantile(finite, 0.01))
            hi = float(np.quantile(finite, 0.995))
            if hi <= lo:
                hi = lo + 1e-6
            alpha = (I_log - lo) / (hi - lo)
        else:
            alpha = I_log

        alpha = np.clip(alpha, 0.0, 1.0)

        # Optional: enforce a minimum alpha so low background is still faintly visible
        # Set to 0.0 to fully hide background.
        alpha_min = 0.02
        alpha = alpha_min + (1.0 - alpha_min) * alpha

        # Optional: gamma on alpha (gamma < 1 boosts faint signals; >1 suppresses)
        alpha_gamma = 0.7
        alpha = np.power(alpha, alpha_gamma)

        # ---- Compose RGBA (float 0..1) ----
        rgba = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3] = alpha

        # ---- Update image ----
        # Convert to uint8 for compatibility with PyQtGraph
        rgba_uint8 = (rgba * 255).astype(np.uint8)
        self.waterfall_img.setImage(rgba_uint8, autoLevels=False)

        # ---- Set axis ranges (keep your convention) ----
        self.plot_widget.setXRange(float(macro_t_s[0]), float(macro_t_s[-1]))
        self.plot_widget.setYRange(float(micro_centers[0]), float(micro_centers[-1]))
    
    def set_position(self, position: float):
        """
        Set the position of the indicator line.
        
        Args:
            position: Position value (in plot coordinates)
        """
        if self.position_line is not None:
            self.position_line.setPos(position)
    
    def show_position_indicator(self, show: bool = True):
        """
        Show or hide the position indicator line.
        
        Args:
            show: True to show, False to hide
        """
        if self.position_line is not None:
            self.position_line.setVisible(show)
    
    def reset_position(self):
        """Reset the position indicator to the start (bin 0)."""
        self.set_position(0)
    
    def get_bin_count(self) -> Tuple[int, int]:
        """
        Get the number of bins in the waterfall.
        
        Returns:
            Tuple of (n_macro_bins, n_micro_bins)
        """
        return self.n_macro_bins, self.n_micro_bins
    
    def clear_plot(self):
        """Clear the waterfall plot."""
        if self.waterfall_img is not None:
            self.waterfall_img.clear()
        self.waterfall_data = None
        self.n_macro_bins = 0
        self.n_micro_bins = 0
        self.show_position_indicator(False)
    
    def set_title(self, title: str):
        """Set the plot title."""
        self.plot_widget.setTitle(title)
    
    def get_plot_widget(self) -> pg.PlotWidget:
        """Get the underlying PlotWidget for advanced customization."""
        return self.plot_widget
