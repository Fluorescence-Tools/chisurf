import sys
import csv

import numpy as np
from scipy.optimize import least_squares
from qtpy import QtWidgets, QtCore
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QToolButton,
    QMessageBox,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QTextEdit,
    QSplitter,
    QTabWidget,
    QProgressBar,
)

import pyqtgraph as pg

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c



name = "Imaging:PSF Determination"


@persist_plugin_state("psf_determination")
class PSFDeterminationWidget(QWidget):
    """PSF Determination plugin with 3D stack visualization and Gaussian PSF fitting.

    This plugin provides:

    - A *Load Stack* button to load 3D image data (e.g. bead stacks) from TIFF.
    - A pyqtgraph ``ImageView`` to browse slices along the z dimension.
    - Click-to-select bead positions for PSF extraction and fitting.
    - 3D Gaussian PSF fitting with parameter reporting (σ_x, σ_y, σ_z, FWHM, axial ratio).
    - Configurable pixel size, z step, and ROI dimensions.
    - Automated bead detection in 3D stacks with distance filtering.
    - Batch PSF fitting with text summary, progress bar, and CSV export.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PSF Determination")
        # Make window reasonably wide by default so results text is readable
        self.setMinimumSize(800, 600)
        self.resize(900, 720)

        self.stack = None
        self.selected_bead = None  # (z, y, x) in pixels
        self.bead_marker = None
        self.fit_roi = None  # CircleROI showing fitted lateral width
        self.detected_beads = []  # list of (z, y, x)
        self.detect_markers = []  # list of ScatterPlotItem for detected beads
        self.current_bead_index = -1

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Top bar with controls
        bar = QHBoxLayout()
        self.load_button = QToolButton()
        self.load_button.setText("Load Stack")
        self.load_button.setToolTip("Load 3D image stack (e.g. TIFF) for PSF inspection")
        self.load_button.clicked.connect(self.load_stack)
        bar.addWidget(self.load_button)

        self.info_label = QLabel("No stack loaded")
        self.info_label.setStyleSheet("color: #666666;")
        bar.addWidget(self.info_label, stretch=1)

        root.addLayout(bar)

        # Main splitter: left = viewer, right = controls/results
        splitter = QSplitter(QtCore.Qt.Horizontal)

        # Left: 3D stack viewer
        self.image_view = pg.ImageView()
        self.image_view.ui.histogram.setMinimumWidth(120)
        # Connect click events for bead selection
        self.image_view.getView().scene().sigMouseClicked.connect(self.on_image_clicked)
        # Track z-slice changes to update per-slice bead markers
        try:
            self.image_view.timeLine.sigPositionChanged.connect(self.on_slice_changed)
        except Exception:
            try:
                self.image_view.sigTimeChanged.connect(self.on_slice_changed)
            except Exception:
                pass
        splitter.addWidget(self.image_view)

        # Right: parameter controls, profiles, and results (tabbed)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Parameters group
        params_group = QGroupBox("PSF Parameters")
        params_layout = QGridLayout()
        params_layout.setContentsMargins(4, 4, 4, 4)
        params_layout.setHorizontalSpacing(4)
        params_layout.setVerticalSpacing(2)

        row = 0
        params_layout.addWidget(QLabel("Pixel size (nm):"), row, 0)
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(1, 1000)
        self.pixel_size_spin.setValue(100)
        self.pixel_size_spin.setDecimals(2)
        params_layout.addWidget(self.pixel_size_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("Z step (nm):"), row, 0)
        self.z_step_spin = QDoubleSpinBox()
        self.z_step_spin.setRange(1, 10000)
        self.z_step_spin.setValue(200)
        self.z_step_spin.setDecimals(2)
        params_layout.addWidget(self.z_step_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("ROI xy (pixels):"), row, 0)
        self.roi_xy_spin = QSpinBox()
        self.roi_xy_spin.setRange(3, 200)
        self.roi_xy_spin.setValue(15)
        params_layout.addWidget(self.roi_xy_spin, row, 1)

        row += 1
        params_layout.addWidget(QLabel("ROI z (slices):"), row, 0)
        self.roi_z_spin = QSpinBox()
        self.roi_z_spin.setRange(3, 200)
        self.roi_z_spin.setValue(15)
        params_layout.addWidget(self.roi_z_spin, row, 1)

        params_group.setLayout(params_layout)
        right_layout.addWidget(params_group)

        # Fit button
        self.fit_button = QToolButton()
        self.fit_button.setText("Fit Selected Bead")
        self.fit_button.setToolTip("Click on a bead in the image, then click here to fit 3D Gaussian PSF")
        self.fit_button.clicked.connect(self.fit_psf)
        self.fit_button.setEnabled(False)
        right_layout.addWidget(self.fit_button)

        # Bead detection and batch fitting controls
        detect_group = QGroupBox("Bead detection & batch")
        detect_layout = QGridLayout()
        detect_layout.setContentsMargins(4, 4, 4, 4)
        detect_layout.setHorizontalSpacing(4)
        detect_layout.setVerticalSpacing(2)

        row = 0
        detect_layout.addWidget(QLabel("Pixels per frame:"), row, 0)
        self.pixels_per_frame_spin = QSpinBox()
        self.pixels_per_frame_spin.setRange(1, 100000)
        self.pixels_per_frame_spin.setValue(20)
        detect_layout.addWidget(self.pixels_per_frame_spin, row, 1)

        row += 1
        detect_layout.addWidget(QLabel("Min distance (px):"), row, 0)
        self.min_distance_spin = QDoubleSpinBox()
        self.min_distance_spin.setRange(1.0, 1000.0)
        self.min_distance_spin.setValue(5.0)
        self.min_distance_spin.setDecimals(1)
        detect_layout.addWidget(self.min_distance_spin, row, 1)

        row += 1
        self.detect_beads_button = QToolButton()
        self.detect_beads_button.setText("Detect beads")
        self.detect_beads_button.setToolTip("Automatic bead detection in the stack using an adaptive quantile threshold")
        self.detect_beads_button.clicked.connect(self.detect_beads)
        detect_layout.addWidget(self.detect_beads_button, row, 0, 1, 2)

        row += 1
        detect_layout.addWidget(QLabel("Bead index:"), row, 0)
        self.bead_index_spin = QSpinBox()
        self.bead_index_spin.setRange(0, 0)
        self.bead_index_spin.setEnabled(False)
        self.bead_index_spin.valueChanged.connect(self.on_bead_index_changed)
        detect_layout.addWidget(self.bead_index_spin, row, 1)

        row += 1
        self.fit_all_button = QToolButton()
        self.fit_all_button.setText("Fit all detected")
        self.fit_all_button.setToolTip("Fit 3D Gaussian PSF for all detected beads and summarize results")
        self.fit_all_button.clicked.connect(self.fit_all_beads)
        self.fit_all_button.setEnabled(False)
        detect_layout.addWidget(self.fit_all_button, row, 0, 1, 2)

        row += 1
        self.export_csv_button = QToolButton()
        self.export_csv_button.setText("Export CSV")
        self.export_csv_button.setToolTip("Export batch PSF fit summary to CSV")
        self.export_csv_button.clicked.connect(self.export_batch_csv)
        detect_layout.addWidget(self.export_csv_button, row, 0, 1, 2)

        row += 1
        self.bead_info_label = QLabel("Detected beads: 0")
        detect_layout.addWidget(self.bead_info_label, row, 0, 1, 2)

        row += 1
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.batch_progress.setTextVisible(True)
        detect_layout.addWidget(self.batch_progress, row, 0, 1, 2)

        detect_group.setLayout(detect_layout)
        right_layout.addWidget(detect_group)

        # Tabbed area: Profiles and Fit Results
        self.tabs = QTabWidget()

        # ---- Profiles tab ----
        profiles_widget = QWidget()
        profiles_layout = QGridLayout(profiles_widget)
        profiles_layout.setContentsMargins(4, 4, 4, 4)
        profiles_layout.setHorizontalSpacing(4)
        profiles_layout.setVerticalSpacing(2)

        self.plot_x = pg.PlotWidget()
        self.plot_x.setLabel("left", "Intensity (a.u.)")
        self.plot_x.setLabel("bottom", "x (pixels)")
        self.plot_x.addLegend()
        profiles_layout.addWidget(QLabel("x-profile (z=z0; y=y0)"), 0, 0)
        profiles_layout.addWidget(self.plot_x, 1, 0)

        self.plot_y = pg.PlotWidget()
        self.plot_y.setLabel("left", "Intensity (a.u.)")
        self.plot_y.setLabel("bottom", "y (pixels)")
        self.plot_y.addLegend()
        profiles_layout.addWidget(QLabel("y-profile (z=z0; x=x0)"), 0, 1)
        profiles_layout.addWidget(self.plot_y, 1, 1)

        self.plot_z = pg.PlotWidget()
        self.plot_z.setLabel("left", "Intensity (a.u.)")
        self.plot_z.setLabel("bottom", "z (slices)")
        self.plot_z.addLegend()
        profiles_layout.addWidget(QLabel("z-profile (through center)"), 2, 0, 1, 2)
        profiles_layout.addWidget(self.plot_z, 3, 0, 1, 2)

        self.tabs.addTab(profiles_widget, "Profiles")

        # ---- Fit Results tab ----
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlainText("Click on a bead, then click 'Fit Selected Bead'.")
        results_layout.addWidget(self.results_text)

        self.tabs.addTab(results_widget, "Fit Results")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)
        self.setLayout(root)

    # -------------------------- data loading --------------------------
    def load_stack(self, path=None):
        """Load a 3D image stack from disk and show it in the viewer.

        Parameters
        ----------
        path : str or None
            Optional path to an image file. If None, a file dialog is opened.
        """
        # Handle boolean from clicked signal
        if isinstance(path, bool):
            path = None

        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Load Image Stack",
                "",
                "TIFF Files (*.tif *.tiff);;All Files (*)",
            )
            if not path:
                return

        # Try to read using imageio
        try:
            try:
                import imageio.v2 as imageio
            except ImportError:  # pragma: no cover - optional dependency
                import imageio
        except Exception:
            QMessageBox.critical(
                self,
                "Missing dependency",
                "The 'imageio' package is required to load image stacks.\n"
                "Install it via 'pip install imageio'.",
            )
            return

        try:
            data = imageio.imread(path)
        except Exception as e:  # pragma: no cover - depends on external files
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to read image stack:\n{e}",
            )
            return

        arr = np.asarray(data)
        if arr.ndim == 2:
            # Single plane → make it a 1-slice stack
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
            # RGB(A) → take first channel as intensity
            arr = arr[..., 0]
            arr = arr[np.newaxis, ...]
        elif arr.ndim != 3:
            QMessageBox.warning(
                self,
                "Unsupported shape",
                f"Expected 2D or 3D image stack, got shape {arr.shape}.",
            )
            return

        # Ensure float32 for plotting; internal layout is (z, y, x)
        self.stack = arr.astype(np.float32)

        # Reset bead detection state
        self.detected_beads = []
        self.current_bead_index = -1
        self.bead_index_spin.blockSignals(True)
        self.bead_index_spin.setRange(0, 0)
        self.bead_index_spin.setValue(0)
        self.bead_index_spin.setEnabled(False)
        self.bead_index_spin.blockSignals(False)
        self.bead_info_label.setText("Detected beads: 0")
        self.fit_all_button.setEnabled(False)
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        # Clear any previous detection markers from the viewer
        for m in self.detect_markers:
            self.image_view.getView().removeItem(m)
        self.detect_markers = []

        # IMPORTANT: Axis mapping
        #
        # self.stack.shape = (z, y, x)
        # We want image coordinates:
        #   x_image -> stack x-index  (axis 2)
        #   y_image -> stack y-index  (axis 1)
        # Therefore we tell ImageView:
        #   t-axis = 0 (z)
        #   x-axis = 2 (x)
        #   y-axis = 1 (y)
        self.image_view.setImage(self.stack, axes={"t": 0, "x": 2, "y": 1})
        z, ny, nx = self.stack.shape
        self.info_label.setText(f"Loaded stack: {nx}×{ny}×{z} (x×y×z)")

    # -------------------------- bead selection --------------------------
    def on_image_clicked(self, event):
        """Handle mouse clicks on the image to select bead positions."""
        if self.stack is None:
            return

        # Get click position in image coordinates
        pos = event.scenePos()
        if self.image_view.getImageItem().sceneBoundingRect().contains(pos):
            mouse_point = self.image_view.getImageItem().mapFromScene(pos)
            x, y = int(mouse_point.x()), int(mouse_point.y())
            
            # Get current z position from the time slider
            z = int(self.image_view.currentIndex)
            
            # Validate bounds
            nz, ny, nx = self.stack.shape
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                self.selected_bead = (z, y, x)
                self.info_label.setText(
                    f"Selected bead at (x={x}, y={y}, z={z}) | "
                    f"Stack: {nx}×{ny}×{nz} (x×y×z)"
                )
                self.fit_button.setEnabled(True)
                
                # Draw marker on the image
                if self.bead_marker is not None:
                    self.image_view.getView().removeItem(self.bead_marker)
                
                self.bead_marker = pg.ScatterPlotItem(
                    [x], [y],
                    pen=pg.mkPen("r", width=2),
                    brush=None,
                    size=15,
                    symbol="o"
                )
                self.image_view.getView().addItem(self.bead_marker)

    # -------------------------- bead detection / batch --------------------------
    def detect_beads(self):
        if self.stack is None:
            return

        nz, ny, nx = self.stack.shape
        roi_xy = self.roi_xy_spin.value()
        roi_z = self.roi_z_spin.value()

        # ROI half-sizes used both for detection and bounds checks
        half_xy = roi_xy // 2
        half_z = roi_z // 2

        pixels_per_frame = float(self.pixels_per_frame_spin.value())
        total_px = float(ny * nx)
        if total_px <= 0:
            return

        q_level = 1.0 - pixels_per_frame / total_px
        q_level = max(0.0, min(1.0, q_level))

        # Minimum lateral distance between beads (default: ROIxy/2)
        min_dist = float(self.min_distance_spin.value())
        if min_dist <= 0:
            min_dist = max(1.0, roi_xy / 2.0)

        # Step in z: max(4, ROIz/4)
        step_z = max(4, roi_z // 4)

        beads = []

        # Scan z from ROIz/2+2 .. size_z-ROIz/2-2
        z_start = half_z + 2
        z_end = nz - half_z - 2
        for z in range(z_start, max(z_start, z_end), step_z):
            frame = self.stack[z]
            if not np.any(np.isfinite(frame)):
                continue

            q = np.quantile(frame, q_level)
            mask = frame >= q
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue

            # Sort candidates by intensity (brightest first)
            intensities = frame[ys, xs]
            order = np.argsort(intensities)[::-1]
            ys = ys[order]
            xs = xs[order]

            accepted = []
            for y, x in zip(ys, xs):
                if not accepted:
                    accepted.append((y, x))
                    continue
                dy = np.array([y - ay for ay, ax in accepted], dtype=float)
                dx = np.array([x - ax for ay, ax in accepted], dtype=float)
                d2 = dx * dx + dy * dy
                if np.all(d2 >= min_dist * min_dist):
                    accepted.append((y, x))

            for y, x in accepted:
                # Skip beads too close to bounds so the full ROI fits into the stack
                if x < half_xy or x >= nx - half_xy:
                    continue
                if y < half_xy or y >= ny - half_xy:
                    continue
                if z < half_z or z >= nz - half_z:
                    continue
                beads.append((z, int(y), int(x)))

        # Clear previous detection markers
        for m in self.detect_markers:
            self.image_view.getView().removeItem(m)
        self.detect_markers = []

        self.detected_beads = beads
        n_beads = len(beads)
        self.bead_info_label.setText(f"Detected beads: {n_beads}")

        if n_beads == 0:
            self.bead_index_spin.blockSignals(True)
            self.bead_index_spin.setRange(0, 0)
            self.bead_index_spin.setValue(0)
            self.bead_index_spin.setEnabled(False)
            self.bead_index_spin.blockSignals(False)
            self.fit_all_button.setEnabled(False)
            return

        # Enable bead index navigation and batch fitting
        self.bead_index_spin.blockSignals(True)
        self.bead_index_spin.setRange(0, n_beads - 1)
        self.bead_index_spin.setValue(0)
        self.bead_index_spin.setEnabled(True)
        self.bead_index_spin.blockSignals(False)
        self.fit_all_button.setEnabled(True)

        # Select first bead
        self.current_bead_index = 0
        self._select_bead_by_index(0)
        # Show markers only for the current z-slice
        self._update_detect_markers_for_current_z()

    def _update_detect_markers_for_current_z(self):
        """Update green markers to show only beads in the current z-slice."""
        # Clear old markers
        for m in self.detect_markers:
            self.image_view.getView().removeItem(m)
        self.detect_markers = []

        if self.stack is None or not self.detected_beads:
            return

        try:
            z_curr = int(self.image_view.currentIndex)
        except Exception:
            return

        xs = [x for (z, y, x) in self.detected_beads if z == z_curr]
        ys = [y for (z, y, x) in self.detected_beads if z == z_curr]
        if not xs:
            return

        marker = pg.ScatterPlotItem(
            xs,
            ys,
            pen=pg.mkPen("g", width=1),
            brush=pg.mkBrush(0, 255, 0, 120),
            size=8,
            symbol="s",
        )
        marker.setZValue(5)
        self.image_view.getView().addItem(marker)
        self.detect_markers.append(marker)

    def on_slice_changed(self, *args):
        """Callback for ImageView time/slider changes to update bead markers."""
        self._update_detect_markers_for_current_z()

    def _select_bead_by_index(self, idx):
        if not (0 <= idx < len(self.detected_beads)):
            return
        z, y, x = self.detected_beads[idx]
        self.selected_bead = (z, y, x)
        self.fit_button.setEnabled(True)
        # Move viewer to bead z-slice
        try:
            self.image_view.setCurrentIndex(int(z))
        except Exception:
            pass

        # Update single-bead marker
        if self.bead_marker is not None:
            self.image_view.getView().removeItem(self.bead_marker)
        self.bead_marker = pg.ScatterPlotItem(
            [x],
            [y],
            pen=pg.mkPen("r", width=2),
            brush=None,
            size=15,
            symbol="o",
        )
        self.bead_marker.setZValue(9)
        self.image_view.getView().addItem(self.bead_marker)

        nz, ny, nx = self.stack.shape
        self.info_label.setText(
            f"Selected bead #{idx} at (x={x}, y={y}, z={z}) | "
            f"Stack: {nx}×{ny}×{nz} (x×y×z)"
        )

    def on_bead_index_changed(self, value):
        if not self.detected_beads:
            return
        self.current_bead_index = int(value)
        # Select bead and update marker
        self._select_bead_by_index(self.current_bead_index)
        # Automatically recompute PSF fit and update profiles/results
        # but keep the current tab (Profiles vs Fit Results) unchanged.
        current_tab = None
        if hasattr(self, "tabs"):
            current_tab = self.tabs.currentIndex()
        self.fit_psf()
        if current_tab is not None:
            self.tabs.setCurrentIndex(current_tab)

    def fit_all_beads(self):
        if self.stack is None or not self.detected_beads:
            return

        roi_xy = self.roi_xy_spin.value()
        roi_z = self.roi_z_spin.value()
        pixel_nm = self.pixel_size_spin.value()
        z_step_nm = self.z_step_spin.value()

        n_beads = len(self.detected_beads)
        lines = ["=== Batch PSF fits ===", f"Detected beads: {n_beads}", ""]

        # Initialize progress bar
        self.batch_progress.setRange(0, max(1, n_beads))
        self.batch_progress.setValue(0)

        idx = 0
        for z0, y0, x0 in self.detected_beads:
            roi, _ = self._extract_roi(z0, y0, x0, roi_xy, roi_z)
            if roi is None:
                lines.append(f"[{idx:03d}] x={x0}, y={y0}, z={z0}: skipped (ROI out of bounds)")
                idx += 1
                self.batch_progress.setValue(idx)
                QtWidgets.QApplication.processEvents()
                continue

            try:
                fit_result = self._fit_3d_gaussian(roi)
            except Exception as e:
                lines.append(f"[{idx:03d}] x={x0}, y={y0}, z={z0}: fit failed ({e})")
                idx += 1
                self.batch_progress.setValue(idx)
                QtWidgets.QApplication.processEvents()
                continue

            params = fit_result["params"]
            z_c, y_c, x_c = params[0], params[1], params[2]
            sigma_z, sigma_y, sigma_x = params[3], params[4], params[5]
            success = fit_result["success"]

            sigma_xy = (sigma_x + sigma_y) / 2.0
            fwhm_x_nm = 2.355 * sigma_x * pixel_nm
            fwhm_y_nm = 2.355 * sigma_y * pixel_nm
            fwhm_z_nm = 2.355 * sigma_z * z_step_nm
            sigma_xy_nm = sigma_xy * pixel_nm
            sigma_z_nm = sigma_z * z_step_nm
            axial_ratio = sigma_z_nm / sigma_xy_nm if sigma_xy_nm > 0 else float("nan")

            lines.append(
                f"[{idx:03d}] x={x0}, y={y0}, z={z0} | "
                f"σx={sigma_x:.2f}, σy={sigma_y:.2f}, σz={sigma_z:.2f} px | "
                f"FWHMxy≈{(fwhm_x_nm+fwhm_y_nm)/2:.1f} nm, FWHMz={fwhm_z_nm:.1f} nm | "
                f"axial={axial_ratio:.2f} | success={success}"
            )
            idx += 1

            # Update progress bar and keep UI responsive
            self.batch_progress.setValue(idx)
            QtWidgets.QApplication.processEvents()

        self.results_text.setPlainText("\n".join(lines))
        if hasattr(self, "tabs"):
            self.tabs.setCurrentIndex(1)

        # Mark completion
        self.batch_progress.setValue(self.batch_progress.maximum())

    def export_batch_csv(self):
        """Export batch PSF fit results for all detected beads to a CSV file.

        This recomputes fits for robustness, similar to fit_all_beads, but writes
        the summary as a table with one row per bead.
        """
        if self.stack is None or not self.detected_beads:
            return

        roi_xy = self.roi_xy_spin.value()
        roi_z = self.roi_z_spin.value()
        pixel_nm = self.pixel_size_spin.value()
        z_step_nm = self.z_step_spin.value()

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export PSF batch results to CSV",
            "psf_batch_results.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        header = [
            "index",
            "x_px",
            "y_px",
            "z_slice",
            "sigma_x_px",
            "sigma_y_px",
            "sigma_z_px",
            "FWHM_x_nm",
            "FWHM_y_nm",
            "FWHM_z_nm",
            "FWHM_xy_nm",
            "sigma_xy_nm",
            "sigma_z_nm",
            "axial_ratio",
            "success",
        ]

        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

                for idx, (z0, y0, x0) in enumerate(self.detected_beads):
                    roi, _ = self._extract_roi(z0, y0, x0, roi_xy, roi_z)
                    if roi is None:
                        writer.writerow([idx, x0, y0, z0] + ["NaN"] * (len(header) - 4))
                        continue

                    try:
                        fit_result = self._fit_3d_gaussian(roi)
                    except Exception:
                        writer.writerow([idx, x0, y0, z0] + ["NaN"] * (len(header) - 4))
                        continue

                    params = fit_result["params"]
                    sigma_z, sigma_y, sigma_x = params[3], params[4], params[5]
                    success = bool(fit_result["success"])

                    sigma_xy = (sigma_x + sigma_y) / 2.0
                    fwhm_x_nm = 2.355 * sigma_x * pixel_nm
                    fwhm_y_nm = 2.355 * sigma_y * pixel_nm
                    fwhm_z_nm = 2.355 * sigma_z * z_step_nm
                    fwhm_xy_nm = (fwhm_x_nm + fwhm_y_nm) / 2.0
                    sigma_xy_nm = sigma_xy * pixel_nm
                    sigma_z_nm = sigma_z * z_step_nm
                    axial_ratio = sigma_z_nm / sigma_xy_nm if sigma_xy_nm > 0 else float("nan")

                    writer.writerow(
                        [
                            idx,
                            x0,
                            y0,
                            z0,
                            f"{sigma_x:.5g}",
                            f"{sigma_y:.5g}",
                            f"{sigma_z:.5g}",
                            f"{fwhm_x_nm:.5g}",
                            f"{fwhm_y_nm:.5g}",
                            f"{fwhm_z_nm:.5g}",
                            f"{fwhm_xy_nm:.5g}",
                            f"{sigma_xy_nm:.5g}",
                            f"{sigma_z_nm:.5g}",
                            f"{axial_ratio:.5g}",
                            int(success),
                        ]
                    )
        except Exception as e:
            QMessageBox.critical(
                self,
                "CSV export error",
                f"Failed to export CSV:\n{e}",
            )

    # -------------------------- PSF fitting --------------------------
    def fit_psf(self):
        """Extract ROI around selected bead and fit 3D Gaussian PSF."""
        if self.stack is None or self.selected_bead is None:
            return

        z0, y0, x0 = self.selected_bead
        roi_xy = self.roi_xy_spin.value()
        roi_z = self.roi_z_spin.value()
        
        # Extract ROI
        roi, roi_bounds = self._extract_roi(z0, y0, x0, roi_xy, roi_z)
        if roi is None:
            QMessageBox.warning(
                self,
                "ROI Error",
                "Selected bead is too close to the stack boundary for the requested ROI size."
            )
            return

        # Fit 3D Gaussian
        try:
            fit_result = self._fit_3d_gaussian(roi)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Fit Error",
                f"3D Gaussian fit failed:\n{e}"
            )
            return

        # Convert pixel-based fit results to physical units
        pixel_nm = self.pixel_size_spin.value()
        z_step_nm = self.z_step_spin.value()
        
        params = fit_result["params"]
        z_c, y_c, x_c = params[0], params[1], params[2]
        sigma_z, sigma_y, sigma_x = params[3], params[4], params[5]
        amplitude, offset = params[6], params[7]
        
        # FWHM = 2.355 * sigma (for Gaussian)
        fwhm_x_nm = 2.355 * sigma_x * pixel_nm
        fwhm_y_nm = 2.355 * sigma_y * pixel_nm
        fwhm_z_nm = 2.355 * sigma_z * z_step_nm
        
        # Axial ratio
        sigma_xy = (sigma_x + sigma_y) / 2.0
        sigma_z_nm = sigma_z * z_step_nm
        sigma_xy_nm = sigma_xy * pixel_nm
        axial_ratio = sigma_z_nm / sigma_xy_nm if sigma_xy_nm > 0 else float('nan')
        
        # Absolute fitted center in stack coordinates
        if roi_bounds is not None:
            z_min, z_max, y_min, y_max, x_min, x_max = roi_bounds
            abs_z = z_min + z_c
            abs_y = y_min + y_c
            abs_x = x_min + x_c
        else:
            abs_z, abs_y, abs_x = z0, y0, x0
        
        # Update fit circle overlay in viewer (lateral FWHM)
        # FWHM_xy in pixels
        fwhm_xy_px = 2.355 * sigma_xy
        radius_px = fwhm_xy_px / 2.0
        if self.fit_roi is not None:
            self.image_view.getView().removeItem(self.fit_roi)
            self.fit_roi = None
        try:
            self.fit_roi = pg.CircleROI(
                [abs_x - radius_px, abs_y - radius_px],
                [2 * radius_px, 2 * radius_px],
                pen=pg.mkPen("y", width=2),
            )
            # Keep overlay above the image
            self.fit_roi.setZValue(10)
            self.image_view.getView().addItem(self.fit_roi)
        except Exception:
            # If CircleROI is unavailable for some reason, ignore overlay
            self.fit_roi = None
        
        # Format results
        results = f"""
=== PSF Fit Results ===

Bead position: x={x0}, y={y0}, z={z0} (pixels)
ROI size: {roi_xy}×{roi_xy}×{roi_z} pixels (xy×z)

--- Fitted Parameters (pixels) ---
Center: x={x_c:.2f}, y={y_c:.2f}, z={z_c:.2f}
Sigma:  σ_x={sigma_x:.2f}, σ_y={sigma_y:.2f}, σ_z={sigma_z:.2f}
Amplitude: {amplitude:.1f}
Offset: {offset:.1f}

--- Physical Units (nm) ---
FWHM_x: {fwhm_x_nm:.1f} nm
FWHM_y: {fwhm_y_nm:.1f} nm
FWHM_z: {fwhm_z_nm:.1f} nm
FWHM_xy (avg): {(fwhm_x_nm + fwhm_y_nm)/2:.1f} nm

σ_xy (avg): {sigma_xy_nm:.1f} nm
σ_z: {sigma_z_nm:.1f} nm
Axial ratio (σ_z/σ_xy): {axial_ratio:.2f}

--- Fit Quality ---
Residual norm: {fit_result['cost']:.2e}
Success: {fit_result['success']}
"""
        self.results_text.setPlainText(results)

        # ---------------- Profiles: data vs fit ----------------
        nz, ny, nx = roi.shape
        z_c_idx = int(round(z_c))
        y_c_idx = int(round(y_c))
        x_c_idx = int(round(x_c))

        # Clamp indices to ROI bounds
        z_c_idx = max(0, min(nz - 1, z_c_idx))
        y_c_idx = max(0, min(ny - 1, y_c_idx))
        x_c_idx = max(0, min(nx - 1, x_c_idx))

        # x-profile at (z=z_c_idx, y=y_c_idx)
        x_axis = np.arange(nx)
        data_x = roi[z_c_idx, y_c_idx, :]
        model_x = amplitude * np.exp(-0.5 * ((x_axis - x_c) / sigma_x) ** 2) + offset

        self.plot_x.clear()
        self.plot_x.plot(x_axis, data_x, pen=pg.mkPen('w'), symbol='o', symbolSize=4, name='data')
        self.plot_x.plot(x_axis, model_x, pen=pg.mkPen('r', width=2), name='fit')

        # y-profile at (z=z_c_idx, x=x_c_idx)
        y_axis = np.arange(ny)
        data_y = roi[z_c_idx, :, x_c_idx]
        model_y = amplitude * np.exp(-0.5 * ((y_axis - y_c) / sigma_y) ** 2) + offset

        self.plot_y.clear()
        self.plot_y.plot(y_axis, data_y, pen=pg.mkPen('w'), symbol='o', symbolSize=4, name='data')
        self.plot_y.plot(y_axis, model_y, pen=pg.mkPen('r', width=2), name='fit')

        # z-profile through center (y=y_c_idx, x=x_c_idx)
        z_axis = np.arange(nz)
        data_z = roi[:, y_c_idx, x_c_idx]
        model_z = amplitude * np.exp(-0.5 * ((z_axis - z_c) / sigma_z) ** 2) + offset

        self.plot_z.clear()
        self.plot_z.plot(z_axis, data_z, pen=pg.mkPen('w'), symbol='o', symbolSize=4, name='data')
        self.plot_z.plot(z_axis, model_z, pen=pg.mkPen('r', width=2), name='fit')

        # Show textual results with maximum width
        if hasattr(self, 'tabs'):
            # Tab index 1 is "Fit Results"
            self.tabs.setCurrentIndex(1)

    def _extract_roi(self, z0, y0, x0, roi_xy, roi_z):
        """Extract a 3D ROI centered at (z0, y0, x0).

        Parameters
        ----------
        z0, y0, x0 : int
            Center position in stack coordinates.
        roi_xy : int
            ROI half-size in x and y (total size = 2*roi_xy+1).
        roi_z : int
            ROI half-size in z (total size = 2*roi_z+1).

        Returns
        -------
        roi : ndarray or None
            3D ROI array, or None if out of bounds.
        bounds : tuple or None
            (z_min, z_max, y_min, y_max, x_min, x_max) in stack coordinates.
        """
        nz, ny, nx = self.stack.shape
        half_xy = roi_xy // 2
        half_z = roi_z // 2
        
        z_min = z0 - half_z
        z_max = z0 + half_z + 1
        y_min = y0 - half_xy
        y_max = y0 + half_xy + 1
        x_min = x0 - half_xy
        x_max = x0 + half_xy + 1
        
        # Check bounds
        if z_min < 0 or z_max > nz or y_min < 0 or y_max > ny or x_min < 0 or x_max > nx:
            return None, None
        
        roi = self.stack[z_min:z_max, y_min:y_max, x_min:x_max].copy()
        return roi, (z_min, z_max, y_min, y_max, x_min, x_max)

    def _fit_3d_gaussian(self, roi):
        """Fit a 3D Gaussian to the given ROI.

        Parameters
        ----------
        roi : ndarray
            3D ROI array (z, y, x).

        Returns
        -------
        result : dict
            Fit result dictionary with keys:
            - 'params': fitted parameters [z_c, y_c, x_c, sigma_z, sigma_y, sigma_x, amplitude, offset]
            - 'success': bool
            - 'cost': residual norm
        """
        nz, ny, nx = roi.shape
        
        # Initial parameter estimates
        offset_init = np.min(roi)
        amplitude_init = np.max(roi) - offset_init
        
        # Find brightest pixel as initial center
        idx_max = np.unravel_index(np.argmax(roi), roi.shape)
        z_c_init, y_c_init, x_c_init = idx_max
        
        # Initial sigma estimates (rough: 1/5 of ROI size)
        sigma_z_init = nz / 5.0
        sigma_y_init = ny / 5.0
        sigma_x_init = nx / 5.0
        
        p0 = [z_c_init, y_c_init, x_c_init, sigma_z_init, sigma_y_init, sigma_x_init, amplitude_init, offset_init]
        
        # Create coordinate grids
        z_grid, y_grid, x_grid = np.meshgrid(
            np.arange(nz),
            np.arange(ny),
            np.arange(nx),
            indexing='ij'
        )
        coords = np.stack([z_grid.ravel(), y_grid.ravel(), x_grid.ravel()], axis=1)
        data_flat = roi.ravel()
        
        # Define residual function
        def residuals(params):
            model = gaussian_3d(coords, params)
            return model - data_flat
        
        # Fit with bounds to keep parameters positive/reasonable
        bounds_lower = [0, 0, 0, 0.5, 0.5, 0.5, 0, -np.inf]
        bounds_upper = [nz, ny, nx, nz, ny, nx, np.inf, np.inf]
        
        result = least_squares(
            residuals,
            p0,
            bounds=(bounds_lower, bounds_upper),
            max_nfev=500
        )
        
        return {
            'params': result.x,
            'success': result.success,
            'cost': result.cost
        }


def gaussian_3d(coords, params):
    """3D Gaussian function.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Coordinate array with columns [z, y, x].
    params : list or ndarray
        Parameters [z_c, y_c, x_c, sigma_z, sigma_y, sigma_x, amplitude, offset].

    Returns
    -------
    values : ndarray, shape (N,)
        Gaussian values at each coordinate.
    """
    z_c, y_c, x_c, sigma_z, sigma_y, sigma_x, amplitude, offset = params
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    
    exponent = -0.5 * (
        ((z - z_c) / sigma_z)**2 +
        ((y - y_c) / sigma_y)**2 +
        ((x - x_c) / sigma_x)**2
    )
    return amplitude * np.exp(exponent) + offset


if __name__ == "plugin":
    window = PSFDeterminationWidget()
    window.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = PSFDeterminationWidget()
    win.show()
    sys.exit(app.exec_())

# ---------------------------------------------------------------------------
# New-style manifest loading
# ---------------------------------------------------------------------------
from pathlib import Path as _Path
import json as _json

_manifest_path = _Path(__file__).parent / "manifest.json"
if _manifest_path.exists():
    _manifest = _json.loads(_manifest_path.read_text())
    name = _manifest.get("display_name", name)

cli_entrypoint = "psf-determination=chisurf.plugins.microscopy.psf_determination.cli:cli"


def __getattr__(attr_name: str):
    """Lazy Qt gate for new-style entrypoints."""
    if attr_name == "PsfDeterminationTool":
        from .gui.tool import PsfDeterminationTool as _cls
        globals()["PsfDeterminationTool"] = _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {attr_name!r}")


__all__ = ["PSFDeterminationWidget", "PsfDeterminationTool", "gaussian_3d"]
