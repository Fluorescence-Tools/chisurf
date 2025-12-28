from __future__ import annotations

from typing import Optional

import numpy as np

from chisurf.gui import QtWidgets, QtCore

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - pyqtgraph may not be available in some environments
    pg = None  # type: ignore


def setup_kappa2_controls(
    owner: QtWidgets.QWidget,
    parent_layout: QtWidgets.QLayout,
) -> None:
    """Attach shared κ² mode controls to *owner* using *parent_layout*.

    This creates Dynamic/Static radio buttons and two tool buttons
    (distribution plot and experimental κ²), wires them to the shared
    helper functions, and stores references on *owner* for later use.
    """

    mode_layout = QtWidgets.QHBoxLayout()
    dyn_radio = QtWidgets.QRadioButton("dynamic κ²")
    stat_radio = QtWidgets.QRadioButton("static κ²")
    mode_layout.addWidget(dyn_radio)
    mode_layout.addWidget(stat_radio)

    mode_layout.addStretch(1)
    
    # Fast optimization checkbox for static kappa2 convolution
    fast_checkbox = QtWidgets.QCheckBox("Fast")
    fast_checkbox.setChecked(True)
    fast_checkbox.setToolTip(
        "Use fast loop-based convolution for static κ² distributions.\n"
        "Provides speedup for large κ² distributions by looping over\n"
        "ratio values and accumulating shifted distance histograms.\n"
        "Disable for exact outer product."
    )
    mode_layout.addWidget(fast_checkbox)

    show_btn = QtWidgets.QToolButton()
    show_btn.setText("show κ²")
    mode_layout.addWidget(show_btn)

    exp_btn = QtWidgets.QToolButton()
    exp_btn.setText("compute κ²")
    mode_layout.addWidget(exp_btn)

    parent_layout.addLayout(mode_layout)

    group = QtWidgets.QButtonGroup(owner)
    group.addButton(dyn_radio)
    group.addButton(stat_radio)
    group.setExclusive(True)

    orientation_param = getattr(owner, "orientation_parameter", None)
    current_mode = getattr(orientation_param, "mode", "fast") if orientation_param is not None else "fast"
    if current_mode == "slow":
        stat_radio.setChecked(True)
    else:
        dyn_radio.setChecked(True)

    def on_mode_changed() -> None:
        op = getattr(owner, "orientation_parameter", None)
        if op is None:
            return
        op.mode = "slow" if stat_radio.isChecked() else "fast"
        try:
            owner.update()
        except Exception:
            pass

    dyn_radio.toggled.connect(on_mode_changed)
    stat_radio.toggled.connect(on_mode_changed)
    
    def on_fft_changed() -> None:
        """Trigger model update when FFT checkbox state changes."""
        try:
            owner.update()
        except Exception:
            pass
    
    fast_checkbox.stateChanged.connect(on_fft_changed)

    def on_show_combined() -> None:
        show_rapp_rda_distribution_plot(
            parent=owner,
            orientation_parameter=getattr(owner, "orientation_parameter", None),
            fret_model=owner,
        )

    def on_open_experimental() -> None:
        open_experimental_k2_dialog(parent=owner, fret_model=owner)

    show_btn.clicked.connect(on_show_combined)
    exp_btn.clicked.connect(on_open_experimental)

    # Expose on owner for potential introspection / testing
    owner._kappa2_dynamic_radio = dyn_radio
    owner._kappa2_static_radio = stat_radio
    owner._kappa2_fft_checkbox = fast_checkbox
    owner._kappa2_show_distribution_button = show_btn
    owner._kappa2_experimental_button = exp_btn
    owner._kappa2_mode_group = group


def _extract_kappa2_pairs(orientation_parameter, fret_parameters):
    """Return list of (amplitude, k2) pairs from model or scalar k2.

    If orientation mode is static (slow_isotropic) and an orientation_spectrum
    is available, use that. Otherwise fall back to a single peak at the scalar k2.
    """
    pairs = []

    if orientation_parameter is not None:
        mode = getattr(orientation_parameter, "mode", "fast")
    else:
        mode = getattr(fret_parameters, "mode", "fast") if fret_parameters is not None else "fast"

    if mode == "slow" and orientation_parameter is not None:
        spec = getattr(orientation_parameter, "orientation_spectrum", None)
        if spec is not None:
            arr = np.asarray(spec, dtype=float).ravel()
            if arr.size >= 2 and arr.size % 2 == 0:
                amps = arr[0::2]
                vals = arr[1::2]
                pairs.extend(zip(amps, vals))

    # Fallback: single scalar k2 from fret_parameters
    if not pairs and fret_parameters is not None:
        try:
            k2_scalar = float(getattr(fret_parameters, "kappa2", 0.666))
        except Exception:
            k2_scalar = 0.666
        pairs.append((1.0, k2_scalar))

    return pairs, mode


def show_kappa2_distribution_plot(
    parent: Optional[QtWidgets.QWidget] = None,
    orientation_parameter=None,
    fret_parameters=None,
) -> None:
    """Show a simple dialog with the current orientation factor distribution.

    This works both for the static distribution (slow) and for the
    dynamic scalar case (fast, shown as a single spike at k2).
    """
    if pg is None:
        QtWidgets.QMessageBox.warning(
            parent,
            "Orientation factor distribution",
            "pyqtgraph is not available, cannot plot k2 distribution.",
        )
        return

    pairs, mode = _extract_kappa2_pairs(orientation_parameter, fret_parameters)

    static_pairs = []
    if orientation_parameter is not None:
        spec = getattr(orientation_parameter, "_k2_slow_iso", None)
        if spec is None:
            spec = getattr(orientation_parameter, "orientation_spectrum", None)
        if spec is not None:
            arr = np.asarray(spec, dtype=float).ravel()
            if arr.size >= 2 and arr.size % 2 == 0:
                static_pairs = list(zip(arr[0::2], arr[1::2]))

    if not static_pairs:
        static_pairs = list(pairs)

    if not static_pairs:
        QtWidgets.QMessageBox.information(
            parent,
            "Orientation factor distribution",
            "No orientation factor distribution is available.",
        )
        return

    try:
        dynamic_k2 = float(getattr(fret_parameters, "kappa2", 0.666)) if fret_parameters is not None else 0.666
    except Exception:
        dynamic_k2 = 0.666

    amps = np.asarray([a for (a, _) in static_pairs], dtype=float)
    k2_vals = np.asarray([k for (_, k) in static_pairs], dtype=float)
    finite = np.isfinite(amps) & np.isfinite(k2_vals)
    amps = amps[finite]
    k2_vals = k2_vals[finite]
    if amps.size == 0:
        QtWidgets.QMessageBox.information(
            parent,
            "Orientation factor distribution",
            "No valid orientation factor distribution is available.",
        )
        return

    total = float(np.sum(amps))
    if total > 0:
        amps = amps / total
    static_mean = float(np.sum(amps * k2_vals))

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Orientation factor distribution (k2)")
    layout = QtWidgets.QVBoxLayout(dialog)

    mode_label = "slow" if mode == "slow" else "fast"
    info_label = QtWidgets.QLabel(
        f"Current orientation mode: {mode_label} | "
        f"dynamic k2={dynamic_k2:.3f} | static avg={static_mean:.3f}"
    )
    layout.addWidget(info_label)

    # Sort by k2 value to have a nice monotonic x-axis
    order = np.argsort(k2_vals)
    k2_vals_sorted = k2_vals[order]
    weights_sorted = amps[order]

    plot_widget = pg.PlotWidget(dialog)
    plot_widget.showGrid(x=True, y=True, alpha=0.3)
    plot_widget.setLabel("bottom", "k2")
    plot_widget.setLabel("left", "Weight")
    plot_widget.addLegend()
    plot_widget.plot(
        k2_vals_sorted,
        weights_sorted,
        pen=pg.mkPen(width=2),
        symbol="o",
        symbolSize=6,
        name="static distribution",
    )

    y_max = float(np.max(weights_sorted)) if weights_sorted.size else 1.0
    y_line = y_max * 1.05
    plot_widget.plot(
        [dynamic_k2, dynamic_k2],
        [0.0, y_line],
        pen=pg.mkPen(color=(80, 160, 255), width=2),
        name="dynamic (k2)",
    )
    plot_widget.plot(
        [static_mean, static_mean],
        [0.0, y_line],
        pen=pg.mkPen(color=(255, 160, 80), width=2, style=QtCore.Qt.DashLine),
        name="static avg",
    )

    try:
        dyn_label = pg.TextItem("dynamic", color=(80, 160, 255), anchor=(0.5, 1.0))
        dyn_label.setPos(dynamic_k2, y_line)
        plot_widget.addItem(dyn_label)
    except Exception:
        pass

    try:
        stat_label = pg.TextItem("static avg", color=(255, 160, 80), anchor=(0.5, 1.0))
        stat_label.setPos(static_mean, y_line)
        plot_widget.addItem(stat_label)
    except Exception:
        pass

    layout.addWidget(plot_widget)

    button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
    button_box.accepted.connect(dialog.accept)
    layout.addWidget(button_box)

    dialog.resize(500, 400)
    dialog.exec_()


def show_rapp_rda_distribution_plot(
    parent: Optional[QtWidgets.QWidget] = None,
    orientation_parameter=None,
    fret_model=None,
) -> None:
    """Show the κ² distribution transformed to R_app/R_DA ratio scale.
    
    This displays the distribution used for FFT convolution:
    R_app/R_DA = (κ²/⟨κ²⟩)^(1/6)
    """
    if pg is None:
        QtWidgets.QMessageBox.warning(
            parent,
            "R_app/R_DA distribution",
            "pyqtgraph is not available, cannot plot distribution.",
        )
        return

    # Get fret_parameters from fret_model if available
    fret_parameters = getattr(fret_model, "fret_parameters", None) if fret_model is not None else None
    pairs, mode = _extract_kappa2_pairs(orientation_parameter, fret_parameters)

    static_pairs = []
    if orientation_parameter is not None:
        spec = getattr(orientation_parameter, "_k2_slow_iso", None)
        if spec is None:
            spec = getattr(orientation_parameter, "orientation_spectrum", None)
        if spec is not None:
            arr = np.asarray(spec, dtype=float).ravel()
            if arr.size >= 2 and arr.size % 2 == 0:
                static_pairs = list(zip(arr[0::2], arr[1::2]))

    if not static_pairs:
        static_pairs = list(pairs)

    if not static_pairs:
        QtWidgets.QMessageBox.information(
            parent,
            "R_app/R_DA distribution",
            "No orientation factor distribution is available.",
        )
        return

    amps = np.asarray([a for (a, _) in static_pairs], dtype=float)
    k2_vals = np.asarray([k for (_, k) in static_pairs], dtype=float)
    finite = np.isfinite(amps) & np.isfinite(k2_vals)
    amps = amps[finite]
    k2_vals = k2_vals[finite]
    
    if amps.size == 0:
        QtWidgets.QMessageBox.information(
            parent,
            "R_app/R_DA distribution",
            "No valid orientation factor distribution is available.",
        )
        return

    # Use shared transformation function from chisurf.fluorescence.general
    from chisurf.fluorescence.general import kappa2_to_distance_ratio
    r_ratio, weights, k2_mean = kappa2_to_distance_ratio(amps, k2_vals, n_bins=256)
    
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("κ² & Rₐₚₚ/Rᴅᴀ distributions")
    layout = QtWidgets.QVBoxLayout(dialog)

    info_label = QtWidgets.QLabel(
        f"Transformation: Rₐₚₚ/Rᴅᴀ = (⟨κ²⟩/κ²)^(1/6) | ⟨κ²⟩ = {k2_mean:.3f}"
    )
    layout.addWidget(info_label)

    # First plot: κ² distribution
    plot_k2 = pg.PlotWidget(dialog)
    plot_k2.showGrid(x=True, y=True, alpha=0.3)
    plot_k2.setLabel("bottom", "κ²")
    plot_k2.setLabel("left", "Probability")
    plot_k2.addLegend()
    plot_k2.plot(
        k2_vals,
        amps / np.sum(amps) if np.sum(amps) > 0 else amps,
        pen=pg.mkPen(width=2, color=(255, 100, 100)),
        symbol='o',
        symbolSize=5,
        name="ρ(κ²)",
    )
    # Mark mean κ²
    k2_max_y = float(np.max(amps / np.sum(amps))) if np.sum(amps) > 0 else 1.0
    plot_k2.plot(
        [k2_mean, k2_mean],
        [0.0, k2_max_y * 1.05],
        pen=pg.mkPen(color=(255, 160, 80), width=2, style=QtCore.Qt.DashLine),
        name=f"⟨κ²⟩ = {k2_mean:.3f}",
    )
    layout.addWidget(plot_k2)

    # Second plot: R_app/R_DA ratio distribution
    plot_widget = pg.PlotWidget(dialog)
    plot_widget.showGrid(x=True, y=True, alpha=0.3)
    plot_widget.setLabel("bottom", "Rₐₚₚ/Rᴅᴀ")
    plot_widget.setLabel("left", "Probability Density")
    plot_widget.addLegend()
    plot_widget.plot(
        r_ratio,
        weights,
        pen=pg.mkPen(width=2, color=(80, 160, 255)),
        name="ρ(Rₐₚₚ/Rᴅᴀ)",
    )

    # Mark the mean ratio (should be 1.0 by definition)
    y_max = float(np.max(weights)) if weights.size else 1.0
    y_line = y_max * 1.05
    plot_widget.plot(
        [1.0, 1.0],
        [0.0, y_line],
        pen=pg.mkPen(color=(255, 160, 80), width=2, style=QtCore.Qt.DashLine),
        name="mean (1.0)",
    )

    try:
        mean_label = pg.TextItem("(2/3)^(1/6)", color=(255, 160, 80), anchor=(0.5, 1.0))
        mean_label.setPos(1.0, y_line)
        plot_widget.addItem(mean_label)
    except Exception:
        pass

    layout.addWidget(plot_widget)
    
    # Second plot: R_DA and R_app distributions after convolution
    # Get distance distribution from fret_model
    distance_dist = None
    if fret_model is not None:
        # Try to get distance distribution directly from the model
        try:
            distance_dist = getattr(fret_model, "distance_distribution", None)
        except Exception:
            pass
    
    if distance_dist is not None:
        try:
            # Extract R_DA distribution
            dist_array = np.asarray(distance_dist, dtype=float)
            if dist_array.ndim == 3 and dist_array.shape[0] > 0:
                # Shape is (n_dist, 2, n_points) - extract first distribution
                amp_r = dist_array[0, 0, :]
                r_da = dist_array[0, 1, :]
                
                # Filter valid points
                valid = (amp_r > 0) & np.isfinite(r_da) & (r_da > 0)
                amp_r = amp_r[valid]
                r_da = r_da[valid]
                
                if len(r_da) > 0:
                    # Get Fast setting from UI checkbox
                    use_fast = True
                    if fret_model is not None:
                        fast_checkbox = getattr(fret_model, "_kappa2_fft_checkbox", None)
                        if fast_checkbox is not None:
                            use_fast = fast_checkbox.isChecked()
                    
                    # Compute R_app distribution via convolution using shared function
                    from chisurf.fluorescence.general import convolve_distance_with_k2_ratio
                    r_app_centers, r_app_hist = convolve_distance_with_k2_ratio(
                        r_da, amp_r, r_ratio, weights, n_bins=256, use_fast=use_fast
                    )
                    
                    # Normalize both distributions to max=1.0 for comparison
                    max_amp_r = np.max(amp_r)
                    max_r_app = np.max(r_app_hist)
                    
                    if max_amp_r > 0:
                        amp_r_norm = amp_r / max_amp_r
                    else:
                        amp_r_norm = amp_r
                    
                    if max_r_app > 0:
                        r_app_hist_norm = r_app_hist / max_r_app
                    else:
                        r_app_hist_norm = r_app_hist
                    
                    # Create second plot
                    plot_widget2 = pg.PlotWidget(dialog)
                    plot_widget2.showGrid(x=True, y=True, alpha=0.3)
                    plot_widget2.setLabel("bottom", "Distance (Å)")
                    plot_widget2.setLabel("left", "Normalized Intensity")
                    plot_widget2.addLegend()
                    
                    # Plot R_DA
                    plot_widget2.plot(
                        r_da,
                        amp_r_norm,
                        pen=pg.mkPen(width=2, color=(80, 255, 160)),
                        symbol='o',
                        symbolSize=4,
                        name="ρ(Rᴅᴀ)",
                    )
                    
                    # Plot R_app
                    plot_widget2.plot(
                        r_app_centers,
                        r_app_hist_norm,
                        pen=pg.mkPen(width=2, color=(255, 80, 160)),
                        name="ρ(Rₐₚₚ)",
                    )
                    
                    layout.addWidget(plot_widget2)
        except Exception as e:
            # Silently skip if distance distribution cannot be processed
            pass

    button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
    button_box.accepted.connect(dialog.accept)
    layout.addWidget(button_box)

    dialog.resize(500, 400)
    dialog.exec_()


def open_experimental_k2_dialog(
    parent: Optional[QtWidgets.QWidget] = None,
    fret_model=None,
) -> None:
    """Open the Kappa2Dist plugin dialog and apply its distribution to a FRET model.

    Fast mode: use mean k2 from the distribution as scalar kappa2.
    Slow mode: use full k2 distribution as orientation spectrum.
    """
    try:
        from chisurf.plugins.kappa2_dist import Kappa2Dist
    except Exception:
        QtWidgets.QMessageBox.critical(
            parent,
            "Experimental k2",
            "Kappa2 Distribution plugin is not available.",
        )
        return

    if fret_model is None:
        QtWidgets.QMessageBox.information(
            parent,
            "Experimental k2",
            "No FRET model is available to apply the kappa2 distribution.",
        )
        return

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Experimental k2 (orientation factor distribution)")
    layout = QtWidgets.QVBoxLayout(dialog)

    # Initialize plugin widget with current scalar k2 as starting value
    try:
        k2_start = float(getattr(fret_model.fret_parameters, "kappa2", 0.666))
    except Exception:
        k2_start = 0.666

    plugin_widget = Kappa2Dist(kappa2=k2_start, parent=dialog)
    layout.addWidget(plugin_widget)

    # Ensure the embedded plugin widget is visible and has a valid default model
    try:
        # If neither model radio button is checked, select a sensible default ("cone")
        rb_cone = getattr(plugin_widget, "radioButton_2", None)
        rb_diff = getattr(plugin_widget, "radioButton", None)
        if rb_cone is not None and rb_diff is not None:
            if not rb_cone.isChecked() and not rb_diff.isChecked():
                rb_cone.setChecked(True)

        # Compute an initial histogram so the user sees a distribution immediately
        plugin_widget.onUpdateHist()
    except Exception:
        # Best-effort only; the user can still interact with the plugin and press its compute button
        pass

    # The plugin hides itself in its constructor; make sure it is shown inside our dialog
    try:
        plugin_widget.show()
    except Exception:
        pass

    button_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    layout.addWidget(button_box)
    button_box.accepted.connect(dialog.accept)
    button_box.rejected.connect(dialog.reject)

    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return

    # Extract distribution and mean from plugin
    try:
        k2scale_raw = np.asarray(getattr(plugin_widget, "k2scale", []), dtype=float)
        k2hist = np.asarray(getattr(plugin_widget, "k2hist", []), dtype=float)
        k2_mean = float(getattr(plugin_widget, "k2_mean", np.nan))
    except Exception:
        k2scale_raw = np.array([], dtype=float)
        k2hist = np.array([], dtype=float)
        k2_mean = np.nan

    if k2hist.size == 0:
        QtWidgets.QMessageBox.warning(
            parent,
            "Experimental k2",
            "No kappa2 distribution computed in plugin.",
        )
        return

    if k2scale_raw.size == k2hist.size:
        k2scale = k2scale_raw
    elif k2scale_raw.size == k2hist.size + 1:
        k2scale = k2scale_raw[1:]
    else:
        QtWidgets.QMessageBox.warning(
            parent,
            "Experimental k2",
            "No valid kappa2 distribution computed in plugin.",
        )
        return

    # Normalize amplitudes
    total = float(np.sum(k2hist))
    if total <= 0:
        QtWidgets.QMessageBox.warning(
            parent,
            "Experimental k2",
            "Computed kappa2 distribution has zero total weight.",
        )
        return

    k2hist_norm = k2hist / total

    orientation_param = getattr(fret_model, "orientation_parameter", None)

    # Always build full orientation spectrum as interleaved (amp, k2, ...)
    interleaved = np.empty(k2scale.size * 2, dtype=float)
    interleaved[0::2] = k2hist_norm
    interleaved[1::2] = k2scale

    # Try to store on orientation_parameter (used for static averaging)
    if orientation_param is not None:
        try:
            orientation_param.orientation_spectrum = interleaved
        except Exception:
            # Only warn if static averaging is actually requested; in dynamic
            # mode the scalar kappa2 is sufficient and will still be updated.
            orientation_mode = getattr(orientation_param, "mode", "fast")
            if orientation_mode == "slow":
                QtWidgets.QMessageBox.warning(
                    parent,
                    "Experimental k2",
                    "Failed to apply kappa2 distribution to orientation parameter.",
                )

    # Always update scalar kappa2 with the mean of the distribution (for dynamic averaging)
    if not np.isfinite(k2_mean):
        k2_mean = float(np.sum(k2scale * k2hist_norm))
    try:
        fret_model.fret_parameters.kappa2 = k2_mean

        # Also update the associated FittingParameterWidget so the GUI reflects
        # the new mean k2 value in the fit controller.
        try:
            param = getattr(fret_model.fret_parameters, "_kappa2", None)
            controller = getattr(param, "controller", None) if param is not None else None
            widget_value = getattr(controller, "widget_value", None) if controller is not None else None
            if widget_value is not None:
                try:
                    widget_value.blockSignals(True)
                    widget_value.setValue(k2_mean)
                finally:
                    widget_value.blockSignals(False)
        except Exception:
            # UI refresh is best-effort; model value is already updated
            pass
    except Exception:
        QtWidgets.QMessageBox.warning(
            parent,
            "Experimental k2",
            "Failed to apply mean kappa2 to FRET parameters.",
        )
        return

    # Trigger model update if available
    update = getattr(fret_model, "update", None)
    if callable(update):
        try:
            update()
        except Exception:
            # Silent failure; model update is best-effort here
            pass
