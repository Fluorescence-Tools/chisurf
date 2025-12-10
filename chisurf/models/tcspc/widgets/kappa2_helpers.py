from __future__ import annotations

from typing import Optional

import numpy as np

from chisurf.gui import QtWidgets

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
    current_mode = getattr(orientation_param, "mode", "fast_isotropic") if orientation_param is not None else "fast_isotropic"
    if current_mode == "slow_isotropic":
        stat_radio.setChecked(True)
    else:
        dyn_radio.setChecked(True)

    def on_mode_changed() -> None:
        op = getattr(owner, "orientation_parameter", None)
        if op is None:
            return
        op.mode = "slow_isotropic" if stat_radio.isChecked() else "fast_isotropic"
        try:
            owner.update()
        except Exception:
            pass

    dyn_radio.toggled.connect(on_mode_changed)
    stat_radio.toggled.connect(on_mode_changed)

    def on_show_distribution() -> None:
        show_kappa2_distribution_plot(
            parent=owner,
            orientation_parameter=getattr(owner, "orientation_parameter", None),
            fret_parameters=getattr(owner, "fret_parameters", None),
        )

    def on_open_experimental() -> None:
        open_experimental_k2_dialog(parent=owner, fret_model=owner)

    show_btn.clicked.connect(on_show_distribution)
    exp_btn.clicked.connect(on_open_experimental)

    # Expose on owner for potential introspection / testing
    owner._kappa2_dynamic_radio = dyn_radio
    owner._kappa2_static_radio = stat_radio
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
        mode = getattr(orientation_parameter, "mode", "fast_isotropic")
    else:
        mode = getattr(fret_parameters, "mode", "fast_isotropic") if fret_parameters is not None else "fast_isotropic"

    if mode == "slow_isotropic" and orientation_parameter is not None:
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

    This works both for the static distribution (slow_isotropic) and for the
    dynamic scalar case (fast_isotropic, shown as a single spike at k2).
    """
    if pg is None:
        QtWidgets.QMessageBox.warning(
            parent,
            "Orientation factor distribution",
            "pyqtgraph is not available, cannot plot k2 distribution.",
        )
        return

    pairs, mode = _extract_kappa2_pairs(orientation_parameter, fret_parameters)

    if not pairs:
        QtWidgets.QMessageBox.information(
            parent,
            "Orientation factor distribution",
            "No orientation factor distribution is available.",
        )
        return

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Orientation factor distribution (k2)")
    layout = QtWidgets.QVBoxLayout(dialog)

    info_label = QtWidgets.QLabel(f"Current orientation mode: {mode.replace('_', ' ')}")
    layout.addWidget(info_label)

    # Sort by k2 value to have a nice monotonic x-axis
    pairs_sorted = sorted(pairs, key=lambda p: p[1])
    k2_vals = [k2 for (_, k2) in pairs_sorted]
    weights = [amp for (amp, _) in pairs_sorted]

    plot_widget = pg.PlotWidget(dialog)
    plot_widget.showGrid(x=True, y=True, alpha=0.3)
    plot_widget.setLabel("bottom", "k2")
    plot_widget.setLabel("left", "Weight")
    plot_widget.plot(
        k2_vals,
        weights,
        pen=pg.mkPen(width=2),
        symbol="o",
        symbolSize=6,
    )
    layout.addWidget(plot_widget)

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

    Dynamic mode (fast_isotropic): use mean k2 from the distribution as scalar kappa2.
    Static mode (slow_isotropic): use full k2 distribution as orientation spectrum.
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
            orientation_mode = getattr(orientation_param, "mode", "fast_isotropic")
            if orientation_mode == "slow_isotropic":
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
