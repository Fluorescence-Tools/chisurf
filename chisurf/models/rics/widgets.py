from __future__ import annotations

from typing import Any

import numpy as np
from qtpy import QtWidgets, QtGui

import chisurf
import chisurf.plots
from chisurf.models.model import ModelCurve, ModelWidget
from chisurf.fitting.parameter import FittingParameter

from playground.rics_experiment.models import rics_simple, rics_diffusion_triplet


def _get_rics_meta(fit_group) -> dict:
    """Return the RICS metadata dictionary from the current fit.

    This helper accepts either a :class:`FitGroup` (with ``.selected_fit``)
    or a plain :class:`Fit` and normalizes access to ``data.meta_data``.
    """

    fit = getattr(fit_group, 'selected_fit', fit_group)
    data = getattr(fit, 'data', None)
    meta = getattr(data, 'meta_data', {}) or {}
    return meta.get('rics', {}) or {}


def get_rics_residual_image(fit_group, weighted: bool = True):
    """Return a 2D residual image for a RICS fit.

    This accessor mirrors :func:`get_pda_residual_image` but operates on the
    2D RICS maps stored in the data metadata (experimental) and on the
    analytic model's 2D prediction (``rics_model_2d`` attribute).
    """

    # Accept either a FitGroup (with .selected_fit) or a plain Fit
    fit = getattr(fit_group, 'selected_fit', fit_group)

    rics_meta = _get_rics_meta(fit)

    try:
        data_2d = np.asarray(rics_meta.get('ics_mean'), dtype=float)
    except Exception:
        return None, None, None

    model_obj = getattr(fit, 'model', None)
    try:
        model_2d = np.asarray(getattr(model_obj, 'rics_model_2d', None), dtype=float)
    except Exception:
        model_2d = None

    if data_2d is None or model_2d is None:
        return None, None, None
    if data_2d.ndim != 2 or model_2d.ndim != 2:
        return None, None, None

    n0 = min(data_2d.shape[0], model_2d.shape[0])
    n1 = min(data_2d.shape[1], model_2d.shape[1])
    d = data_2d[:n0, :n1]
    m = model_2d[:n0, :n1]

    if weighted:
        sigma = np.sqrt(np.maximum(d, 1.0))
        img = (d - m) / sigma
    else:
        img = d - m

    x = np.arange(n1, dtype=float)
    y = np.arange(n0, dtype=float)
    return img, x, y


def get_rics_data_image(fit_group):
    """Return the experimental 2D RICS map (ICS mean image)."""

    rics_meta = _get_rics_meta(fit_group)
    try:
        data_2d = np.asarray(rics_meta.get('ics_mean'), dtype=float)
    except Exception:
        return None, None, None
    if data_2d is None or data_2d.ndim != 2:
        return None, None, None
    n0, n1 = data_2d.shape
    x = np.arange(n1, dtype=float)
    y = np.arange(n0, dtype=float)
    return data_2d, x, y


def get_rics_model_image(fit_group):
    """Return the analytic RICS model image from the current fit."""

    fit = getattr(fit_group, 'selected_fit', fit_group)
    model_obj = getattr(fit, 'model', None)
    try:
        model_2d = np.asarray(getattr(model_obj, 'rics_model_2d', None), dtype=float)
    except Exception:
        return None, None, None
    if model_2d is None or model_2d.ndim != 2:
        return None, None, None
    n0, n1 = model_2d.shape
    x = np.arange(n1, dtype=float)
    y = np.arange(n0, dtype=float)
    return model_2d, x, y


def get_rics_intensity_image(fit_group, mode: str = 'mean', frame: int | None = None):
    """Return an intensity image derived from the CLSM stack used for RICS.

    Parameters
    ----------
    mode : {"mean", "frame"}
        - ``"mean"``: return the ROI-averaged intensity image over all frames.
        - ``"frame"``: return a single frame from the intensity stack as
          selected by ``frame`` (index clamped to valid range).
    frame : int, optional
        Frame index when ``mode == "frame"``. If *None*, defaults to 0.
    """

    rics_meta = _get_rics_meta(fit_group)

    stack = rics_meta.get('intensity_stack', None)
    mean_img = rics_meta.get('intensity_mean', None)

    try:
        stack_arr = np.asarray(stack, dtype=float) if stack is not None else None
    except Exception:
        stack_arr = None

    if mode == 'mean':
        # Prefer the precomputed mean image; fall back to averaging the stack.
        try:
            img = np.asarray(mean_img, dtype=float)
        except Exception:
            img = None
        if (img is None or img.ndim != 2) and stack_arr is not None and stack_arr.ndim == 3:
            try:
                img = stack_arr.mean(axis=0)
            except Exception:
                img = None
        if img is None or img.ndim != 2:
            return None, None, None
    elif mode == 'frame':
        if stack_arr is None or stack_arr.ndim != 3 or stack_arr.size == 0:
            return None, None, None
        n_frames = stack_arr.shape[0]
        if frame is None:
            frame = 0
        try:
            idx = int(frame)
        except Exception:
            idx = 0
        if idx < 0:
            idx = 0
        if idx >= n_frames:
            idx = n_frames - 1
        img = stack_arr[idx]
        if img is None or img.ndim != 2:
            return None, None, None
    else:
        return None, None, None

    n0, n1 = img.shape
    x = np.arange(n1, dtype=float)
    y = np.arange(n0, dtype=float)
    return img, x, y


def get_rics_number_of_frames(fit_group) -> int:
    """Return the number of frames in the stored intensity stack, if any."""

    rics_meta = _get_rics_meta(fit_group)
    stack = rics_meta.get('intensity_stack', None)
    try:
        stack_arr = np.asarray(stack) if stack is not None else None
    except Exception:
        stack_arr = None
    if stack_arr is not None and stack_arr.ndim == 3:
        try:
            return int(stack_arr.shape[0])
        except Exception:
            return 0
    try:
        return int(rics_meta.get('n_frames', 0))
    except Exception:
        return 0


class RicsSimpleModel(ModelCurve):
    """Simple analytic 2D RICS model using `rics_simple`.

    The model operates on the 2D lag grid (line_shift, pixel_shift) stored in
    the experimental data's ``meta_data['rics']`` dictionary and produces a
    2D ICS prediction. The flattened model curve is stored in ``self.y`` so
    that standard 1D plots and residuals continue to work.
    """

    name = "RICS simple diffusion"

    def __init__(self, fit: chisurf.fitting.fit.Fit, *args: Any, **kwargs: Any) -> None:
        super().__init__(fit, *args, **kwargs)

        # Core physical parameters (mirroring playground.rics_experiment.models)
        self._n = FittingParameter(
            name="n",
            value=1.0,
            lb=1e-6,
            ub=1e9,
            bounds_on=False,
            fixed=False,
            registry_id="rics.n",
        )
        self._D = FittingParameter(
            name="D",
            value=2.0,
            lb=1e-6,
            ub=1e3,
            bounds_on=False,
            fixed=False,
            registry_id="rics.D",
        )  # µm^2/s
        self._offset = FittingParameter(
            name="offset",
            value=0.0,
            lb=-1e3,
            ub=1e3,
            bounds_on=False,
            fixed=False,
            registry_id="rics.offset",
        )
        # Abbreviations: "pixel" -> "pxl", "duration" -> "dur" in the
        # displayed parameter names to keep the RICS table compact.
        self._pixel_duration = FittingParameter(
            name="pxl_dur",
            value=11.1,
            lb=0.01,
            ub=1e3,
            bounds_on=False,
            fixed=True,
            registry_id="rics.pxl_dur",
        )  # µs
        self._line_duration = FittingParameter(
            name="line_dur",
            value=3.33,
            lb=0.001,
            ub=1e4,
            bounds_on=False,
            fixed=True,
            registry_id="rics.line_dur",
        )  # ms
        self._pixel_size = FittingParameter(
            name="pxl_size",
            value=40.0,
            lb=1.0,
            ub=1e4,
            bounds_on=False,
            fixed=True,
            registry_id="rics.pxl_size",
        )  # nm
        self._w_r = FittingParameter(
            name="w_r",
            value=0.2,
            lb=1e-3,
            ub=1e2,
            bounds_on=False,
            fixed=True,
            registry_id="rics.w_r",
        )  # µm
        self._w_z = FittingParameter(
            name="w_z",
            value=1.0,
            lb=1e-3,
            ub=1e3,
            bounds_on=False,
            fixed=True,
            registry_id="rics.w_z",
        )  # µm

        # If the attached RICS dataset already carries timing metadata,
        # initialize the fixed pixel/line duration parameters from it so the
        # parameter table reflects the experimental values.
        try:
            fit_obj = getattr(fit, 'selected_fit', fit)
            data = getattr(fit_obj, 'data', None)
            meta = getattr(data, 'meta_data', {}) or {}
            rics_meta = meta.get('rics', {}) or {}
            pd_meta = rics_meta.get('pixel_duration_us', None)
            ld_meta = rics_meta.get('line_duration_ms', None)
            if isinstance(pd_meta, (int, float)) and pd_meta > 0:
                self._pixel_duration.value = float(pd_meta)
            if isinstance(ld_meta, (int, float)) and ld_meta > 0:
                self._line_duration.value = float(ld_meta)
        except Exception:
            pass

        # Register parameters with the fitting machinery
        try:
            self.find_parameters()
        except Exception:
            pass

        # 2D model cache used by the Residual2DPlot accessor
        self.rics_model_2d: np.ndarray | None = None

    def update_model(self, **kwargs: Any) -> None:  # type: ignore[override]
        # Choose current fit (FitGroup or plain Fit)
        fit = getattr(self.fit, 'selected_fit', self.fit)
        data = getattr(fit, 'data', None)

        meta = getattr(data, 'meta_data', {}) or {}
        rics_meta = meta.get('rics', {}) or {}
        try:
            line_shift = np.asarray(rics_meta.get('line_shift'), dtype=float)
            pixel_shift = np.asarray(rics_meta.get('pixel_shift'), dtype=float)
        except Exception:
            line_shift = None
            pixel_shift = None

        if line_shift is None or pixel_shift is None or line_shift.shape != pixel_shift.shape:
            # Fallback: keep model consistent but trivially zero
            y_model = np.zeros_like(getattr(data, 'y', np.zeros(0, dtype=float)), dtype=float)
            self.rics_model_2d = None
            try:
                self.y = y_model
            except Exception:
                pass
            return

        # Prefer imaging timing from the experiment metadata when available.
        # pixel_duration_us is stored in µs and line_duration_ms in ms; convert
        # to the units expected by the analytic RICS model (µs/ms).
        try:
            pd_meta = rics_meta.get('pixel_duration_us', None)
        except Exception:
            pd_meta = None
        try:
            ld_meta = rics_meta.get('line_duration_ms', None)
        except Exception:
            ld_meta = None

        pixel_duration_val = float(self._pixel_duration.value)
        line_duration_val = float(self._line_duration.value)
        if isinstance(pd_meta, (int, float)) and pd_meta > 0:
            pixel_duration_val = float(pd_meta)
        if isinstance(ld_meta, (int, float)) and ld_meta > 0:
            line_duration_val = float(ld_meta)

        # Keep the underlying fitting parameters synchronized so that the
        # GUI displays the actual values used in the model calculation.
        try:
            self._pixel_duration.value = float(pixel_duration_val)
            self._line_duration.value = float(line_duration_val)
        except Exception:
            pass

        try:
            model_2d = rics_simple(
                line_shift=line_shift,
                pixel_shift=pixel_shift,
                n=float(self._n.value),
                diffusion_coefficient=float(self._D.value),
                offset=float(self._offset.value),
                pixel_duration=pixel_duration_val,
                line_duration=line_duration_val,
                pixel_size=float(self._pixel_size.value),
                w_r=float(self._w_r.value),
                w_z=float(self._w_z.value),
            )
        except Exception:
            model_2d = np.zeros_like(line_shift, dtype=float)

        self.rics_model_2d = np.asarray(model_2d, dtype=float)
        y_model = self.rics_model_2d.ravel()

        # Ensure the 1D model curve matches the experimental x-grid length
        x_data = getattr(data, 'x', None)
        if x_data is not None and getattr(x_data, 'size', 0) == y_model.size:
            try:
                self.x = np.asarray(x_data, dtype=float)
            except Exception:
                pass
        else:
            try:
                self.x = np.arange(y_model.size, dtype=float)
            except Exception:
                pass

        try:
            self.y = y_model
        except Exception:
            pass


class RicsSimpleModelWidget(ModelWidget, RicsSimpleModel):
    """Minimal RICS model widget with 1D + 2D plots.

    - 1D line plot of experimental vs model correlation (flattened).
    - 1D residual plot.
    - 2D residual image using :class:`Residual2DPlot`.
    - Simple parameter table for the RICS model parameters.
    """

    # Attach a plot stack that treats RICS as intrinsically 2D in the GUI.
    # Only 2D residual / image views plus fit diagnostics are exposed; no
    # 1D residual line plot is created for RICS-specific fits.
    try:
        plot_classes = [
            (
                chisurf.plots.Residual2DPlot,
                {
                    'accessor': get_rics_residual_image,
                    'accessor_kwargs': {
                        'weighted': True,
                    },
                    'sources': {
                        'Residual (w)': (get_rics_residual_image, {'weighted': True}),
                        'Residual (raw)': (get_rics_residual_image, {'weighted': False}),
                        'RICS data': (get_rics_data_image, {}),
                        'RICS model': (get_rics_model_image, {}),
                        'Intensity (mean)': (get_rics_intensity_image, {'mode': 'mean'}),
                        'Intensity (frame)': (get_rics_intensity_image, {'mode': 'frame', 'frame': 0}),
                    },
                    'frame_kw': 'frame',
                    'max_frames_accessor': get_rics_number_of_frames,
                },
            ),
            (chisurf.plots.FitInfo, {}),
            (chisurf.plots.ParameterScanPlot, {}),
        ]
    except Exception:
        plot_classes = []

    name = "RICS simple diffusion"

    def __init__(
        self,
        fit: chisurf.fitting.fit.FitGroup,
        icon: QtGui.QIcon | None = None,
        **kwargs: Any,
    ) -> None:
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/fcs.png")
        # Initialize Model + QWidget side via MRO
        super().__init__(fit=fit, icon=icon, **kwargs)

        # Generic parameter table for all RICS parameters
        try:
            from chisurf.gui.widgets.fitting.widgets import make_fitting_parameter_group_widget
        except Exception:  # pragma: no cover - GUI import guard
            make_fitting_parameter_group_widget = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._param_widget = None
        if make_fitting_parameter_group_widget is not None:
            try:
                self._param_widget = make_fitting_parameter_group_widget(self)
                layout.addWidget(self._param_widget)
            except Exception:
                pass

        self.setLayout(layout)
        self.layout = layout

    def update_widgets(self) -> None:  # type: ignore[override]
        # Synchronize parameter table with internal parameter values
        try:
            if self._param_widget is not None and hasattr(self._param_widget, 'finalize'):
                self._param_widget.finalize()
        except Exception:
            pass


class RicsTripletModel(ModelCurve):

    name = "RICS diffusion triplet"

    def __init__(self, fit: chisurf.fitting.fit.Fit, *args: Any, **kwargs: Any) -> None:
        super().__init__(fit, *args, **kwargs)

        self._n = FittingParameter(
            name="n",
            value=1.0,
            lb=1e-6,
            ub=1e9,
            bounds_on=False,
            fixed=False,
            registry_id="rics.n",
        )
        self._D = FittingParameter(
            name="D",
            value=2.0,
            lb=1e-6,
            ub=1e3,
            bounds_on=False,
            fixed=False,
            registry_id="rics.D",
        )
        self._offset = FittingParameter(
            name="offset",
            value=0.0,
            lb=-1e3,
            ub=1e3,
            bounds_on=False,
            fixed=False,
            registry_id="rics.offset",
        )
        self._pixel_duration = FittingParameter(
            name="pxl_dur",
            value=11.1,
            lb=0.01,
            ub=1e3,
            bounds_on=False,
            fixed=True,
            registry_id="rics.pxl_dur",
        )
        self._line_duration = FittingParameter(
            name="line_dur",
            value=3.33,
            lb=0.001,
            ub=1e4,
            bounds_on=False,
            fixed=True,
            registry_id="rics.line_dur",
        )
        self._pixel_size = FittingParameter(
            name="pxl_size",
            value=40.0,
            lb=1.0,
            ub=1e4,
            bounds_on=False,
            fixed=True,
            registry_id="rics.pxl_size",
        )
        self._w_r = FittingParameter(
            name="w_r",
            value=0.2,
            lb=1e-3,
            ub=1e2,
            bounds_on=False,
            fixed=True,
            registry_id="rics.w_r",
        )
        self._w_z = FittingParameter(
            name="w_z",
            value=1.0,
            lb=1e-3,
            ub=1e3,
            bounds_on=False,
            fixed=True,
            registry_id="rics.w_z",
        )
        self._tauT = FittingParameter(
            name="tauT",
            value=0.002,
            lb=1e-6,
            ub=1.0,
            bounds_on=False,
            fixed=False,
            registry_id="rics.tauT",
        )
        self._aT = FittingParameter(
            name="aT",
            value=0.1,
            lb=0.0,
            ub=0.99,
            bounds_on=False,
            fixed=False,
            registry_id="rics.aT",
        )

        # Initialize timing parameters from RICS metadata when available so
        # that the Triplet model shows the correct experimental durations.
        try:
            fit_obj = getattr(fit, 'selected_fit', fit)
            data = getattr(fit_obj, 'data', None)
            meta = getattr(data, 'meta_data', {}) or {}
            rics_meta = meta.get('rics', {}) or {}
            pd_meta = rics_meta.get('pixel_duration_us', None)
            ld_meta = rics_meta.get('line_duration_ms', None)
            if isinstance(pd_meta, (int, float)) and pd_meta > 0:
                self._pixel_duration.value = float(pd_meta)
            if isinstance(ld_meta, (int, float)) and ld_meta > 0:
                self._line_duration.value = float(ld_meta)
        except Exception:
            pass

        try:
            self.find_parameters()
        except Exception:
            pass

        self.rics_model_2d: np.ndarray | None = None

    def update_model(self, **kwargs: Any) -> None:  # type: ignore[override]
        fit = getattr(self.fit, 'selected_fit', self.fit)
        data = getattr(fit, 'data', None)

        meta = getattr(data, 'meta_data', {}) or {}
        rics_meta = meta.get('rics', {}) or {}
        try:
            line_shift = np.asarray(rics_meta.get('line_shift'), dtype=float)
            pixel_shift = np.asarray(rics_meta.get('pixel_shift'), dtype=float)
        except Exception:
            line_shift = None
            pixel_shift = None

        if line_shift is None or pixel_shift is None or line_shift.shape != pixel_shift.shape:
            y_model = np.zeros_like(getattr(data, 'y', np.zeros(0, dtype=float)), dtype=float)
            self.rics_model_2d = None
            try:
                self.y = y_model
            except Exception:
                pass
            return

        # Prefer imaging timing from the experiment metadata when available.
        try:
            pd_meta = rics_meta.get('pixel_duration_us', None)
        except Exception:
            pd_meta = None
        try:
            ld_meta = rics_meta.get('line_duration_ms', None)
        except Exception:
            ld_meta = None

        pixel_duration_val = float(self._pixel_duration.value)
        line_duration_val = float(self._line_duration.value)
        if isinstance(pd_meta, (int, float)) and pd_meta > 0:
            pixel_duration_val = float(pd_meta)
        if isinstance(ld_meta, (int, float)) and ld_meta > 0:
            line_duration_val = float(ld_meta)

        # Keep timing parameters in sync with the values actually used for
        # the model so that the fitting table reflects the experimental
        # pixel/line durations derived from the RICS metadata.
        try:
            self._pixel_duration.value = float(pixel_duration_val)
            self._line_duration.value = float(line_duration_val)
        except Exception:
            pass

        try:
            model_2d = rics_diffusion_triplet(
                line_shift=line_shift,
                pixel_shift=pixel_shift,
                n=float(self._n.value),
                diffusion_coefficient=float(self._D.value),
                offset=float(self._offset.value),
                pixel_duration=pixel_duration_val,
                line_duration=line_duration_val,
                pixel_size=float(self._pixel_size.value),
                w_r=float(self._w_r.value),
                w_z=float(self._w_z.value),
                tauT=float(self._tauT.value),
                aT=float(self._aT.value),
            )
        except Exception:
            model_2d = np.zeros_like(line_shift, dtype=float)

        self.rics_model_2d = np.asarray(model_2d, dtype=float)
        y_model = self.rics_model_2d.ravel()

        x_data = getattr(data, 'x', None)
        if x_data is not None and getattr(x_data, 'size', 0) == y_model.size:
            try:
                self.x = np.asarray(x_data, dtype=float)
            except Exception:
                pass
        else:
            try:
                self.x = np.arange(y_model.size, dtype=float)
            except Exception:
                pass

        try:
            self.y = y_model
        except Exception:
            pass


class RicsTripletModelWidget(ModelWidget, RicsTripletModel):

    try:
        plot_classes = RicsSimpleModelWidget.plot_classes
    except Exception:
        plot_classes = []

    name = "RICS diffusion + triplet"

    def __init__(
        self,
        fit: chisurf.fitting.fit.FitGroup,
        icon: QtGui.QIcon | None = None,
        **kwargs: Any,
    ) -> None:
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/fcs.png")
        super().__init__(fit=fit, icon=icon, **kwargs)

        try:
            from chisurf.gui.widgets.fitting.widgets import make_fitting_parameter_group_widget
        except Exception:
            make_fitting_parameter_group_widget = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._param_widget = None
        if make_fitting_parameter_group_widget is not None:
            try:
                self._param_widget = make_fitting_parameter_group_widget(self)
                layout.addWidget(self._param_widget)
            except Exception:
                pass

        self.setLayout(layout)
        self.layout = layout

    def update_widgets(self) -> None:  # type: ignore[override]
        try:
            if self._param_widget is not None and hasattr(self._param_widget, 'finalize'):
                self._param_widget.finalize()
        except Exception:
            pass
