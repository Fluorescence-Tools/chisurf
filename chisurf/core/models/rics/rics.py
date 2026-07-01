from __future__ import annotations

"""Qt-free RICS fitting models (PRD-38 model/view-spec split).

Raster Image Correlation Spectroscopy models operating on the 2D scan-lag grid
``(line_shift, pixel_shift)`` stored in ``data.meta_data['rics']``. Compute lives
in :mod:`chisurf.core.models.rics.models`; these classes wrap those functions as
pure (Qt-free) ``ModelCurve`` subclasses with :class:`FittingParameterGroup`
parameter groups and a declarative ``*.view.json`` editor.

Models
------
* :class:`RicsSimpleModel` — one 3D-diffusion component.
* :class:`RicsTripletModel` — 3D diffusion + triplet/blinking.
* :class:`RicsImmobileModel` — mobile diffusion + immobile fraction (PAM
  ``RICS_2Comp_Imm``).
* :class:`RicsFlowModel` — 3D diffusion + uniform flow (PAM
  ``2D_Gaussian_Cor_Flow``).
"""

import numpy as np

import chisurf as cs
from chisurf.core.fitting.parameter import FittingParameter, FittingParameterGroup
from chisurf.core.models.model import ModelCurve
from chisurf.core.models.rics.models import (
    ics_gaussian_2d,
    rics_diffusion_triplet,
    rics_flow,
    rics_full,
    rics_immobile,
    rics_simple,
)


def _rics_meta(fit_group) -> tuple[dict, object]:
    """Return ``(rics_meta_dict, data)`` for a fit group or plain fit."""
    fit = getattr(fit_group, "selected_fit", fit_group)
    data = getattr(fit, "data", None)
    meta = getattr(data, "meta_data", {}) or {}
    return (meta.get("rics", {}) or {}), data


# --- image accessors (Qt-free; referenced from view.json) ------------------
def get_rics_residual_image(fit_group, weighted: bool = True):
    """Return the 2D RICS residual image ``(image, x, y)`` (weighted optional)."""
    rics_meta, _ = _rics_meta(fit_group)
    fit = getattr(fit_group, "selected_fit", fit_group)
    try:
        data_2d = np.asarray(rics_meta.get("ics_mean"), dtype=float)
        model_2d = np.asarray(getattr(getattr(fit, "model", None), "rics_model_2d", None), dtype=float)
    except Exception:
        return None, None, None
    if data_2d is None or model_2d is None or data_2d.ndim != 2 or model_2d.ndim != 2:
        return None, None, None
    n0 = min(data_2d.shape[0], model_2d.shape[0])
    n1 = min(data_2d.shape[1], model_2d.shape[1])
    d, m = data_2d[:n0, :n1], model_2d[:n0, :n1]
    img = (d - m) / np.sqrt(np.maximum(d, 1.0)) if weighted else d - m
    return img, np.arange(n1, dtype=float), np.arange(n0, dtype=float)


def get_rics_data_image(fit_group):
    """Return the experimental 2D RICS map (ICS mean image)."""
    rics_meta, _ = _rics_meta(fit_group)
    try:
        data_2d = np.asarray(rics_meta.get("ics_mean"), dtype=float)
    except Exception:
        return None, None, None
    if data_2d is None or data_2d.ndim != 2:
        return None, None, None
    n0, n1 = data_2d.shape
    return data_2d, np.arange(n1, dtype=float), np.arange(n0, dtype=float)


def get_rics_model_image(fit_group):
    """Return the analytic 2D RICS model image from the current fit."""
    fit = getattr(fit_group, "selected_fit", fit_group)
    try:
        model_2d = np.asarray(getattr(getattr(fit, "model", None), "rics_model_2d", None), dtype=float)
    except Exception:
        return None, None, None
    if model_2d is None or model_2d.ndim != 2:
        return None, None, None
    n0, n1 = model_2d.shape
    return model_2d, np.arange(n1, dtype=float), np.arange(n0, dtype=float)


# --- parameter groups ------------------------------------------------------
class RicsImaging(FittingParameterGroup):
    """Fixed acquisition parameters (pixel/line timing, PSF, pixel size)."""

    def __init__(self, name: str = "rics_imaging", **kwargs):
        """Initialize the imaging/acquisition parameter group."""
        super().__init__(name=name, **kwargs)
        self._pixel_duration = FittingParameter(
            value=11.1, name="pxl_dur", lb=0.01, ub=1e3, bounds_on=False, fixed=True,
            label_text="t<sub>pix</sub>[µs]", registry_id="rics.pxl_dur")
        self._line_duration = FittingParameter(
            value=3.33, name="line_dur", lb=0.001, ub=1e4, bounds_on=False, fixed=True,
            label_text="t<sub>line</sub>[ms]", registry_id="rics.line_dur")
        self._pixel_size = FittingParameter(
            value=40.0, name="pxl_size", lb=1.0, ub=1e4, bounds_on=False, fixed=True,
            label_text="a[nm]", registry_id="rics.pxl_size")
        self._w_r = FittingParameter(
            value=0.2, name="w_r", lb=1e-3, ub=1e2, bounds_on=False, fixed=True,
            label_text="w<sub>r</sub>[µm]", registry_id="rics.w_r")
        self._w_z = FittingParameter(
            value=1.0, name="w_z", lb=1e-3, ub=1e3, bounds_on=False, fixed=True,
            label_text="w<sub>z</sub>[µm]", registry_id="rics.w_z")

    pixel_duration = property(lambda s: float(s._pixel_duration.value))
    line_duration = property(lambda s: float(s._line_duration.value))
    pixel_size = property(lambda s: float(s._pixel_size.value))
    w_r = property(lambda s: float(s._w_r.value))
    w_z = property(lambda s: float(s._w_z.value))


class RicsDiffusion(FittingParameterGroup):
    """Core diffusion parameters (number of molecules, D, offset)."""

    def __init__(self, name: str = "rics_diffusion", **kwargs):
        """Initialize the diffusion parameter group."""
        super().__init__(name=name, **kwargs)
        self._n = FittingParameter(
            value=1.0, name="N", lb=1e-6, ub=1e9, bounds_on=True, fixed=False,
            label_text="N", registry_id="rics.n")
        self._D = FittingParameter(
            value=2.0, name="D", lb=1e-6, ub=1e3, bounds_on=True, fixed=False,
            label_text="D[µm²/s]", registry_id="rics.D")
        self._offset = FittingParameter(
            value=0.0, name="offset", lb=-1e3, ub=1e3, bounds_on=True, fixed=False,
            label_text="y<sub>0</sub>", registry_id="rics.offset")

    n = property(lambda s: float(s._n.value))
    D = property(lambda s: float(s._D.value))
    offset = property(lambda s: float(s._offset.value))


class RicsTriplet(FittingParameterGroup):
    """Triplet/blinking parameters."""

    def __init__(self, name: str = "rics_triplet", **kwargs):
        """Initialize the triplet parameter group."""
        super().__init__(name=name, **kwargs)
        self._tauT = FittingParameter(
            value=0.002, name="tauT", lb=1e-6, ub=1.0, bounds_on=True, fixed=False,
            label_text="&tau;<sub>T</sub>[ms]", registry_id="rics.tauT")
        self._aT = FittingParameter(
            value=0.1, name="aT", lb=0.0, ub=0.99, bounds_on=True, fixed=False,
            label_text="a<sub>T</sub>", registry_id="rics.aT")

    tauT = property(lambda s: float(s._tauT.value))
    aT = property(lambda s: float(s._aT.value))


class RicsImmobile(FittingParameterGroup):
    """Immobile-fraction amplitude."""

    def __init__(self, name: str = "rics_immobile", **kwargs):
        """Initialize the immobile-fraction parameter group."""
        super().__init__(name=name, **kwargs)
        self._a_imm = FittingParameter(
            value=0.0, name="a_imm", lb=0.0, ub=1e3, bounds_on=True, fixed=False,
            label_text="A<sub>imm</sub>", registry_id="rics.a_imm")

    a_immobile = property(lambda s: float(s._a_imm.value))


class RicsFlow(FittingParameterGroup):
    """Uniform-flow velocities along the fast/slow scan axes."""

    def __init__(self, name: str = "rics_flow", **kwargs):
        """Initialize the flow parameter group."""
        super().__init__(name=name, **kwargs)
        self._vx = FittingParameter(
            value=0.0, name="v_x", lb=-1e4, ub=1e4, bounds_on=True, fixed=False,
            label_text="v<sub>x</sub>[µm/s]", registry_id="rics.v_x")
        self._vy = FittingParameter(
            value=0.0, name="v_y", lb=-1e4, ub=1e4, bounds_on=True, fixed=False,
            label_text="v<sub>y</sub>[µm/s]", registry_id="rics.v_y")

    v_x = property(lambda s: float(s._vx.value))
    v_y = property(lambda s: float(s._vy.value))


class RicsImmobileShift(FittingParameterGroup):
    """Immobile component with its own width and a lateral (ccRICS) shift."""

    def __init__(self, name: str = "rics_immobile_shift", **kwargs):
        """Initialize the immobile + shift parameter group."""
        super().__init__(name=name, **kwargs)
        self._n_imm = FittingParameter(
            value=0.0, name="N_imm", lb=0.0, ub=1e9, bounds_on=True, fixed=False,
            label_text="N<sub>imm</sub>", registry_id="rics.n_imm")
        self._w_imm = FittingParameter(
            value=0.2, name="w_imm", lb=1e-3, ub=1e2, bounds_on=True, fixed=True,
            label_text="w<sub>imm</sub>[µm]", registry_id="rics.w_imm")
        self._sx = FittingParameter(
            value=0.0, name="sx", lb=-1e4, ub=1e4, bounds_on=True, fixed=True,
            label_text="s<sub>x</sub>[nm]", registry_id="rics.sx")
        self._sy = FittingParameter(
            value=0.0, name="sy", lb=-1e4, ub=1e4, bounds_on=True, fixed=True,
            label_text="s<sub>y</sub>[nm]", registry_id="rics.sy")
        #: Membrane/2D geometry toggle (drop the axial w_z term).
        self.two_d = False

    n_immobile = property(lambda s: float(s._n_imm.value))
    w_immobile = property(lambda s: float(s._w_imm.value))
    shift_x = property(lambda s: float(s._sx.value))
    shift_y = property(lambda s: float(s._sy.value))


class IcsGaussian2D(FittingParameterGroup):
    """Anisotropic 2D-Gaussian structure parameters (two widths + angle)."""

    def __init__(self, name: str = "ics_gaussian2d", **kwargs):
        """Initialize the anisotropic-Gaussian parameter group."""
        super().__init__(name=name, **kwargs)
        self._a0 = FittingParameter(
            value=1.0, name="A0", lb=0.0, ub=1e9, bounds_on=True, fixed=False,
            label_text="A<sub>0</sub>", registry_id="ics.a0")
        self._s1 = FittingParameter(
            value=200.0, name="sigma1", lb=1.0, ub=1e5, bounds_on=True, fixed=False,
            label_text="&sigma;<sub>1</sub>[nm]", registry_id="ics.sigma1")
        self._s2 = FittingParameter(
            value=200.0, name="sigma2", lb=1.0, ub=1e5, bounds_on=True, fixed=False,
            label_text="&sigma;<sub>2</sub>[nm]", registry_id="ics.sigma2")
        self._angle = FittingParameter(
            value=0.0, name="angle", lb=0.0, ub=6.3, bounds_on=True, fixed=False,
            label_text="&theta;[rad]", registry_id="ics.angle")
        self._xo = FittingParameter(
            value=0.0, name="x_off", lb=-1e5, ub=1e5, bounds_on=True, fixed=True,
            label_text="x<sub>0</sub>[nm]", registry_id="ics.x_off")
        self._yo = FittingParameter(
            value=0.0, name="y_off", lb=-1e5, ub=1e5, bounds_on=True, fixed=True,
            label_text="y<sub>0</sub>[nm]", registry_id="ics.y_off")
        self._offset = FittingParameter(
            value=0.0, name="offset", lb=-1e3, ub=1e3, bounds_on=True, fixed=False,
            label_text="I<sub>0</sub>", registry_id="ics.offset")

    amplitude = property(lambda s: float(s._a0.value))
    sigma_1 = property(lambda s: float(s._s1.value))
    sigma_2 = property(lambda s: float(s._s2.value))
    angle = property(lambda s: float(s._angle.value))
    x_offset = property(lambda s: float(s._xo.value))
    y_offset = property(lambda s: float(s._yo.value))
    offset = property(lambda s: float(s._offset.value))


# --- base model ------------------------------------------------------------
class _RicsModelBase(ModelCurve):
    """Shared RICS plumbing: read the lag grid + timing, compute, store 2D/1D."""

    def __init__(self, fit: cs.core.fitting.fit.Fit, **kwargs) -> None:
        """Initialize common RICS groups and seed timing from metadata."""
        super().__init__(fit, **kwargs)
        self.diffusion = RicsDiffusion(name="rics_diffusion", fit=fit)
        self.imaging = RicsImaging(name="rics_imaging", fit=fit)
        self.rics_model_2d: np.ndarray | None = None
        self._seed_timing_from_meta()

    def _seed_timing_from_meta(self) -> None:
        """Initialize the fixed pixel/line durations from RICS metadata."""
        meta, _ = _rics_meta(self.fit)
        pd = meta.get("pixel_duration_us")
        ld = meta.get("line_duration_ms")
        if isinstance(pd, (int, float)) and pd > 0:
            self.imaging._pixel_duration.value = float(pd)
        if isinstance(ld, (int, float)) and ld > 0:
            self.imaging._line_duration.value = float(ld)

    def _compute(self, line_shift: np.ndarray, pixel_shift: np.ndarray) -> np.ndarray:
        """Return the 2D model on the lag grid. Implemented by subclasses."""
        raise NotImplementedError

    def update_model(self, **kwargs) -> None:
        """Read the lag grid + timing, compute the 2D model, store 2D and 1D."""
        meta, data = _rics_meta(self.fit)
        try:
            line_shift = np.asarray(meta.get("line_shift"), dtype=float)
            pixel_shift = np.asarray(meta.get("pixel_shift"), dtype=float)
        except Exception:
            line_shift = pixel_shift = None

        if line_shift is None or pixel_shift is None or line_shift.shape != pixel_shift.shape:
            self.rics_model_2d = None
            self.y = np.zeros_like(getattr(data, "y", np.zeros(0)), dtype=float)
            return

        self._seed_timing_from_meta()
        model_2d = np.asarray(self._compute(line_shift, pixel_shift), dtype=float)
        self.rics_model_2d = model_2d
        y_model = model_2d.ravel()

        x_data = getattr(data, "x", None)
        if x_data is not None and getattr(x_data, "size", 0) == y_model.size:
            self.x = np.asarray(x_data, dtype=float)
        else:
            self.x = np.arange(y_model.size, dtype=float)
        self.y = y_model


# --- concrete models -------------------------------------------------------
class RicsSimpleModel(_RicsModelBase):
    """Simple one-component 3D-diffusion RICS model."""

    name = "RICS diffusion"
    view_spec_file = "rics_simple.view.json"

    def _compute(self, line_shift, pixel_shift):
        """Compute the simple 3D-diffusion RICS surface."""
        d, im = self.diffusion, self.imaging
        return rics_simple(
            line_shift, pixel_shift, n=d.n, diffusion_coefficient=d.D, offset=d.offset,
            pixel_duration=im.pixel_duration, line_duration=im.line_duration,
            pixel_size=im.pixel_size, w_r=im.w_r, w_z=im.w_z)


class RicsTripletModel(_RicsModelBase):
    """RICS model with 3D diffusion and triplet/blinking."""

    name = "RICS diffusion + triplet"
    view_spec_file = "rics_triplet.view.json"

    def __init__(self, fit, **kwargs):
        """Initialize with an added triplet parameter group."""
        super().__init__(fit, **kwargs)
        self.triplet = RicsTriplet(name="rics_triplet", fit=fit)

    def _compute(self, line_shift, pixel_shift):
        """Compute the diffusion + triplet RICS surface."""
        d, im, t = self.diffusion, self.imaging, self.triplet
        return rics_diffusion_triplet(
            line_shift, pixel_shift, n=d.n, diffusion_coefficient=d.D, offset=d.offset,
            pixel_duration=im.pixel_duration, line_duration=im.line_duration,
            pixel_size=im.pixel_size, w_r=im.w_r, w_z=im.w_z, tauT=t.tauT, aT=t.aT)


class RicsImmobileModel(_RicsModelBase):
    """RICS model with a mobile diffusion component and an immobile fraction."""

    name = "RICS + immobile fraction"
    view_spec_file = "rics_immobile.view.json"

    def __init__(self, fit, **kwargs):
        """Initialize with an added immobile-fraction parameter group."""
        super().__init__(fit, **kwargs)
        self.immobile = RicsImmobile(name="rics_immobile", fit=fit)

    def _compute(self, line_shift, pixel_shift):
        """Compute the mobile + immobile RICS surface."""
        d, im, imm = self.diffusion, self.imaging, self.immobile
        return rics_immobile(
            line_shift, pixel_shift, n=d.n, diffusion_coefficient=d.D, offset=d.offset,
            pixel_duration=im.pixel_duration, line_duration=im.line_duration,
            pixel_size=im.pixel_size, w_r=im.w_r, w_z=im.w_z, a_immobile=imm.a_immobile)


class RicsFlowModel(_RicsModelBase):
    """RICS model with 3D diffusion and uniform flow."""

    name = "RICS + flow"
    view_spec_file = "rics_flow.view.json"

    def __init__(self, fit, **kwargs):
        """Initialize with an added flow parameter group."""
        super().__init__(fit, **kwargs)
        self.flow = RicsFlow(name="rics_flow", fit=fit)

    def _compute(self, line_shift, pixel_shift):
        """Compute the diffusion + flow RICS surface."""
        d, im, fl = self.diffusion, self.imaging, self.flow
        return rics_flow(
            line_shift, pixel_shift, n=d.n, diffusion_coefficient=d.D, offset=d.offset,
            pixel_duration=im.pixel_duration, line_duration=im.line_duration,
            pixel_size=im.pixel_size, w_r=im.w_r, w_z=im.w_z, v_x=fl.v_x, v_y=fl.v_y)


class RicsFullModel(_RicsModelBase):
    """RICS with mobile diffusion + immobile fraction + blinking + shift (2D/3D)."""

    name = "RICS (mobile+immobile+blinking+shift)"
    view_spec_file = "rics_full.view.json"

    def __init__(self, fit, **kwargs):
        """Initialize with triplet and immobile+shift parameter groups."""
        super().__init__(fit, **kwargs)
        self.triplet = RicsTriplet(name="rics_triplet", fit=fit)
        self.immobile_shift = RicsImmobileShift(name="rics_immobile_shift", fit=fit)

    def _compute(self, line_shift, pixel_shift):
        """Compute the full RICS surface (all optional terms combined)."""
        d, im, t, s = self.diffusion, self.imaging, self.triplet, self.immobile_shift
        return rics_full(
            line_shift, pixel_shift, n=d.n, diffusion_coefficient=d.D, offset=d.offset,
            pixel_duration=im.pixel_duration, line_duration=im.line_duration,
            pixel_size=im.pixel_size, w_r=im.w_r, w_z=im.w_z, tauT=t.tauT, aT=t.aT,
            n_immobile=s.n_immobile, w_immobile=s.w_immobile,
            shift_x=s.shift_x, shift_y=s.shift_y, two_d=bool(s.two_d))


class IcsGaussian2DModel(_RicsModelBase):
    """Anisotropic 2D-Gaussian spatial-correlation model (structure sizing)."""

    name = "ICS 2D Gaussian (2 sigma + angle)"
    view_spec_file = "ics_gaussian2d.view.json"

    def __init__(self, fit, **kwargs):
        """Initialize with the anisotropic-Gaussian parameter group."""
        super().__init__(fit, **kwargs)
        self.gaussian = IcsGaussian2D(name="ics_gaussian2d", fit=fit)

    def _compute(self, line_shift, pixel_shift):
        """Compute the anisotropic 2D-Gaussian spatial correlation surface."""
        gp, im = self.gaussian, self.imaging
        return ics_gaussian_2d(
            line_shift, pixel_shift, amplitude=gp.amplitude, pixel_size=im.pixel_size,
            sigma_1=gp.sigma_1, sigma_2=gp.sigma_2, angle=gp.angle,
            x_offset=gp.x_offset, y_offset=gp.y_offset, offset=gp.offset)
