from __future__ import annotations

"""Qt-free DEER/PELDOR fitting models (PRD-38 model/view-spec split).

Native, self-contained reimplementation (numpy/scipy only) of 4-pulse DEER analysis.
Models operate on the time-domain trace ``V(t)`` stored as the ``DataCurve``
``x``/``y`` plus ``data.meta_data['deer']`` produced by
:class:`chisurf.core.experiments.deer.DeerReader`. The physics lives in
:mod:`chisurf.core.models.deer.kernel` / ``.tikhonov`` and is wrapped by the
thin functions in :mod:`chisurf.core.models.deer.models`.

Models
------
* :class:`DeerGaussianModel` — one or more Gaussian distance components.
* :class:`DeerRiceModel` — a single 3D-Rice distance component.
* :class:`DeerTikhonovModel` — model-free (Tikhonov-regularised) ``P(r)``.
"""

import numpy as np

import chisurf as cs
from chisurf.core.fitting.parameter import FittingParameter, FittingParameterGroup
from chisurf.core.models.model import ModelCurve
from chisurf.core.models.deer import models as _m
from chisurf.core.models.deer.kernel import dipolar_kernel


# --- metadata helper -------------------------------------------------------
def _deer_meta(fit_group) -> tuple[dict, object]:
    """Return ``(deer_meta_dict, data)`` for a fit group or plain fit."""
    fit = getattr(fit_group, "selected_fit", fit_group)
    data = getattr(fit, "data", None)
    meta = getattr(data, "meta_data", {}) or {}
    return (meta.get("deer", {}) or {}), data


# --- parameter groups ------------------------------------------------------
class DeerModulation(FittingParameterGroup):
    """Modulation depth, dipolar zero-time and overall amplitude scale."""

    def __init__(self, name: str = "deer_modulation", **kwargs):
        """Initialize the modulation/zero-time/scale group."""
        super().__init__(name=name, **kwargs)
        self._lam = FittingParameter(
            value=0.3, name="lambda", lb=0.0, ub=1.0, bounds_on=True, fixed=False,
            label_text="&lambda;", registry_id="deer.lambda")
        self._t0 = FittingParameter(
            value=0.0, name="t0", lb=-0.5, ub=0.5, bounds_on=True, fixed=True,
            label_text="t<sub>0</sub>[µs]", registry_id="deer.t0")
        # Fixed by default: the data is normalised to V(t0)=1 and the model
        # already yields V(t0)=scale, so a free scale is degenerate with the
        # overall amplitude and stalls the shape parameters. Users can release
        # it in the editor if a trace is not pre-normalised.
        self._scale = FittingParameter(
            value=1.0, name="scale", lb=1e-3, ub=1e3, bounds_on=True, fixed=True,
            label_text="scale", registry_id="deer.scale")

    mod_depth = property(lambda s: float(s._lam.value))
    zero_time = property(lambda s: float(s._t0.value))
    scale = property(lambda s: float(s._scale.value))


class DeerBackground(FittingParameterGroup):
    """Intermolecular background model (kind + decay rate + fractal dim)."""

    def __init__(self, name: str = "deer_background", **kwargs):
        """Initialize the background group."""
        super().__init__(name=name, **kwargs)
        #: Background kind selected in the editor (choice widget).
        self.model = "hom3d"
        self._k = FittingParameter(
            value=0.05, name="bg_k", lb=0.0, ub=1e2, bounds_on=True, fixed=False,
            label_text="k[µs<sup>-1</sup>]", registry_id="deer.bg_k")
        self._d = FittingParameter(
            value=3.0, name="bg_d", lb=1.0, ub=6.0, bounds_on=True, fixed=True,
            label_text="d", registry_id="deer.bg_d")

    k = property(lambda s: float(s._k.value))
    d = property(lambda s: float(s._d.value))


class DeerGaussians(FittingParameterGroup):
    """Variable number of Gaussian distance components ``(r_mean, sigma, amp)``."""

    def __init__(self, name: str = "deer_gaussians", short: str = "", **kwargs):
        """Initialize an (initially empty) multi-Gaussian group."""
        super().__init__(name=name, **kwargs)
        self.short = short
        self._means: list = []
        self._sigmas: list = []
        self._amps: list = []

    def __len__(self) -> int:
        """Return the number of Gaussian components."""
        return len(self._means)

    @property
    def n(self) -> int:
        """Number of Gaussian components."""
        return len(self._means)

    @property
    def means(self) -> np.ndarray:
        """Component centre distances (nm)."""
        return np.array([p.value for p in self._means], dtype=float)

    @property
    def sigmas(self) -> np.ndarray:
        """Component widths (nm), always positive."""
        return np.array([abs(p.value) for p in self._sigmas], dtype=float)

    @property
    def amplitudes(self) -> np.ndarray:
        """Component relative amplitudes, non-negative and summing to one."""
        vs = np.array([abs(p.value) for p in self._amps], dtype=float)
        s = vs.sum()
        return vs / s if s > 0 else vs

    def append(self, mean: float = 35.0, sigma: float = 3.0, amplitude: float = 1.0,
               fixed: bool = False, **kwargs):
        """Add a Gaussian component (distances in Å)."""
        i = len(self._means) + 1
        s = self.short
        r_mean = FittingParameter(
            value=mean, name=f"r{s}{i}", lb=10.0, ub=150.0, bounds_on=True, fixed=fixed,
            label_text=f"r<sub>{i}</sub>[&#8491;]", registry_id=f"deer.r{s}{i}")
        r_sigma = FittingParameter(
            value=sigma, name=f"sig{s}{i}", lb=0.1, ub=50.0, bounds_on=True, fixed=fixed,
            label_text=f"&sigma;<sub>{i}</sub>[&#8491;]", registry_id=f"deer.sig{s}{i}")
        amp = FittingParameter(
            value=amplitude, name=f"a{s}{i}", lb=0.0, ub=1.0, bounds_on=True,
            fixed=(i == 1), label_text=f"a<sub>{i}</sub>", registry_id=f"deer.a{s}{i}")
        self._means.append(r_mean)
        self._sigmas.append(r_sigma)
        self._amps.append(amp)
        if getattr(self, "_parameters", None) is not None:
            self.append_parameter(r_mean)
            self.append_parameter(r_sigma)
            self.append_parameter(amp)

    def pop(self):
        """Remove and return the last component ``(mean, sigma, amp)``."""
        if not self._means:
            return None
        return self._means.pop(), self._sigmas.pop(), self._amps.pop()

    def _gaussian_parameter_rows(self) -> list:
        """Interleave parameters as ``(mean_i, sigma_i, amp_i)`` triples for the editor."""
        rows: list = []
        for m, s, a in zip(self._means, self._sigmas, self._amps):
            rows.extend((m, s, a))
        return rows


class DeerRice(FittingParameterGroup):
    """Single 3D-Rice distance component ``(nu, sigma)``."""

    def __init__(self, name: str = "deer_rice", **kwargs):
        """Initialize the Rice-distribution group."""
        super().__init__(name=name, **kwargs)
        self._nu = FittingParameter(
            value=35.0, name="nu", lb=10.0, ub=150.0, bounds_on=True, fixed=False,
            label_text="&nu;[&#8491;]", registry_id="deer.nu")
        self._sigma = FittingParameter(
            value=3.0, name="rice_sigma", lb=0.1, ub=50.0, bounds_on=True, fixed=False,
            label_text="&sigma;[&#8491;]", registry_id="deer.rice_sigma")

    nu = property(lambda s: float(s._nu.value))
    sigma = property(lambda s: float(abs(s._sigma.value)))


class DeerDistanceGrid(FittingParameterGroup):
    """User-controllable distance axis (Å) for the ``P(r)`` inversion/render.

    ``r_max = 0`` means "auto" (derived from the trace length). Increase
    ``r_max`` for long-distance samples and reduce ``n_points`` if the inversion
    is slow or ill-conditioned.
    """

    def __init__(self, name: str = "deer_grid", **kwargs):
        """Initialize the distance-grid settings group."""
        super().__init__(name=name, **kwargs)
        self._r_min = FittingParameter(
            value=15.0, name="r_min", lb=5.0, ub=100.0, bounds_on=True, fixed=True,
            label_text="r<sub>min</sub>[&#8491;]", registry_id="deer.r_min")
        self._r_max = FittingParameter(
            value=0.0, name="r_max", lb=0.0, ub=300.0, bounds_on=True, fixed=True,
            label_text="r<sub>max</sub>[&#8491;] (0=auto)", registry_id="deer.r_max")
        self._n = FittingParameter(
            value=100.0, name="n_r", lb=16.0, ub=400.0, bounds_on=True, fixed=True,
            label_text="n<sub>points</sub>", registry_id="deer.n_r")

    r_min = property(lambda s: float(s._r_min.value))
    r_max = property(lambda s: float(s._r_max.value))
    n_points = property(lambda s: int(round(s._n.value)))


class DeerRegularization(FittingParameterGroup):
    """Tikhonov regularisation weight ``alpha`` (``0`` = auto-select)."""

    def __init__(self, name: str = "deer_regularization", **kwargs):
        """Initialize the regularisation-weight group."""
        super().__init__(name=name, **kwargs)
        #: Auto-selection criterion for ``alpha`` when it is left at 0.
        self.method = "gcv"  # 'gcv' or 'lcurve'
        self._alpha = FittingParameter(
            value=0.0, name="alpha", lb=0.0, ub=1e2, bounds_on=True, fixed=True,
            label_text="&alpha;", registry_id="deer.alpha")

    alpha = property(lambda s: float(s._alpha.value))


# --- base model ------------------------------------------------------------
class _DeerModelBase(ModelCurve):
    """Shared DEER plumbing: read ``V(t)``, build the r-grid, store ``P(r)``."""

    def __init__(self, fit: cs.core.fitting.fit.Fit, **kwargs) -> None:
        """Initialize the shared modulation/background groups and caches."""
        super().__init__(fit, **kwargs)
        self.modulation = DeerModulation(name="deer_modulation", fit=fit)
        self.background = DeerBackground(name="deer_background", fit=fit)
        self.grid = DeerDistanceGrid(name="deer_grid", fit=fit)
        self._r: np.ndarray | None = None
        self._p_r: np.ndarray | None = None
        self._kernel: np.ndarray | None = None
        self._kernel_key: tuple | None = None
        #: Cached auto-selected regularisation weight (model-free models).
        self._alpha_cached: float | None = None
        # Seed the dipolar zero time from the reader metadata when available.
        meta, data = _deer_meta(fit)
        t0 = meta.get("t0")
        if isinstance(t0, (int, float)):
            self.modulation._t0.value = float(t0)
        # Seed the modulation depth from the trace plateau so the fit starts near
        # the solution (V(t0)=1 is normalised; the tail sits near ~(1-lambda)).
        v = getattr(data, "y", None)
        if v is not None and np.size(v) > 8:
            v = np.asarray(v, dtype=float)
            plateau = float(np.median(v[int(0.75 * v.size):]))
            self.modulation._lam.value = float(np.clip(1.0 - plateau, 0.05, 0.6))

    def _alpha_for_update(self, reg) -> float | None:
        """Return the regularisation ``alpha`` to use this update.

        Mirrors the FCS MaxEnt design: ``alpha`` is chosen *once* and then kept
        fixed, so the outer least-squares objective (over ``lambda``/background)
        stays smooth. A manual, non-trivial ``reg.alpha`` always wins; otherwise
        the first automatic selection — made at the data-seeded ``lambda`` — is
        cached and reused. The small floor also guards against a stray tiny value
        left by parameter sampling that would otherwise collapse the distribution.
        """
        manual = reg.alpha if reg.alpha and reg.alpha > 1e-6 else None
        if manual is not None:
            return manual
        return self._alpha_cached

    def _remember_alpha(self, reg, alpha_used: float) -> None:
        """Cache the first auto-selected ``alpha`` (no manual override active)."""
        manual = reg.alpha if reg.alpha and reg.alpha > 1e-6 else None
        if manual is None and self._alpha_cached is None:
            self._alpha_cached = float(alpha_used)

    # -- to be provided by subclasses --
    def _distribution(self, r: np.ndarray, t: np.ndarray, v_data: np.ndarray):
        """Return ``(v_model, p_r)`` for the current parameters.

        ``t`` is the zero-time-corrected axis and ``v_data`` the measured trace
        (needed by the model-free Tikhonov subclass).
        """
        raise NotImplementedError

    def _time_and_data(self):
        """Return ``(t_axis_us, v_data)`` from the attached fit, or ``(None, None)``."""
        meta, data = _deer_meta(self.fit)
        if data is None:
            return None, None
        t = getattr(data, "x", None)
        v = getattr(data, "y", None)
        if t is None or v is None:
            return None, None
        return np.asarray(t, dtype=float), np.asarray(v, dtype=float)

    def _build_r(self, t: np.ndarray) -> np.ndarray:
        """Build the distance axis (Å) from the grid settings.

        ``r_max = 0`` derives the upper distance from the trace length
        (``r_max = 5 * (t_max/2)**(1/3)`` nm, in Å); otherwise the user value is
        used. Rebuilt every update so edits to the grid take effect immediately.
        """
        g = self.grid
        t_max = max(float(np.max(np.abs(t))), 1e-3)
        auto_max = float(np.clip(50.0 * (t_max / 2.0) ** (1.0 / 3.0), 30.0, 300.0))
        r_max = g.r_max if g.r_max > 0 else auto_max
        r_min = min(g.r_min, r_max - 1.0)
        return np.linspace(r_min, r_max, max(g.n_points, 16))

    def _current_form_factor(self):
        """Return ``(K, r, F)`` — kernel, distance grid and form factor now.

        ``F = (V/(scale*B) - (1-lambda)) / lambda`` is the intramolecular signal
        the model-free inversions target for the current parameters.
        """
        from chisurf.core.models.deer.kernel import background as _bg

        t_raw, v = self._time_and_data()
        if t_raw is None:
            return None
        t = t_raw - self.modulation.zero_time
        self._r = self._build_r(t)
        r = self._r
        k_mat = self._get_kernel(t, r)
        b = _bg(t, self.background.model, self.background.k, self.background.d)
        lam = float(np.clip(self.modulation.mod_depth, 1e-3, 1.0))
        s = self.modulation.scale or 1.0
        f = (np.asarray(v, dtype=float) / (s * np.clip(b, 1e-9, None)) - (1.0 - lam)) / lam
        return k_mat, r, f

    def _form_factor_from(self, v: np.ndarray):
        """Return ``(K, r, F)`` for an arbitrary signal ``v`` at current nuisance.

        Same construction as :meth:`_current_form_factor` but for a supplied
        trace (used by the uncertainty bootstrap).
        """
        from chisurf.core.models.deer.kernel import background as _bg

        t_raw, _ = self._time_and_data()
        if t_raw is None:
            return None
        t = t_raw - self.modulation.zero_time
        r = self._r if self._r is not None else self._build_r(t)
        k_mat = self._get_kernel(t, r)
        b = _bg(t, self.background.model, self.background.k, self.background.d)
        lam = float(np.clip(self.modulation.mod_depth, 1e-3, 1.0))
        s = self.modulation.scale or 1.0
        f = (np.asarray(v, dtype=float) / (s * np.clip(b, 1e-9, None)) - (1.0 - lam)) / lam
        return k_mat, r, f

    def _data_sigma(self) -> float:
        """Return the per-point noise level used for the parametric bootstrap."""
        meta, data = _deer_meta(self.fit)
        s = meta.get("noise_level")
        if isinstance(s, (int, float)) and s > 0:
            return float(s)
        ey = getattr(data, "ey", None)
        try:
            m = float(np.median(np.asarray(ey, dtype=float)))
            if np.isfinite(m) and m > 0:
                return m
        except Exception:
            pass
        return 1e-2

    def _pr_bootstrap(self, v_b: np.ndarray) -> np.ndarray | None:
        """Return a ``P(r)`` realisation for a noisy signal ``v_b``.

        Implemented per model (fast re-inversion for model-free; local shape
        re-fit for parametric). ``None`` means "no uncertainty available".
        """
        return None

    def compute_uncertainty(self, n_boot: int = 120, ci: float = 95.0, seed: int = 0):
        """Return ``(r, p_best, p_lo, p_hi)`` — a bootstrap confidence band on P(r).

        A parametric (residual) bootstrap: Gaussian noise at the data noise level
        is repeatedly added to the fitted trace, ``P(r)`` is re-derived for each
        realisation (nuisance parameters held at their fitted values), and the
        pointwise ``ci``% percentile band is returned. ``None`` when the model
        does not support it or there is no data.
        """
        self.update_model()
        r = self._r
        p_best = self._p_r
        if r is None or p_best is None:
            return None
        v_model = np.asarray(self.y, dtype=float)
        if v_model.size == 0:
            return None
        sigma = self._data_sigma()
        rng = np.random.default_rng(seed)
        reals: list[np.ndarray] = []
        for _ in range(int(n_boot)):
            v_b = v_model + rng.normal(0.0, sigma, size=v_model.shape)
            try:
                p_b = self._pr_bootstrap(v_b)
            except Exception:
                p_b = None
            if p_b is not None and np.size(p_b) == np.size(p_best) and np.all(np.isfinite(p_b)):
                reals.append(np.asarray(p_b, dtype=float))
        if len(reals) < 5:
            return r, p_best, p_best, p_best
        arr = np.vstack(reals)
        half = (100.0 - float(ci)) / 2.0
        lo = np.percentile(arr, half, axis=0)
        hi = np.percentile(arr, 100.0 - half, axis=0)
        p_best = np.asarray(p_best, dtype=float)
        # Envelope the pointwise band so it always contains the fitted estimate
        # (a sharp peak whose position jitters can otherwise poke above the upper
        # pointwise percentile — confusing in a plot).
        lo = np.minimum(lo, p_best)
        hi = np.maximum(hi, p_best)
        return r, p_best, lo, hi

    def _get_kernel(self, t: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Return a cached dipolar kernel for the current ``(t, r)`` axes."""
        key = (t.shape[0], r.shape[0], float(t[0]), float(t[-1]), float(r[0]), float(r[-1]))
        if self._kernel is None or self._kernel_key != key:
            self._kernel = dipolar_kernel(t, r)
            self._kernel_key = key
        return self._kernel

    def update_model(self, **kwargs) -> None:
        """Read ``V(t)``, build ``P(r)`` and the model trace ``self.y``."""
        t_raw, v_data = self._time_and_data()
        if t_raw is None:
            self.y = np.zeros(0)
            return
        t = t_raw - self.modulation.zero_time
        self._r = self._build_r(t)
        r = self._r
        v_model, p_r = self._distribution(r, t, v_data)
        self._p_r = p_r
        self.x = t_raw
        self.y = np.asarray(v_model, dtype=float)


# --- concrete models -------------------------------------------------------
class DeerGaussianModel(_DeerModelBase):
    """DEER model with one or more Gaussian distance components."""

    name = "DEER Gaussian(s)"
    view_spec_file = "deer_gauss.view.json"

    def __init__(self, fit: cs.core.fitting.fit.Fit, **kwargs) -> None:
        """Initialize and seed a single Gaussian component."""
        super().__init__(fit, **kwargs)
        self.gaussians = DeerGaussians(name="deer_gaussians", fit=fit)
        if len(self.gaussians) == 0:
            self.gaussians.append(mean=35.0, sigma=3.0, amplitude=1.0)  # Å

    def _distribution(self, r, t, v_data):
        """Build the (multi-)Gaussian signal for the current parameters."""
        g, mo, bg = self.gaussians, self.modulation, self.background
        return _m.gaussian_signal(
            t, r, g.means, g.sigmas, g.amplitudes,
            mod_depth=mo.mod_depth, bg_model=bg.model, bg_k=bg.k, bg_d=bg.d,
            scale=mo.scale, kernel=self._get_kernel(t, r))

    def _pr_bootstrap(self, v_b):
        """Re-fit the Gaussian shape parameters to a noisy trace (nuisance fixed)."""
        from scipy.optimize import least_squares

        from chisurf.core.models.deer.kernel import dd_gauss_multi, deer_signal

        t_raw, _ = self._time_and_data()
        if t_raw is None:
            return None
        t = t_raw - self.modulation.zero_time
        r = self._r if self._r is not None else self._build_r(t)
        k_mat = self._get_kernel(t, r)
        mo, bg, g = self.modulation, self.background, self.gaussians
        n = len(g)
        m0, s0 = g.means, g.sigmas
        a0 = np.array([abs(p.value) for p in g._amps], dtype=float)
        x0 = np.concatenate([m0, s0, a0[1:]]) if n > 1 else np.concatenate([m0, s0])

        def unpack(x):
            mm = x[:n]
            ss = np.abs(x[n:2 * n])
            aa = np.concatenate([[a0[0]], np.abs(x[2 * n:])]) if n > 1 else np.array([a0[0]])
            return mm, ss, aa

        def resid(x):
            mm, ss, aa = unpack(x)
            p = dd_gauss_multi(r, mm, ss, aa)
            vm = deer_signal(t, r, p, mo.mod_depth, bg.model, bg.k, bg.d, mo.scale, kernel=k_mat)
            return vm - v_b

        try:
            res = least_squares(resid, x0, method="lm", max_nfev=60)
            mm, ss, aa = unpack(res.x)
        except Exception:
            mm, ss, aa = m0, s0, a0
        return dd_gauss_multi(r, mm, ss, aa)


class DeerRiceModel(_DeerModelBase):
    """DEER model with a single 3D-Rice distance component."""

    name = "DEER Rice"
    view_spec_file = "deer_rice.view.json"

    def __init__(self, fit: cs.core.fitting.fit.Fit, **kwargs) -> None:
        """Initialize with a Rice-distribution parameter group."""
        super().__init__(fit, **kwargs)
        self.rice = DeerRice(name="deer_rice", fit=fit)

    def _distribution(self, r, t, v_data):
        """Build the Rice-distribution signal for the current parameters."""
        rc, mo, bg = self.rice, self.modulation, self.background
        return _m.rice_signal(
            t, r, rc.nu, rc.sigma,
            mod_depth=mo.mod_depth, bg_model=bg.model, bg_k=bg.k, bg_d=bg.d,
            scale=mo.scale, kernel=self._get_kernel(t, r))

    def _pr_bootstrap(self, v_b):
        """Re-fit the Rice shape parameters to a noisy trace (nuisance fixed)."""
        from scipy.optimize import least_squares

        from chisurf.core.models.deer.kernel import dd_rice, deer_signal

        t_raw, _ = self._time_and_data()
        if t_raw is None:
            return None
        t = t_raw - self.modulation.zero_time
        r = self._r if self._r is not None else self._build_r(t)
        k_mat = self._get_kernel(t, r)
        mo, bg, rc = self.modulation, self.background, self.rice

        def resid(x):
            p = dd_rice(r, x[0], abs(x[1]))
            vm = deer_signal(t, r, p, mo.mod_depth, bg.model, bg.k, bg.d, mo.scale, kernel=k_mat)
            return vm - v_b

        try:
            res = least_squares(resid, [rc.nu, rc.sigma], method="lm", max_nfev=50)
            nu, sig = res.x[0], abs(res.x[1])
        except Exception:
            nu, sig = rc.nu, rc.sigma
        return dd_rice(r, nu, sig)


class DeerTikhonovModel(_DeerModelBase):
    """Model-free DEER: Tikhonov-regularised, non-negative ``P(r)``."""

    name = "DEER model-free (Tikhonov)"
    view_spec_file = "deer_tikhonov.view.json"

    def __init__(self, fit: cs.core.fitting.fit.Fit, **kwargs) -> None:
        """Initialize with a regularisation-weight parameter (0 = auto/GCV)."""
        super().__init__(fit, **kwargs)
        self.regularization = DeerRegularization(name="deer_regularization", fit=fit)
        self._alpha_used: float = 0.0

    def _distribution(self, r, t, v_data):
        """Invert ``P(r)`` by Tikhonov regularisation and rebuild the trace.

        Tikhonov's NNLS+GCV solve is deterministic and smooth in ``lambda``/
        background, so — unlike MaxEnt — its ``alpha`` is re-selected every
        iteration (no caching) which is stable and slightly more accurate.
        """
        mo, bg, reg = self.modulation, self.background, self.regularization
        manual = reg.alpha if reg.alpha and reg.alpha > 1e-6 else None
        v_model, p_r, alpha_used = _m.tikhonov_signal(
            t, r, v_data, mod_depth=mo.mod_depth, bg_model=bg.model,
            bg_k=bg.k, bg_d=bg.d, scale=mo.scale, alpha=manual,
            method=getattr(reg, "method", "gcv"), kernel=self._get_kernel(t, r))
        self._alpha_used = alpha_used
        return v_model, p_r

    def _pr_bootstrap(self, v_b):
        """Re-invert P(r) for a noisy trace at the fitted regularisation weight."""
        ff = self._form_factor_from(v_b)
        if ff is None:
            return None
        k_mat, r, f = ff
        alpha = self._alpha_used if self._alpha_used and self._alpha_used > 0 else None
        p, _ = _m.tikhonov_distance_distribution(
            k_mat, r, f, alpha=alpha,
            method=getattr(self.regularization, "method", "gcv"))
        return p

    def compute_lcurve(self) -> dict | None:
        """Return the Tikhonov L-curve for the current parameters.

        Keys: ``alphas``, ``rho`` (residual norm), ``eta`` (roughness),
        ``corner`` (index of the L-curve corner) and ``used`` (the value in
        effect). Used by the L-curve plot.
        """
        from chisurf.core.math.regularization import discrete_lcurve_corner
        from chisurf.core.models.deer import tikhonov as _t

        ff = self._current_form_factor()
        if ff is None:
            return None
        k_mat, r, f = ff
        dr = float(np.mean(np.diff(r)))
        A = k_mat * dr
        L = _t.second_derivative_operator(r.size)
        alphas, rho, eta = _t.lcurve(A, f, L)
        return {"alphas": alphas, "rho": rho, "eta": eta,
                "corner": discrete_lcurve_corner(rho, eta), "used": self._alpha_used}


class DeerMaxEntModel(_DeerModelBase):
    """Model-free DEER via maximum-entropy inversion of ``P(r)``."""

    name = "DEER model-free (MaxEnt)"
    view_spec_file = "deer_maxent.view.json"

    def __init__(self, fit: cs.core.fitting.fit.Fit, **kwargs) -> None:
        """Initialize with an entropy-weight parameter (0 = auto/L-curve)."""
        super().__init__(fit, **kwargs)
        self.regularization = DeerRegularization(name="deer_regularization", fit=fit)
        self._alpha_used: float = 0.0

    def _noise_level(self) -> float:
        """Return the data noise level (from reader metadata, or a default)."""
        meta, _ = _deer_meta(self.fit)
        sigma = meta.get("noise_level")
        return float(sigma) if isinstance(sigma, (int, float)) and sigma > 0 else 1e-2

    def _distribution(self, r, t, v_data):
        """Invert ``P(r)`` by maximum entropy and rebuild the trace."""
        mo, bg, reg = self.modulation, self.background, self.regularization
        v_model, p_r, alpha_used = _m.maxent_signal(
            t, r, v_data, mod_depth=mo.mod_depth, bg_model=bg.model,
            bg_k=bg.k, bg_d=bg.d, scale=mo.scale, sigma=self._noise_level(),
            alpha=self._alpha_for_update(reg), kernel=self._get_kernel(t, r))
        self._remember_alpha(reg, alpha_used)
        self._alpha_used = alpha_used
        return v_model, p_r

    def _pr_bootstrap(self, v_b):
        """Re-invert P(r) for a noisy trace at the fitted entropy weight."""
        from chisurf.core.models.deer.maxent import maxent_distance_distribution

        ff = self._form_factor_from(v_b)
        if ff is None:
            return None
        k_mat, r, f = ff
        alpha = self._alpha_used if self._alpha_used and self._alpha_used > 0 else None
        p, _ = maxent_distance_distribution(
            k_mat, r, f, sigma=self._noise_level(), alpha=alpha, n_iter=400)
        return p

    def compute_lcurve(self) -> dict | None:
        """Return the MaxEnt L-curve (residual vs roughness) for current params."""
        from chisurf.core.models.deer.maxent import maxent_distance_distribution

        ff = self._current_form_factor()
        if ff is None:
            return None
        k_mat, r, f = ff
        _p, _a, info = maxent_distance_distribution(
            k_mat, r, f, sigma=self._noise_level(), alpha=None, return_lcurve=True)
        if info is None:
            return None
        info["used"] = self._alpha_used
        return info
