"""Generic, mixable FRET line computation engine.

A FRET line is computed for a *mixture* of one or more lifetime/FRET models.
Each component is any model registered in MODEL_REGISTRY.  The swept quantity
is either:

  * a parameter of any component model (e.g. ``R(G,1)``, ``l``, ``lp``), or
  * the mixing fraction of any component (for a >1-component mixture).

A single-component mixture reproduces an ordinary single-model FRET line.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# append_attr  : attribute carrying an .append() method for adding components
#               (Gaussians, discrete distances, …); None → no append step.
# append_args  : positional args for a single append() call.

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "FRET: FD (Gaussian)": {
        "module": "chisurf.core.models.tcspc.fret",
        "class": "GaussianModel",
        "append_attr": "gaussians",
        "append_args": (50.0, 6.0, 1.0),
    },
    "FRET: FD (Worm-like chain)": {
        "module": "chisurf.core.models.tcspc.fret",
        "class": "WormLikeChainModel",
        "append_attr": None,
        "append_args": None,
    },
    "FRET: FD (Discrete)": {
        "module": "chisurf.core.models.tcspc.fret",
        "class": "FRETrateModel",
        "append_attr": "fret_rates",
        "append_args": (50.0, 1.0),
    },
    "FRET: Fixed distance": {
        "module": "chisurf.core.models.tcspc.fret",
        "class": "SingleDistanceModel",
        "append_attr": None,
        "append_args": None,
    },
    "Lifetime (new)": {
        "module": "chisurf.core.models.tcspc.lifetime",
        "class": "LifetimeNewModel",
        "append_attr": None,
        "append_args": None,
    },
}


def list_models() -> list[str]:
    """Return the display names of all registered component models."""
    return list(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_fit():
    import chisurf.core.data
    import chisurf.core.fitting.fit

    f = chisurf.core.fitting.fit.Fit()
    x = np.linspace(0.0, 50.0, 200)
    f.data = chisurf.core.data.DataCurve(x=x, y=np.zeros_like(x))
    return f


def _build_component(model_name: str, n_components: int = 1):
    """Instantiate one component model, append sub-components, find_parameters().

    Returns the model instance (its ``fit`` is held alive by the model).
    """
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {list_models()}")

    mod = importlib.import_module(entry["module"])
    cls = getattr(mod, entry["class"])
    fit = _make_fit()
    model = cls(fit=fit)

    append_attr = entry.get("append_attr")
    append_args = entry.get("append_args") or ()
    if append_attr is not None:
        container = getattr(model, append_attr, None)
        if container is not None:
            for _ in range(max(1, int(n_components))):
                container.append(*append_args)

    model.find_parameters()
    return model


def _params_of(model) -> dict[str, Any]:
    """Return {name: FittingParameter} for ALL parameters (including fixed)."""
    return {p.name: p for p in model.parameters_all}


def _infer_tau_d0(models, fallback: float = 4.0) -> float:
    """Best-effort donor lifetime from the first FRET component."""
    for model in models:
        fret_params = getattr(model, "fret_parameters", None)
        if fret_params is not None:
            try:
                return float(fret_params.tauD0)
            except Exception:
                pass
    for model in models:
        pd = _params_of(model)
        for key in ("t0", "tL1"):
            if key in pd:
                try:
                    return float(pd[key].value)
                except Exception:
                    pass
    return fallback


def _build_mixture(component_models):
    """Wrap one or more component models in a LifetimeMixtureModel."""
    import chisurf.core.models.tcspc.lifetime as lt

    fit = _make_fit()
    mix = lt.LifetimeMixtureModel(fit=fit)
    for i, m in enumerate(component_models):
        mix.append_model(m, name=f"x{i + 1}")
    return mix


def _set_fraction(mix, idx: int, frac: float) -> None:
    """Set component *idx* to mixing fraction *frac* (0..1).

    The remaining weight (1 − frac) is distributed among the other components
    in proportion to their current weights, so for a 2-component mixture this
    gives exactly ``[1 − frac, frac]``.
    """
    n = mix.n_model
    if n <= 1:
        return
    others = [j for j in range(n) if j != idx]
    cur = np.array([max(mix._fractions[j].value, 0.0) for j in others], float)
    if cur.sum() <= 0:
        cur = np.ones(len(others))
    cur = cur / cur.sum() * (1.0 - frac)
    weights = np.zeros(n)
    weights[idx] = frac
    for k, j in enumerate(others):
        weights[j] = cur[k]
    mix.fractions = weights


def _normalize_components(components) -> list[dict]:
    """Validate / normalize the component spec list."""
    if not components:
        raise ValueError("At least one component is required.")
    out = []
    for c in components:
        out.append(
            {
                "model_name": c["model_name"],
                "n_components": int(c.get("n_components", 1)),
                "params": dict(c.get("params") or {}),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_model_parameters(model_name: str, n_components: int = 1) -> list[dict]:
    """Return all parameters of a freshly built model.

    Each item is ``{"name": str, "value": float, "fixed": bool}``.
    """
    model = _build_component(model_name, n_components)
    return [
        {"name": p.name, "value": float(p.value), "fixed": bool(p.fixed)}
        for p in model.parameters_all
    ]


# ---------------------------------------------------------------------------
# Sweep core (shared by the spec-based and instance-based entry points)
# ---------------------------------------------------------------------------

#: Timing / IRF / background / anisotropy nuisance parameters that are present
#: on every TCSPC model but never meaningfully sweep a FRET line. Hidden by
#: default in the sweep-target list (``relevant_only=True``).
NUISANCE_PARAMS: frozenset[str] = frozenset(
    {
        "sc",
        "bg",
        "ts",
        "tBg",
        "tMeas",
        "tDead",
        "win-size",
        "n0",
        "dt",
        "rep",
        "start",
        "stop",
        "lb",
        "irf_start",
        "irf_stop",
        "iw",
        "ik",
        "r0",
        "g",
        "l1",
        "l2",
        "xDOnly",
        "E_FRET",
    }
)

#: Prefixes of indexed / grouped nuisance parameters: anisotropy rotational
#: terms (``b(…)``, ``rho(…)``), steady-state anisotropy (``r_ss_…``),
#: polarized background intensities (``vv_…``, ``vh_…``) and photon counts
#: (``nPh_…``).
_NUISANCE_PREFIXES: tuple[str, ...] = (
    "b(",
    "rho(",
    "r_ss",
    "vv_",
    "vh_",
    "nPh",
)


def _is_relevant(name: str) -> bool:
    """Return True if *name* is a meaningful sweep target (not a nuisance)."""
    if name in NUISANCE_PARAMS:
        return False
    return not name.startswith(_NUISANCE_PREFIXES)


def _targets_for_models(
    models, names: list[str] | None = None, relevant_only: bool = True
) -> list[dict]:
    """Build sweep targets from already-built model instances.

    *names* optionally supplies a display name per component (defaults to the
    model class ``name`` attribute).  When *relevant_only* is True the timing /
    IRF / background / anisotropy nuisance parameters are omitted.
    """
    targets: list[dict] = []
    for ci, model in enumerate(models):
        disp = (
            names[ci] if names and ci < len(names) else getattr(model, "name", type(model).__name__)
        )
        prefix = f"C{ci} [{disp}]"
        for p in model.parameters_all:
            if relevant_only and not _is_relevant(p.name):
                continue
            targets.append(
                {
                    "label": f"{prefix} · {p.name}",
                    "kind": "param",
                    "component": ci,
                    "name": p.name,
                }
            )
    if len(models) > 1:
        for ci, model in enumerate(models):
            disp = (
                names[ci]
                if names and ci < len(names)
                else getattr(model, "name", type(model).__name__)
            )
            targets.append(
                {
                    "label": f"fraction · C{ci} [{disp}]",
                    "kind": "fraction",
                    "component": ci,
                    "name": None,
                }
            )
    return targets


def _run_sweep(
    models,
    sweep: dict,
    param_min: float,
    param_max: float,
    n_points: int,
    fractions: list[float] | None,
    tau_d0: float | None,
    log_scale: bool,
) -> dict:
    """Mix *models*, sweep the requested quantity, return the result dict.

    Raises ``ValueError`` with a human-readable message on a bad sweep spec.
    """
    if not models:
        raise ValueError("At least one component is required.")

    # find_parameters() so parameters_all reflects the current structure.
    for m in models:
        try:
            m.find_parameters()
        except Exception:
            pass
    param_dicts = [_params_of(m) for m in models]

    mix = _build_mixture(models)
    if fractions:
        for fp, w in zip(mix._fractions, fractions):
            fp.value = max(float(w), 1e-9)

    if tau_d0 is None:
        tau_d0 = _infer_tau_d0(models)

    kind = sweep.get("kind", "param")
    ci = int(sweep.get("component", 0))
    if ci < 0 or ci >= len(models):
        raise ValueError(f"Component index {ci} out of range.")

    if kind == "param":
        pname = sweep.get("name")
        pd = param_dicts[ci]
        if pname not in pd:
            raise ValueError(
                f"Parameter {pname!r} not found in component {ci}. Available: {sorted(pd.keys())}"
            )
        sweep_param = pd[pname]

        def _apply(value):
            sweep_param.value = float(value)

    elif kind == "fraction":
        if len(models) <= 1:
            raise ValueError("Fraction sweep requires at least two components.")

        def _apply(value):
            _set_fraction(mix, ci, float(value))

    else:
        raise ValueError(f"Unknown sweep kind {kind!r}.")

    if log_scale:
        if param_min <= 0 or param_max <= 0:
            raise ValueError("Log-scale sweep requires param_min and param_max > 0.")
        param_values = np.geomspace(float(param_min), float(param_max), int(n_points))
    else:
        param_values = np.linspace(float(param_min), float(param_max), int(n_points))

    tau_f_arr = np.empty(int(n_points))
    tau_x_arr = np.empty(int(n_points))
    e_fret_arr = np.empty(int(n_points))
    for i, pv in enumerate(param_values):
        _apply(pv)
        tau_f_arr[i] = float(mix.fluorescence_averaged_lifetime)
        tau_x_arr[i] = float(mix.species_averaged_lifetime)
        e_fret_arr[i] = 1.0 - tau_x_arr[i] / tau_d0 if tau_d0 > 0 else float("nan")

    return {
        "parameter_values": param_values.tolist(),
        "tau_f": tau_f_arr.tolist(),
        "tau_x": tau_x_arr.tolist(),
        "e_fret": e_fret_arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Public API — spec-based (headless: CLI / RPC / tests)
# ---------------------------------------------------------------------------


def list_sweep_targets(components: list[dict], relevant_only: bool = True) -> list[dict]:
    """Enumerate sweepable parameters/fractions for a component *spec* list.

    Each target::

        {"label": str, "kind": "param"|"fraction", "component": int, "name": str|None}

    When *relevant_only* is True nuisance parameters are omitted.
    """
    comps = _normalize_components(components)
    models = [_build_component(c["model_name"], c["n_components"]) for c in comps]
    names = [c["model_name"] for c in comps]
    for m in models:
        m.find_parameters()
    return _targets_for_models(models, names, relevant_only)


def compute_fret_line(
    components: list[dict],
    sweep: dict,
    param_min: float,
    param_max: float,
    n_points: int = 100,
    fractions: list[float] | None = None,
    tau_d0: float | None = None,
    log_scale: bool = False,
) -> dict:
    """Compute a FRET line for a mixture of models built from specs.

    Parameters
    ----------
    components : list of dict
        One entry per mixture component::

            {"model_name": str, "n_components": int, "params": {name: value}}

    sweep : dict
        What to vary::

            {"kind": "param", "component": int, "name": str}   # a model parameter
            {"kind": "fraction", "component": int}             # a mixing fraction

    param_min, param_max : float
        Sweep range endpoints.
    n_points : int
        Number of sweep points.
    fractions : list of float, optional
        Initial mixing weights (one per component). Defaults to equal weights.
    tau_d0 : float, optional
        Reference donor lifetime for E_FRET = 1 − τ_X / τ_D0. Auto-detected
        from the first FRET component when not supplied.
    log_scale : bool
        Logarithmic sweep spacing (both endpoints must be > 0) when True.

    Returns
    -------
    dict
        ``{"ok": True, "result": {...}}`` or ``{"ok": False, "error": "..."}``.
    """
    try:
        comps = _normalize_components(components)
        models = []
        for c in comps:
            m = _build_component(c["model_name"], c["n_components"])
            pd = _params_of(m)
            for k, v in c["params"].items():
                if k in pd:
                    pd[k].value = float(v)
            models.append(m)
        result = _run_sweep(
            models, sweep, param_min, param_max, n_points, fractions, tau_d0, log_scale
        )
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Public API — instance-based (GUI: live widget-models)
# ---------------------------------------------------------------------------


def sweep_targets_for_models(
    models, names: list[str] | None = None, relevant_only: bool = True
) -> list[dict]:
    """Enumerate sweepable parameters/fractions for live model instances.

    When *relevant_only* is True nuisance parameters are omitted.
    """
    for m in models:
        try:
            m.find_parameters()
        except Exception:
            pass
    return _targets_for_models(models, names, relevant_only)


def compute_fret_line_for_models(
    models,
    sweep: dict,
    param_min: float,
    param_max: float,
    n_points: int = 100,
    fractions: list[float] | None = None,
    tau_d0: float | None = None,
    log_scale: bool = False,
) -> dict:
    """Compute a FRET line by sweeping over already-built model instances.

    Used by the GUI, which edits live widget-models with the native fitting
    editors and passes them here directly (no rebuild from specs).
    """
    try:
        result = _run_sweep(
            models, sweep, param_min, param_max, n_points, fractions, tau_d0, log_scale
        )
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
