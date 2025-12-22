from __future__ import annotations

"""Utilities for serializing and restoring fit/model parameter state.

These helpers are intentionally GUI‑independent and operate purely on the
core fitting objects (:class:`chisurf.fitting.fit.Fit` and its models).
They are meant to be used by higher‑level project save/load code.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for static type checking to avoid circular imports at
    # runtime (chisurf.fitting.fit -> chisurf.project.fit_state -> Fit).
    from chisurf.fitting.fit import Fit


def _model_to_state(model: Any) -> Dict[str, Any]:
    """Extract a JSON‑serializable snapshot of a model's state.

    This is the core implementation used by :func:`fit_to_state` as well as
    :meth:`chisurf.models.model.Model.get_state`. It operates directly on a
    model instance without requiring a full :class:`Fit` wrapper.
    """

    try:
        find_params = getattr(model, "find_parameters", None)
        if callable(find_params):
            find_params()
    except Exception:
        pass

    params = getattr(model, "parameters_all_dict", {}) or {}

    # First pass: basic scalar attributes
    param_states: Dict[str, Dict[str, Any]] = {}
    for name, p in params.items():
        # Bounds may be numpy arrays; normalize to a simple [lb, ub] list
        try:
            bounds = p.bounds
            lb = float(bounds[0])
            ub = float(bounds[1])
        except Exception:
            lb, ub = float("-inf"), float("inf")

        try:
            value = float(p.value)
        except Exception:
            # Fall back to 0.0 if the parameter cannot be coerced cleanly
            value = 0.0

        state = {
            "value": value,
            "fixed": bool(getattr(p, "fixed", False)),
            "bounds": [lb, ub],
            "bounds_on": bool(getattr(p, "bounds_on", False)),
            # link_target will be filled in a second pass
            "link_target": None,
        }
        param_states[name] = state

    # Second pass: resolve links within this model by parameter *name*
    # We key by object id to detect internal links only.
    obj_to_name = {id(p): name for name, p in params.items()}
    for name, p in params.items():
        link = getattr(p, "link", None)
        if link is None:
            continue
        target_name = obj_to_name.get(id(link))
        if target_name is not None:
            param_states[name]["link_target"] = target_name

    # Optional model-specific extras. These are deliberately small and
    # JSON-friendly. Structural information such as component counts is
    # provided by model-specific get_state/set_state overrides; the generic
    # helper here only wires through sub-group state for TCSPC models.
    extra: Dict[str, Any] = {}

    # TCSPC-specific extras (e.g. IRF, linearization state). We delegate to
    # small get_state helpers on the corresponding sub-groups if available.
    tcspc_state: Dict[str, Any] = {}
    for key in ("generic", "corrections", "convolve"):
        comp = getattr(model, key, None)
        get_state = getattr(comp, "get_state", None) if comp is not None else None
        if callable(get_state):
            try:
                sub_state = get_state()
            except Exception:
                sub_state = None
            if isinstance(sub_state, dict) and sub_state:
                tcspc_state[key] = sub_state
    if tcspc_state:
        extra["tcspc"] = tcspc_state

    state: Dict[str, Any] = {
        "model_module": type(model).__module__,
        "model_class": type(model).__name__,
        "parameters": param_states,
    }
    if extra:
        state["extra"] = extra
    return state


def fit_to_state(fit: Fit) -> Dict[str, Any]:
    """Extract a JSON‑serializable snapshot of a single fit's model state.

    This thin wrapper forwards to :func:`_model_to_state` using
    ``fit.model``. It is kept for backwards-compatibility with existing
    callers that work at the :class:`Fit` level.
    """

    return _model_to_state(fit.model)


def _apply_state_to_model(model: Any, state: Dict[str, Any]) -> None:
    """Apply a previously captured state dictionary to a model instance.

    This is the core implementation used by :func:`apply_state_to_fit` as
    well as :meth:`chisurf.models.model.Model.set_state`. It assumes that
    ``model`` is already an instance of the desired class and only updates
    parameters, links and small structural extras (e.g. component counts).
    """

    params = getattr(model, "parameters_all_dict", {}) or {}
    stored_params: Dict[str, Dict[str, Any]] = state.get("parameters", {}) or {}

    # Optional structural extras (e.g. component counts for dynamic groups)
    # are interpreted by model-specific get_state/set_state overrides.
    extra: Dict[str, Any] = state.get("extra", {}) or {}

    # Refresh parameter layout if the model supports it so that
    # parameters_all_dict is up to date before applying scalar state.
    try:
        find_params = getattr(model, "find_parameters", None)
        if callable(find_params):
            find_params()
            params = getattr(model, "parameters_all_dict", {}) or {}
    except Exception:
        pass

    # First pass: scalar attributes (value, fixed, bounds, bounds_on)
    for name, p_state in stored_params.items():
        p = params.get(name)
        if p is None:
            # Parameter not present in this model; skip gracefully
            continue

        if "value" in p_state:
            try:
                p.value = float(p_state["value"])
            except Exception:
                # Ignore value assignment errors; model may compute value itself
                pass

        if "fixed" in p_state:
            try:
                p.fixed = bool(p_state["fixed"])
            except Exception:
                pass

        if "bounds" in p_state:
            b = p_state["bounds"]
            if isinstance(b, (list, tuple)) and len(b) == 2:
                try:
                    p.bounds = (float(b[0]), float(b[1]))
                except Exception:
                    pass

        if "bounds_on" in p_state:
            try:
                p.bounds_on = bool(p_state["bounds_on"])
            except Exception:
                pass

    # Second pass: restore intra‑fit links by name
    for name, p_state in stored_params.items():
        p = params.get(name)
        if p is None:
            continue
        target_name = p_state.get("link_target")
        if not target_name:
            # Explicitly clear existing links if any
            try:
                p.link = None
            except Exception:
                pass
            continue
        target = params.get(target_name)
        if target is not None:
            try:
                p.link = target
            except Exception:
                # If linking fails, leave parameter unlinked
                pass

    # TCSPC-specific extras (e.g. IRF and linearization state) restored via
    # dedicated set_state helpers, if present on the model's sub-groups.
    tcspc_state = extra.get("tcspc")
    if isinstance(tcspc_state, dict):
        for key, sub in tcspc_state.items():
            comp = getattr(model, key, None)
            if comp is None or not isinstance(sub, dict):
                continue
            set_state = getattr(comp, "set_state", None)
            if callable(set_state):
                try:
                    set_state(sub)
                except Exception:
                    # Never let TCSPC extras break overall fit restoration
                    pass


def apply_state_to_fit(fit: Fit, state: Dict[str, Any]) -> None:
    """Apply a previously captured state dictionary to a :class:`Fit`.

    This wrapper simply forwards to :func:`_apply_state_to_model` using
    ``fit.model`` and is kept for backwards-compatibility with existing
    callers that work at the :class:`Fit` level.
    """

    _apply_state_to_model(fit.model, state)


def make_fit_record(
        fit_id: str,
        fit: Fit,
        dataset_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a serializable record for a single fit.

    The resulting dictionary is suitable for storage in ``Project.fits``
    under the given ``fit_id``. It contains lightweight references to the
    dataset / experiment and the full parameter snapshot from
    :func:`fit_to_state`.
    """

    return {
        "fit_id": str(fit_id),
        "dataset_id": dataset_id,
        "experiment_id": experiment_id,
        "fit_state": fit_to_state(fit),
    }


def apply_fit_record(fit: Fit, record: Dict[str, Any]) -> None:
    """Apply a previously stored fit record to a :class:`Fit` instance.

    This is a thin wrapper around :func:`apply_state_to_fit` that expects
    the structure produced by :func:`make_fit_record`.
    """

    state = record.get("fit_state") or {}
    if not isinstance(state, dict):
        return
    apply_state_to_fit(fit, state)


def global_links_to_state(global_model: Any) -> Dict[str, Any]:
    """Serialize cross-fit links from a :class:`GlobalFitModel`-like object.

    The function inspects the ``links`` attribute, which is expected to be a
    list of ``[enabled, origin_fit_index, origin_param_name, formula]``
    entries as used by :class:`chisurf.models.global_model.GlobalFitModel`.
    It returns a JSON-serializable dictionary containing a normalized list
    of link records.
    """

    raw_links = getattr(global_model, "links", [])
    records: list[Dict[str, Any]] = []

    if isinstance(raw_links, (list, tuple)):
        for entry in raw_links:
            if not isinstance(entry, (list, tuple)) or len(entry) < 4:
                continue
            en, origin_fit_index, origin_param_name, formula = entry[:4]
            try:
                rec = {
                    "enabled": bool(en),
                    "origin_fit_index": int(origin_fit_index),
                    "origin_param_name": str(origin_param_name),
                    "formula": str(formula),
                }
            except Exception:
                continue
            records.append(rec)

    return {"links": records}


def apply_global_links_state(global_model: Any, state: Dict[str, Any]) -> None:
    """Restore cross-fit links on a :class:`GlobalFitModel`-like object.

    The input ``state`` must be a dictionary produced by
    :func:`global_links_to_state`. After assigning the reconstructed list to
    ``global_model.links``, the helper attempts to call ``setLinks()`` so the
    actual parameter :attr:`link` relationships are re-established.
    """

    raw_records = state.get("links") or []
    if not isinstance(raw_records, list):
        return

    encoded_links = []
    for rec in raw_records:
        if not isinstance(rec, dict):
            continue
        try:
            en = bool(rec.get("enabled", True))
            origin_fit_index = int(rec.get("origin_fit_index", 0))
            origin_param_name = str(rec.get("origin_param_name", ""))
            formula = str(rec.get("formula", ""))
        except Exception:
            continue
        encoded_links.append([en, origin_fit_index, origin_param_name, formula])

    try:
        setattr(global_model, "links", encoded_links)
    except Exception:
        return

    try:
        set_links = getattr(global_model, "setLinks", None)
        if callable(set_links):
            set_links()
    except Exception:
        pass
