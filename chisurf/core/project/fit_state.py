from __future__ import annotations
import chisurf as cs

"""Utilities for serializing and restoring fit/model parameter state.

These helpers are intentionally GUI‑independent and operate purely on the
core fitting objects (:class:`cs.core.fitting.fit.Fit` and its models).
They are meant to be used by higher‑level project save/load code.
"""

import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for static type checking to avoid circular imports at
    # runtime (cs.core.fitting.fit -> cs.core.project.fit_state -> Fit).
    from chisurf.core.fitting.fit import Fit


def _model_to_state(model: Any) -> Dict[str, Any]:
    """Extract a JSON‑serializable snapshot of a model's state.

    Version 4: parameters are keyed by UID, and links are resolved by UID.
    This is the core implementation used by :func:`fit_to_state` as well as
    :meth:`cs.core.models.model.Model.get_state`. It operates directly on a
    model instance without requiring a full :class:`Fit` wrapper.
    """

    try:
        find_params = getattr(model, "find_parameters", None)
        if callable(find_params):
            find_params()
    except Exception:
        pass

    params = getattr(model, "parameters_all_dict", {}) or {}

    # First pass: basic scalar attributes, keyed by UID
    param_states: Dict[str, Dict[str, Any]] = {}
    uid_to_obj: Dict[str, Any] = {}
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

        uid = str(getattr(p, "unique_identifier", "")) or str(uuid.uuid4())
        uid_to_obj[uid] = p

        state = {
            "uid": uid,
            "name": name,
            "value": value,
            "fixed": bool(getattr(p, "fixed", False)),
            "bounds": [lb, ub],
            "bounds_on": bool(getattr(p, "bounds_on", False)),
            # link_target will be filled in a second pass
            "link_target": None,
            "link_target_fit_uid": None,
        }
        param_states[uid] = state

    # Second pass: resolve links by UID
    for uid, p_state in param_states.items():
        p = uid_to_obj.get(uid)
        if p is None:
            continue
        link = getattr(p, "link", None)
        if link is None:
            continue
        
        target_uid = str(getattr(link, "unique_identifier", ""))
        
        # Intra-fit link discovery
        if target_uid in uid_to_obj:
            p_state["link_target"] = target_uid
        else:
            # Inter-fit link discovery (cross-fit)
            for other_fit in getattr(cs, "fits", []):
                for op in getattr(other_fit.model, "parameters_all", []):
                    if str(getattr(op, "unique_identifier", "")) == target_uid:
                        p_state["link_target"] = target_uid
                        p_state["link_target_fit_uid"] = str(getattr(other_fit, "unique_identifier", ""))
                        break
                else:
                    continue
                break

    # Optional model-specific extras. These are deliberately small and
    # JSON-friendly. Structural information such as component counts is
    # provided by model-specific get_state/set_state overrides; the generic
    # helper here only wires through sub-group state for TCSPC models.
    extra: Dict[str, Any] = {}

    # TCSPC-specific extras (e.g. IRF, background, linearization state). 
    # We delegate to small get_state helpers on the corresponding sub-groups 
    # if available, and also capture UIDs for external curve dependencies.
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

    # Explicitly capture UIDs for background and IRF if present.
    # These are used for reattachment during project loading.
    try:
        generic = getattr(model, "generic", None)
        bg_curve = getattr(generic, "background_curve", None)
        if bg_curve is not None:
            bg_uid = getattr(bg_curve, "unique_identifier", None)
            if bg_uid:
                if "generic" not in tcspc_state:
                    tcspc_state["generic"] = {}
                tcspc_state["generic"]["background_curve_uid"] = str(bg_uid)
    except Exception:
        pass

    try:
        convolve = getattr(model, "convolve", None)
        # Prefer the original IRF (_irf) over the processed one (irf)
        # to ensure we capture the correct UID.
        irf_curve = getattr(convolve, "_irf", None)
        if irf_curve is None:
            irf_curve = getattr(convolve, "irf", None)
        # Fallback for name mangling if internal attribute is used
        if irf_curve is None:
            irf_curve = getattr(convolve, "_Convolve__irf", None)
        
        if irf_curve is not None:
            irf_uid = getattr(irf_curve, "unique_identifier", None)
            if irf_uid:
                if "convolve" not in tcspc_state:
                    tcspc_state["convolve"] = {}
                tcspc_state["convolve"]["irf_uid"] = str(irf_uid)
    except Exception:
        pass

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

    Version 4: parameters are restored by UID. The stored ``parameters``
    dict is keyed by UID, and ``link_target`` values are UIDs (not names).

    This is the core implementation used by :func:`apply_state_to_fit` as
    well as :meth:`cs.core.models.model.Model.set_state`. It assumes that
    ``model`` is already an instance of the desired class and only updates
    parameters, links and small structural extras (e.g. component counts).
    """

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
    except Exception:
        pass

    # Build lookup dicts by UID (primary) and by name (fallback)
    all_params = getattr(model, "parameters_all", []) or []
    uid_to_param: Dict[str, Any] = {
        str(getattr(p, "unique_identifier", "")): p
        for p in all_params
    }
    name_to_param: Dict[str, Any] = getattr(model, "parameters_all_dict", {}) or {}

    # First pass: scalar attributes (value, fixed, bounds, bounds_on)
    for uid, p_state in stored_params.items():
        p = uid_to_param.get(uid)
        if p is None:
            p = name_to_param.get(p_state.get("name", ""))
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

    # Second pass: restore links by UID
    for uid, p_state in stored_params.items():
        p = uid_to_param.get(uid) or name_to_param.get(p_state.get("name", ""))
        if p is None:
            continue
        target_uid = p_state.get("link_target")
        target_fit_uid = p_state.get("link_target_fit_uid")
        
        if not target_uid:
            # Explicitly clear existing links if any
            try:
                p.link = None
            except Exception:
                pass
            continue
            
        if not target_fit_uid:
            # Intra-fit link restoration by UID
            target = uid_to_param.get(target_uid)
            if target is not None:
                try:
                    p.link = target
                except Exception:
                    pass
        else:
            # Inter-fit link restoration by UID
            target_fit = next((f for f in getattr(cs, "fits", []) 
                               if str(getattr(f, "unique_identifier", "")) == target_fit_uid), None)
            if target_fit:
                target_params = getattr(target_fit.model, "parameters_all", []) or []
                target = next(
                    (op for op in target_params
                     if str(getattr(op, "unique_identifier", "")) == target_uid),
                    None
                )
                if target:
                    try:
                        p.link = target
                    except Exception:
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

        # UID Reattachment Pass for TCSPC components
        datasets = getattr(cs, "imported_datasets", [])
        
        # 1. Background curve reattachment
        bg_uid = tcspc_state.get("generic", {}).get("background_curve_uid")
        if bg_uid:
            target_bg = next((d for d in datasets if getattr(d, "unique_identifier", None) == bg_uid), None)
            if target_bg:
                generic = getattr(model, "generic", None)
                if generic is not None:
                    try:
                        generic.background_curve = target_bg
                    except Exception:
                        pass

        # 2. IRF reattachment
        irf_uid = tcspc_state.get("convolve", {}).get("irf_uid")
        if irf_uid:
            target_irf = next((d for d in datasets if getattr(d, "unique_identifier", None) == irf_uid), None)
            if target_irf:
                convolve = getattr(model, "convolve", None)
                if convolve is not None:
                    try:
                        convolve._irf = target_irf
                    except Exception:
                        pass


def apply_state_to_fit(
    fit: Fit,
    state: Dict[str, Any],
    dependency_edges: list[dict[str, Any]] | None = None,
    fit_record_id: str = "",
) -> None:
    """Apply a previously captured state dictionary to a :class:`Fit`.

    This wrapper simply forwards to :func:`_apply_state_to_model` using
    ``fit.model`` and is kept for backwards-compatibility with existing
    callers that work at the :class:`Fit` level.

    Parameters
    ----------
    fit : Fit
        The fit to apply the state to.
    state : Dict[str, Any]
        The state dictionary from :func:`fit_to_state` or similar.
    dependency_edges : list[dict], optional
        MFDB-sourced dependency edges for parameter links. These are used
        to re-establish parameter links in the correct order, handling
        the case where the linked-to fit must exist before the linking fit.
    fit_record_id : str, optional
        The fit record ID from the archiver. Used to match against operation_id.
    """

    _apply_state_to_model(fit.model, state)

    if dependency_edges:
        _restore_parameter_links_from_edges(fit.model, dependency_edges, fit_record_id)


def _restore_parameter_links_from_edges(
    model: Any,
    dependency_edges: list[dict[str, Any]],
    fit_record_id: str = "",
) -> None:
    """Restore parameter links from MFDB dependency edges.

    This function filters dependency_edges to those belonging to the current
    fit's operation and uses them to establish parameter links in the correct
    order. This ensures that linked-to fits exist before the linking fit is
    processed.

    Parameters
    ----------
    model : Any
        The model instance whose parameters should have links restored.
    dependency_edges : list[dict]
        List of dependency edges from MFDB. Each edge should have:
        - source_node_id: UID of the parameter being linked TO
        - target_node_id: UID of the parameter that has the link
        - operation_id: The operation ID for this fit
    fit_record_id : str, optional
        The fit record ID from the archiver. Used to match against operation_id.
    """
    # Filter edges to those belonging to this fit
    # Operation IDs are in the format: fit_{version_id}:{fit_record_id}:{lf_id}
    # The fit_record_id in the operation_id matches the archiver's fit record ID
    edges_for_fit = []
    for edge in dependency_edges:
        op_id = edge.get("operation_id", "")
        # Check if operation_id belongs to this version and fit
        # The operation_id format is: fit_{version_id}:{fit_record_id}:{lf_id}
        parts = op_id.split(":")
        if len(parts) >= 3 and parts[0].startswith("fit_") and parts[1] == fit_record_id:
            edges_for_fit.append(edge)

    if not edges_for_fit:
        return

    # Build lookup dict by parameter UID
    all_params = getattr(model, "parameters_all", []) or []
    uid_to_param: dict[str, Any] = {
        str(getattr(p, "unique_identifier", "")): p
        for p in all_params
    }

    # Restore links from dependency edges
    # Edge structure: source_node_id (target_uid) -> target_node_id (uid)
    # This means uid links TO target_uid
    for edge in edges_for_fit:
        source_uid = edge.get("source_node_id")
        target_uid = edge.get("target_node_id")

        if not source_uid or not target_uid:
            continue

        source_param = uid_to_param.get(source_uid)
        target_param = uid_to_param.get(target_uid)

        if source_param and target_param:
            try:
                target_param.link = source_param
            except Exception:
                pass


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


