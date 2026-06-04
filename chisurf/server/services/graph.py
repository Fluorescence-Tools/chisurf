from __future__ import annotations

from typing import Any, Dict, List, Optional

from chisurf.server.services import ServiceResult
from chisurf.server.session import SessionState


def _safe_float(val: Any) -> Optional[float]:
    """Convert *val* to float, returning ``None`` on failure.

    Parameters
    ----------
    val : any
        Value to convert.

    """
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def build_fit_graph(
    state: SessionState,
    fit_indices: Optional[List[int]] = None,
    fit_uids: Optional[List[str]] = None,
    include_fixed: bool = True,
    connect_fits: bool = False,
) -> ServiceResult:
    """Build a graph from fits on the server, returning nodes and edges as JSON.

    Mirrors the client-side ``GraphWizard.build_graph()`` logic so the
    global-view wizard can work with pure DTO data.
    """
    fits = list(state.fits)
    selected = []

    if fit_uids:
        uid_set = set(fit_uids)
        for i, f in enumerate(fits):
            uid = str(getattr(f, "unique_identifier", ""))
            if uid in uid_set:
                selected.append((i, f))
    elif fit_indices:
        for i in fit_indices:
            if 0 <= i < len(fits):
                selected.append((i, fits[i]))
    else:
        selected = list(enumerate(fits))

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_idx = 0
    fit_node_ids: Dict[int, int] = {}  # fit_index -> node_idx

    for fit_idx, fit in selected:
        fit_name = str(getattr(fit, "name", "fit"))
        try:
            data_filename = str(getattr(getattr(fit, "data", None), "filename", "") or "")
        except Exception:
            data_filename = ""
        try:
            model_module = getattr(fit.model.__class__, "__module__", "")
            model_class_name = getattr(fit.model.__class__, "__name__", "")
            model_full = f"{model_module}.{model_class_name}"
        except Exception:
            model_full = ""

        node = {
            "node_idx": node_idx,
            "node_type": "fit",
            "name": fit_name,
            "fit_idx": fit_idx,
            "data_filename": data_filename,
            "model": model_full,
        }
        node_id = node_idx
        nodes.append(node)
        fit_node_ids[fit_idx] = node_id
        node_idx += 1

        try:
            parameters = list(getattr(fit.model, "parameters_all", []) or [])
        except Exception:
            parameters = []

        for param in parameters:
            try:
                fixed = bool(getattr(param, "fixed", False))
            except Exception:
                fixed = False
            if fixed and not include_fixed:
                continue

            param_name = str(getattr(param, "name", "param"))
            param_value = _safe_float(getattr(param, "value", None))
            try:
                is_linked = bool(getattr(param, "is_linked", False))
            except Exception:
                is_linked = False
            try:
                link_name = str(getattr(getattr(param, "link", None), "name", "") or "")
            except Exception:
                link_name = ""

            param_node = {
                "node_idx": node_idx,
                "node_type": "parameter",
                "name": param_name,
                "fit_idx": fit_idx,
                "value": param_value,
                "fixed": fixed,
                "is_linked": is_linked,
                "link_name": link_name,
            }
            param_node_id = node_idx
            nodes.append(param_node)
            edges.append({"source": param_node_id, "target": node_id})
            node_idx += 1

    # Connect linked parameters
    for n in nodes:
        if n["node_type"] != "parameter":
            continue
        if not n.get("is_linked"):
            continue
        link_name = n.get("link_name", "")
        if not link_name:
            continue
        for m in nodes:
            if m["node_type"] != "parameter":
                continue
            if m["name"] == link_name and m["fit_idx"] != n["fit_idx"]:
                edges.append({"source": n["node_idx"], "target": m["node_idx"]})

    # Optional: connect all fit nodes to each other
    if connect_fits:
        fit_nodes = [n for n in nodes if n["node_type"] == "fit"]
        for i, a in enumerate(fit_nodes):
            for b in fit_nodes[i + 1:]:
                edges.append({"source": a["node_idx"], "target": b["node_idx"]})

    return {
        "ok": True,
        "graph": {
            "nodes": nodes,
            "edges": edges,
        },
    }
