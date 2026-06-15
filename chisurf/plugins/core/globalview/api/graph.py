from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GraphNode:
    node_idx: int
    node_type: str  # "fit" or "parameter"
    name: str
    fit_idx: int
    value: Optional[float] = None
    fixed: bool = False
    is_linked: bool = False
    link_name: str = ""
    fit_name: str = ""
    data_filename: str = ""
    model: str = ""


@dataclass
class GraphEdge:
    source: int
    target: int


@dataclass
class GraphResult:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def build_graph(
    fit_list: List[Any],
    include_fixed: bool = True,
    connect_fits: bool = False,
    skip_global_fit: bool = True,
) -> GraphResult:
    """Build a graph representation from fit objects.

    Parameters
    ----------
    fit_list : list
        List of fit objects (Fit or FitGroup).
    include_fixed : bool
        Whether to include fixed parameters.
    connect_fits : bool
        Whether to add edges between all fit nodes.
    skip_global_fit : bool
        Whether to skip fits with GlobalFitModel.

    Returns
    -------
    GraphResult
        Dataclass with nodes and edges.
    """
    from chisurf.core.models.global_model import GlobalFitModel

    result = GraphResult()
    node_idx = 0
    fit_node_ids: Dict[int, int] = {}

    for fi, fit in enumerate(fit_list):
        if skip_global_fit and isinstance(getattr(fit, "model", None), GlobalFitModel):
            continue

        fit_name = str(getattr(fit, "name", f"fit_{fi}"))
        try:
            data_filename = str(getattr(getattr(fit, "data", None), "filename", "") or "")
        except Exception:
            data_filename = ""
        try:
            model_full = (
                f"{getattr(fit.model.__class__, '__module__', '')}."
                f"{getattr(fit.model.__class__, '__name__', '')}"
            )
        except Exception:
            model_full = ""

        node = GraphNode(
            node_idx=node_idx,
            node_type="fit",
            name=fit_name,
            fit_idx=fi,
            fit_name=fit_name,
            data_filename=data_filename,
            model=model_full,
        )
        node_id = node_idx
        result.nodes.append(node)
        fit_node_ids[fi] = node_id
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

            param_node = GraphNode(
                node_idx=node_idx,
                node_type="parameter",
                name=param_name,
                fit_idx=fi,
                value=param_value,
                fixed=fixed,
                is_linked=is_linked,
                link_name=link_name,
            )
            param_node_id = node_idx
            result.nodes.append(param_node)
            result.edges.append(GraphEdge(source=param_node_id, target=node_id))
            node_idx += 1

    # Connect linked parameters
    for n in result.nodes:
        if n.node_type != "parameter":
            continue
        if not n.is_linked or not n.link_name:
            continue
        for m in result.nodes:
            if m.node_type != "parameter":
                continue
            if m.name == n.link_name:
                result.edges.append(GraphEdge(source=n.node_idx, target=m.node_idx))

    # Optional: connect all fit nodes
    if connect_fits:
        fit_nodes = [n for n in result.nodes if n.node_type == "fit"]
        for i, a in enumerate(fit_nodes):
            for b in fit_nodes[i + 1:]:
                result.edges.append(GraphEdge(source=a.node_idx, target=b.node_idx))

    return result
