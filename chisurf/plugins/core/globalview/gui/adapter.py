from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

from chisurf.plugins.core.globalview.api.graph import build_graph, GraphResult


NODE_COLORS = {
    0: [0, 0, 128, 255],    # fit
    1: [0, 128, 0, 128],    # parameter fixed
    2: [0, 128, 0, 255],    # parameter linked
    3: [128, 0, 128, 255],  # parameter free
}


def compute_node_types(
    result: GraphResult,
    include_fixed: bool = True,
) -> List[int]:
    """Map each node to its type code for color-coding.

    Returns
    -------
    list of int
        0=fit, 1=fixed parameter, 2=linked parameter, 3=free parameter.
    """
    types = []
    for n in result.nodes:
        if n.node_type == "fit":
            types.append(0)
        else:
            if n.fixed:
                types.append(1)
            elif n.is_linked:
                types.append(2)
            else:
                types.append(3)
    return types


def graph_result_to_networkx(result: GraphResult) -> nx.Graph:
    """Convert a GraphResult to a networkx Graph for layout computation."""
    G = nx.Graph()
    for n in result.nodes:
        G.add_node(n.node_idx, **{
            "node.idx": n.node_idx,
            "node.type": n.node_type,
            "node.name": n.name,
            "fit.idx": n.fit_idx,
            "value": n.value,
            "fixed": n.fixed,
            "name": n.name,
        })
    for e in result.edges:
        G.add_edge(e.source, e.target)
    return G


def compute_layout(G: nx.Graph, layout: str = "kamada_kawai", scale: float = 1.0) -> Dict[int, Any]:
    """Compute node positions using the given layout algorithm."""
    if layout == "shell":
        return nx.shell_layout(G, scale=scale)
    elif layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G, scale=scale)
    elif layout == "planar":
        return nx.planar_layout(G, scale=scale)
    elif layout == "arf":
        return nx.arf_layout(G, etol=1e-9, dt=0.01)
    elif layout == "spectral":
        return nx.spectral_layout(G, scale=scale)
    else:
        return nx.spring_layout(G, iterations=500, scale=scale)
