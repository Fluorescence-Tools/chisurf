import os
from typing import Any, Dict, List, Tuple


def node_key(node_type: str, node_id: str) -> str:
    """Return stable node editor ID."""
    return f"{node_type}:{node_id}"

def record_kind(node: Dict[str, Any]) -> str:
    """Return the node type / kind of record."""
    return node.get("node_type") or ""

def record_title(node: Dict[str, Any]) -> str:
    """Return formatted node title according to node type."""
    nt = node.get("node_type") or ""
    nid = node.get("node_id") or ""
    rec = node.get("record") or node

    if nt == "raw_data":
        dt = rec.get("data_type")
        if not dt:
            for key in ("file_path", "url", "folder_path", "path"):
                location = rec.get(key)
                if location:
                    _, ext = os.path.splitext(location)
                    dt = ext.lstrip(".")
                    break
        return f"Raw: {dt or 'data'}"
    elif nt == "processing_run":
        pt = rec.get("type") or rec.get("processing_type") or "processing"
        return f"Process: {pt}"
    elif nt == "processed_data":
        pt = rec.get("product_type") or "product"
        return f"Product: {pt}"
    elif nt == "analysis_run":
        at = rec.get("analysis_type") or rec.get("model_name") or "analysis"
        return f"Analysis: {at}"
    elif nt == "analysis_parameter":
        name = rec.get("name") or "parameter"
        return f"Parameter: {name}"

    return f"{nt}: {nid}"

def relation_color(rel: str) -> List[int]:
    """Color edge config by relationship type."""
    rel = str(rel).lower()
    if rel == "input_to":
        return [70, 120, 200]      # blue
    elif rel == "produced":
        return [70, 180, 100]      # green
    elif rel == "parameter_of":
        return [200, 180, 70]      # yellow
    elif rel == "derived_from":
        return [120, 120, 120]     # gray
    else:
        return [180, 180, 180]     # light gray

def layout_nodes(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """Compute left-to-right dependency levels and grid coordinates deterministically."""
    # Find levels via longest path BFS / bellman-ford style propagation
    levels = {n["id"]: 0 for n in nodes}
    for _ in range(len(nodes)):
        changed = False
        for edge in edges:
            s = edge["source"]
            t = edge["target"]
            if s in levels and t in levels:
                if levels[t] < levels[s] + 1:
                    levels[t] = levels[s] + 1
                    changed = True
        if not changed:
            break

    # Group nodes by level
    nodes_by_level: Dict[int, List[Dict[str, Any]]] = {}
    for n in nodes:
        lvl = levels[n["id"]]
        nodes_by_level.setdefault(lvl, []).append(n)

    # Sort deterministically
    positions = {}
    for lvl in sorted(nodes_by_level.keys()):
        # Sort key: (level, kind, title, id)
        level_nodes = nodes_by_level[lvl]
        level_nodes.sort(key=lambda x: (
            lvl,
            record_kind(x),
            x.get("title", ""),
            x.get("id", "")
        ))

        for idx, n in enumerate(level_nodes):
            x = lvl * 260.0
            y = idx * 130.0
            positions[n["id"]] = (x, y)

    return positions

def mfdb_graph_to_node_editor_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw MFDB provenance export format to Node Editor graph schema."""
    if not graph:
        return {
            "version": 1,
            "meta": {
                "purpose": "provenance_view",
                "schema_name": "mfdb.provenance.node_editor.v1"
            },
            "nodes": [],
            "edges": []
        }

    raw_nodes = list(graph.get("nodes") or [])
    raw_edges = list(graph.get("edges") or [])

    nodes_dict = {}
    out_nodes = []

    def add_node(node: Dict[str, Any]) -> None:
        nt = node.get("node_type")
        nid = node.get("node_id")
        if not nt or not nid:
            return

        key = node_key(nt, nid)
        if key in nodes_dict:
            return

        title = record_title(node)
        node_entry = {
            "id": key,
            "title": title,
            "inputs": [{"name": "in", "type": "mfdb"}],
            "outputs": [{"name": "out", "type": "mfdb"}],
            "type": "mfdb_record",
            "config": {
                "record": node,
                "node_type": nt,
                "node_id": nid,
                "workflow_runtime": None
            },
            "collapsed": False,
            "z": 1.0
        }
        out_nodes.append(node_entry)
        nodes_dict[key] = node_entry

    # Process nodes
    for rn in raw_nodes:
        add_node(rn)

    if not raw_nodes:
        # Dependency responses may contain only edges; synthesize endpoint nodes
        # so the node editor can render upstream/downstream traces.
        for re in raw_edges:
            for side in ("source", "target"):
                nt = re.get(f"{side}_node_type")
                nid = re.get(f"{side}_node_id")
                if not nt or not nid:
                    continue
                key = node_key(nt, nid)
                if key not in nodes_dict:
                    record = re.get(f"{side}_record") or re.get(side) or {
                        "node_type": nt,
                        "node_id": nid
                    }
                    add_node({
                        "node_type": nt,
                        "node_id": nid,
                        "record": record
                    })

    # Process edges
    out_edges = []
    for re in raw_edges:
        s_type = re.get("source_node_type")
        s_id = re.get("source_node_id")
        t_type = re.get("target_node_type")
        t_id = re.get("target_node_id")

        if not s_type or not s_id or not t_type or not t_id:
            continue

        s_key = node_key(s_type, s_id)
        t_key = node_key(t_type, t_id)

        # Skip edge if source or target node is missing in the node set
        if s_key not in nodes_dict or t_key not in nodes_dict:
            continue

        rel = re.get("relationship_type", "")
        color = relation_color(rel)

        edge_entry = {
            "source": s_key,
            "source_port": 1,  # out port index
            "target": t_key,
            "target_port": 0,  # in port index
            "config": {
                "edge_id": re.get("edge_id"),
                "relationship_type": rel,
                "metadata": re.get("metadata"),
                "color": color
            }
        }
        out_edges.append(edge_entry)

    # Layout nodes
    pos_map = layout_nodes(out_nodes, out_edges)
    for n in out_nodes:
        n["pos"] = list(pos_map.get(n["id"], (0.0, 0.0)))

    return {
        "version": 1,
        "meta": {
            "purpose": "provenance_view",
            "schema_name": "mfdb.provenance.node_editor.v1"
        },
        "nodes": out_nodes,
        "edges": out_edges
    }
