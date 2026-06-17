import os
from typing import Any


def node_key(node_type: str, node_id: str) -> str:
    """Return stable node editor ID."""
    return f"{node_type}:{node_id}"

def record_kind(node: dict[str, Any]) -> str:
    """Return the node type / kind of record."""
    return node.get("node_type") or ""

def record_title(node: dict[str, Any]) -> str:
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

def relation_color(rel: str) -> list[int]:
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

def layout_nodes(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
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
    nodes_by_level: dict[int, list[dict[str, Any]]] = {}
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

def mfdb_graph_to_node_editor_graph(graph: dict[str, Any]) -> dict[str, Any]:
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

    def add_node(node: dict[str, Any]) -> None:
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


def mfdb_chinet_to_node_editor_graph(
    operation_id: str,
    artifacts: list[dict[str, Any]],
    parameters: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert MFDB chinet artifacts to node editor graph format.

    Creates a unified dependency graph showing source files, datasets,
    chinet sessions, nodes, fits, and parameter links.

    Parameters
    ----------
    operation_id : str
        Project version operation ID.
    artifacts : list of dict
        Artifacts from ``get_operation_artifacts()``.
    parameters : list of dict
        Parameters from ``list_project_parameters_handler()``.
    edges : list of dict
        Edges from ``get_downstream_dependencies()`` or similar.

    Returns
    -------
    dict
        Node editor graph dict with ``version``, ``meta``, ``nodes``, ``edges``.
    """
    nodes_dict: dict[str, dict[str, Any]] = {}
    out_nodes: list[dict[str, Any]] = []
    out_edges: list[dict[str, Any]] = []

    def _add_node(
        node_id: str,
        title: str,
        node_type: str,
        inputs: list[str | dict[str, str]] | None = None,
        outputs: list[str | dict[str, str]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if node_id in nodes_dict:
            return
        entry = {
            "id": node_id,
            "title": title,
            "inputs": inputs or [{"name": "in", "type": "mfdb"}],
            "outputs": outputs or [{"name": "out", "type": "mfdb"}],
            "type": node_type,
            "config": config or {},
            "collapsed": False,
            "z": 1.0,
        }
        out_nodes.append(entry)
        nodes_dict[node_id] = entry

    def _add_edge(
        source_id: str,
        target_id: str,
        relationship: str,
        source_port: int = 1,
        target_port: int = 0,
        color: list[int] | None = None,
    ) -> None:
        if source_id not in nodes_dict or target_id not in nodes_dict:
            return
        out_edges.append({
            "source": source_id,
            "source_port": source_port,
            "target": target_id,
            "target_port": target_port,
            "config": {
                "relationship_type": relationship,
                "color": color or relation_color(relationship),
            },
        })

    # Build project node
    project_node_id = f"operation:{operation_id}"
    _add_node(
        project_node_id,
        f"Project: {operation_id[:20]}",
        "project",
        inputs=[],
        outputs=[{"name": "datasets", "type": "mfdb"}, {"name": "fits", "type": "mfdb"}],
        config={"operation_id": operation_id},
    )

    # Index artifacts by kind
    source_arts = [a for a in artifacts if a.get("artifact_kind") in ("raw_measurement", "raw_data")]
    dataset_arts = [a for a in artifacts if a.get("artifact_kind") == "processed_data"]
    session_arts = [a for a in artifacts if a.get("artifact_kind") == "chinet_session"]
    node_arts = [a for a in artifacts if a.get("artifact_kind") == "chinet_node"]
    fit_arts = [a for a in artifacts if a.get("artifact_kind") == "fit_result"]

    # Add source file nodes
    source_node_map: dict[str, str] = {}
    for art in source_arts:
        aid = art.get("artifact_id", "")
        nid = f"artifact:{aid}"
        fname = (art.get("metadata") or {}).get("file_path") or art.get("file_path") or aid
        source_node_map[aid] = nid
        _add_node(
            nid,
            f"Source: {os.path.basename(fname)}",
            "source_file",
            outputs=[{"name": "data", "type": "mfdb"}],
            config={"artifact_id": aid, "format": art.get("data_format"), "size_bytes": art.get("size_bytes")},
        )

    # Add dataset nodes
    dataset_node_map: dict[str, str] = {}
    for art in dataset_arts:
        aid = art.get("artifact_id", "")
        nid = f"artifact:{aid}"
        meta = art.get("metadata") or {}
        ds_name = meta.get("name") or aid
        dataset_node_map[aid] = nid
        _add_node(
            nid,
            f"Dataset: {ds_name}",
            "dataset",
            inputs=[{"name": "source", "type": "mfdb"}],
            outputs=[{"name": "data", "type": "mfdb"}],
            config={"artifact_id": aid, "experiment": meta.get("experiment_name")},
        )
        # Link to project
        _add_edge(project_node_id, nid, "project_contains", source_port=0, target_port=0)

    # Add chinet session nodes
    session_node_map: dict[str, str] = {}
    for art in session_arts:
        aid = art.get("artifact_id", "")
        nid = f"artifact:{aid}"
        session_node_map[aid] = nid
        _add_node(
            nid,
            f"Session: {aid[:30]}",
            "chinet_session",
            inputs=[{"name": "in", "type": "mfdb"}],
            outputs=[{"name": "nodes", "type": "mfdb"}],
            config={"artifact_id": aid},
        )

    # Add chinet node artifacts
    for art in node_arts:
        aid = art.get("artifact_id", "")
        nid = f"artifact:{aid}"
        meta = art.get("metadata") or {}
        node_name = meta.get("name") or aid
        _add_node(
            nid,
            f"Node: {node_name}",
            "chinet_node",
            inputs=[{"name": "params", "type": "mfdb"}],
            outputs=[{"name": "out", "type": "mfdb"}],
            config={"artifact_id": aid, "session_id": meta.get("session_id")},
        )
        # Link to parent session
        session_id = meta.get("session_id")
        for said, snid in session_node_map.items():
            if session_id and session_id in said:
                _add_edge(snid, nid, "contains", source_port=0, target_port=0)
                break

    # Add fit result nodes
    fit_node_map: dict[str, str] = {}
    for art in fit_arts:
        aid = art.get("artifact_id", "")
        nid = f"artifact:{aid}"
        meta = art.get("metadata") or {}
        model_class = meta.get("model_class") or aid
        fit_node_map[aid] = nid
        _add_node(
            nid,
            f"Fit: {model_class}",
            "fit_result",
            inputs=[{"name": "dataset", "type": "mfdb"}, {"name": "params", "type": "mfdb"}],
            outputs=[{"name": "result", "type": "mfdb"}],
            config={"artifact_id": aid, "model_class": model_class, "model_module": meta.get("model_module")},
        )
        # Link to project
        _add_edge(project_node_id, nid, "project_contains", source_port=0, target_port=0)

    # Add parameter nodes and link edges
    param_node_map: dict[str, str] = {}
    for param in parameters:
        puuid = param.get("parameter_uuid", "")
        nid = f"parameter:{puuid}"
        param_node_map[puuid] = nid
        _add_node(
            nid,
            f"Param: {param.get('name', puuid[:12])}",
            "parameter",
            inputs=[{"name": "in", "type": "mfdb"}],
            outputs=[{"name": "value", "type": "mfdb"}],
            config={
                "parameter_uuid": puuid,
                "value": param.get("value"),
                "fixed": param.get("parameter_type") == "fixed",
                "bounds": [param.get("lower_bound"), param.get("upper_bound")],
                "link_target": param.get("link_target"),
            },
        )

    # Add dependency edges from mfdb_edge
    for edge in edges:
        src_type = edge.get("source_node_type", "")
        src_id = edge.get("source_node_id", "")
        tgt_type = edge.get("target_node_type", "")
        tgt_id = edge.get("target_node_id", "")
        rel = edge.get("relationship_type", "")

        src_key = f"{src_type}:{src_id}"
        tgt_key = f"{tgt_type}:{tgt_id}"

        if src_key in nodes_dict and tgt_key in nodes_dict:
            color = [200, 180, 70] if rel == "parameter_depends_on" else relation_color(rel)
            _add_edge(src_key, tgt_key, rel, color=color)

    # Layout
    pos_map = layout_nodes(out_nodes, out_edges)
    for n in out_nodes:
        n["pos"] = list(pos_map.get(n["id"], (0.0, 0.0)))

    return {
        "version": 1,
        "meta": {
            "purpose": "chinet_dependency_view",
            "schema_name": "mfdb.chinet.node_editor.v1",
            "operation_id": operation_id,
        },
        "nodes": out_nodes,
        "edges": out_edges,
    }


def mfdb_version_graph_to_node_editor_graph(
    project_id: str,
    versions: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert project version DAG to node editor graph format.

    Parameters
    ----------
    project_id : str
        Project identifier.
    versions : list of dict
        Version metadata dicts with ``version_id``, ``version_number``,
        ``branch_uuid``, ``created_at``, etc.
    edges : list of dict
        Version lineage edges with ``source``, ``target``, ``relationship``.

    Returns
    -------
    dict
        Node editor graph dict.
    """
    out_nodes: list[dict[str, Any]] = []
    out_edges: list[dict[str, Any]] = []
    nodes_dict: dict[str, dict[str, Any]] = {}

    for v in versions:
        vid = v.get("version_id", "")
        nid = f"version:{vid}"
        branch = v.get("branch_uuid", "")
        vn = v.get("version_number", 0)
        is_head = v.get("is_head", False)
        title = f"v{vn}"
        if branch:
            title += f" ({branch[:8]})"
        if is_head:
            title += " *"

        entry = {
            "id": nid,
            "title": title,
            "inputs": [{"name": "parent", "type": "mfdb"}],
            "outputs": [{"name": "children", "type": "mfdb"}],
            "type": "version",
            "config": {
                "version_id": vid,
                "version_number": vn,
                "branch_uuid": branch,
                "created_at": v.get("created_at"),
                "fit_count": v.get("fit_count", 0),
                "dataset_count": v.get("dataset_count", 0),
                "notes": v.get("notes", ""),
            },
            "collapsed": False,
            "z": 1.0,
        }
        out_nodes.append(entry)
        nodes_dict[nid] = entry

    for edge in edges:
        src = f"version:{edge.get('source', '')}"
        tgt = f"version:{edge.get('target', '')}"
        if src in nodes_dict and tgt in nodes_dict:
            out_edges.append({
                "source": src,
                "source_port": 1,
                "target": tgt,
                "target_port": 0,
                "config": {
                    "relationship_type": edge.get("relationship", "supersedes"),
                    "color": [70, 180, 100],  # green for version lineage
                },
            })

    pos_map = layout_nodes(out_nodes, out_edges)
    for n in out_nodes:
        n["pos"] = list(pos_map.get(n["id"], (0.0, 0.0)))

    return {
        "version": 1,
        "meta": {
            "purpose": "version_graph_view",
            "schema_name": "mfdb.version_graph.node_editor.v1",
            "project_id": project_id,
        },
        "nodes": out_nodes,
        "edges": out_edges,
    }
