from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def normalize_node_type(node_type: str) -> str:
    """Normalize equivalent node type names to avoid synonym issues."""
    t = node_type.lower()
    if t in ("analysis_parameter", "parameter"):
        return "parameter"
    operation_types = {
        "operation", "processing_run", "analysis_run", "measurement_import",
        "validation", "burst_selection", "filtering", "fcs_correlation",
        "microtime_histogram", "tcspc_fitting", "model_fitting",
        "ndxplorer_selection", "ndxplorer_clustering", "project_snapshot",
        "project_restore", "archive_export", "import", "burst_filtering",
        "gmm_fitting", "analysis", "fitting", "project_archive",
        "local_fit", "global_fit", "project", "decay_fit"
    }
    if t in operation_types:
        return "operation"
    return "artifact"


def map_legacy_node_type(kind: str) -> str:
    """Return the legacy display node type for a canonical or legacy type.

    Parameters
    ----------
    kind : str
        Node type or artifact kind.

    Returns
    -------
    str
        Legacy display node type.
    """
    if kind in ("analysis_parameter", "parameter"):
        return "analysis_parameter"
    norm = normalize_node_type(kind)
    if norm == "artifact":
        return "raw_data" if kind == "raw_data" else "processed_data"
    else:
        if kind in ("local_fit", "global_fit", "analysis", "analysis_run"):
            return "analysis_run"
        return "processing_run"


def _canonical_node_type(node_type: str) -> str:
    normalized = normalize_node_type(node_type)
    if normalized == "operation":
        return "operation"
    if normalized == "parameter":
        return "parameter"
    return "artifact"


def _annotate_canonical_edge(
    edge: dict[str, Any],
    include_legacy: bool = True,
) -> dict[str, Any]:
    source_legacy = edge["source_node_type"]
    target_legacy = edge["target_node_type"]
    source_node_type = _canonical_node_type(source_legacy)
    target_node_type = _canonical_node_type(target_legacy)

    if include_legacy:
        edge["legacy_source_node_type"] = source_legacy
        edge["legacy_target_node_type"] = target_legacy
    if source_node_type == "artifact":
        edge["source_artifact_kind"] = source_legacy
    if target_node_type == "artifact":
        edge["target_artifact_kind"] = target_legacy
    if source_node_type == "operation":
        edge["source_operation_type"] = source_legacy
    if target_node_type == "operation":
        edge["target_operation_type"] = target_legacy

    edge["source_node_type"] = source_node_type
    edge["target_node_type"] = target_node_type
    return edge


def _edge(
    edge_id: str,
    source_node_type: str,
    source_node_id: str,
    target_node_type: str,
    target_node_id: str,
    relationship_type: str,
    metadata: dict[str, Any] | None = None,
    canonical: bool = True,
) -> dict[str, Any]:
    edge = {
        "edge_id": edge_id,
        "source_node_type": source_node_type,
        "source_node_id": source_node_id,
        "target_node_type": target_node_type,
        "target_node_id": target_node_id,
        "relationship_type": relationship_type,
        "metadata": metadata or {},
    }
    return _annotate_canonical_edge(edge, include_legacy=True) if canonical else edge


def get_canonical_neighbors(
    conn: sqlite3.Connection,
    node_type: str,
    node_id: str,
    direction: str = "upstream",
    canonical: bool = True,
) -> List[Dict[str, Any]]:
    """Get direct neighbor edges and nodes in the specified direction for canonical MFDB tables.

    Parameters
    ----------
    conn : sqlite3.Connection
        The SQLite database connection.
    node_type : str
        The node type.
    node_id : str
        The node identifier.
    direction : str
        Traveral direction, either 'upstream' or 'downstream'.
    canonical : bool, default=False
        Whether to return canonical node types instead of legacy display names.

    Returns
    -------
    List[Dict[str, Any]]
        List of edge dictionaries.
    """
    edges: List[Dict[str, Any]] = []
    norm_type = normalize_node_type(node_type)

    if direction == "upstream":
        if norm_type == "artifact":
            # Find producing operations (target is artifact, source is operation)
            query = """
                SELECT operation_id, role, ordinal, checksum_snapshot, metadata_json
                FROM mfdb_operation_artifact
                WHERE artifact_id = ? AND direction = 'output'
            """
            rows = conn.execute(query, (node_id,)).fetchall()
            for r in rows:
                op_type = "operation"
                op_row = conn.execute("SELECT operation_type FROM mfdb_operation WHERE operation_id = ?", (r["operation_id"],)).fetchone()
                if op_row:
                    op_type = op_row["operation_type"]
                edges.append(_edge(
                    f"op_art_{r['operation_id']}_{node_id}_output",
                    op_type,
                    r["operation_id"],
                    node_type,
                    node_id,
                    "produced",
                    json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                    canonical=canonical,
                ))
        elif norm_type == "operation":
            # Find input artifacts (target is operation, source is artifact)
            query = """
                SELECT artifact_id, role, ordinal, checksum_snapshot, metadata_json
                FROM mfdb_operation_artifact
                WHERE operation_id = ? AND direction = 'input'
            """
            rows = conn.execute(query, (node_id,)).fetchall()
            for r in rows:
                art_type = "artifact"
                art_row = conn.execute("SELECT artifact_kind FROM mfdb_artifact WHERE artifact_id = ?", (r["artifact_id"],)).fetchone()
                if art_row:
                    art_type = art_row["artifact_kind"]
                edges.append(_edge(
                    f"op_art_{node_id}_{r['artifact_id']}_input",
                    art_type,
                    r["artifact_id"],
                    node_type,
                    node_id,
                    "input_to",
                    json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                    canonical=canonical,
                ))

        # Search mfdb_edge for non-operation relationships only.
        # Operation input/output (produced, input_to) is already covered by
        # mfdb_operation_artifact above — including them from mfdb_edge would
        # produce duplicate semantic edges for dual-written records.
        op_rel_types = ("produced", "input_to")
        type_cond = "target_node_type = ?"
        params = [node_type]
        if norm_type == "artifact":
            type_cond = "target_node_type IN ('artifact', 'processed_data', 'raw_data')"
            params = []
        elif norm_type == "operation":
            type_cond = "target_node_type IN ('operation', 'processing_run', 'analysis_run')"
            params = []
        elif norm_type == "parameter":
            type_cond = "target_node_type IN ('analysis_parameter', 'parameter')"
            params = []

        placeholders = ",".join("?" for _ in op_rel_types)
        query = f"""
            SELECT * FROM mfdb_edge
            WHERE {type_cond} AND target_node_id = ?
              AND relationship_type NOT IN ({placeholders})
        """
        rows = conn.execute(query, params + [node_id] + list(op_rel_types)).fetchall()
        for r in rows:
            edges.append(_edge(
                r["edge_id"],
                r["source_node_type"],
                r["source_node_id"],
                r["target_node_type"],
                r["target_node_id"],
                r["relationship_type"],
                json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                canonical=canonical,
            ))

    else:  # downstream
        if norm_type == "artifact":
            # Find consuming operations (source is artifact, target is operation)
            query = """
                SELECT operation_id, role, ordinal, checksum_snapshot, metadata_json
                FROM mfdb_operation_artifact
                WHERE artifact_id = ? AND direction = 'input'
            """
            rows = conn.execute(query, (node_id,)).fetchall()
            for r in rows:
                op_type = "operation"
                op_row = conn.execute("SELECT operation_type FROM mfdb_operation WHERE operation_id = ?", (r["operation_id"],)).fetchone()
                if op_row:
                    op_type = op_row["operation_type"]
                edges.append(_edge(
                    f"op_art_{r['operation_id']}_{node_id}_input",
                    node_type,
                    node_id,
                    op_type,
                    r["operation_id"],
                    "input_to",
                    json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                    canonical=canonical,
                ))
        elif norm_type == "operation":
            # Find produced artifacts (source is operation, target is artifact)
            query = """
                SELECT artifact_id, role, ordinal, checksum_snapshot, metadata_json
                FROM mfdb_operation_artifact
                WHERE operation_id = ? AND direction = 'output'
            """
            rows = conn.execute(query, (node_id,)).fetchall()
            op_type = "operation"
            op_row = conn.execute("SELECT operation_type FROM mfdb_operation WHERE operation_id = ?", (node_id,)).fetchone()
            if op_row:
                op_type = op_row["operation_type"]
            for r in rows:
                art_type = "artifact"
                art_row = conn.execute("SELECT artifact_kind FROM mfdb_artifact WHERE artifact_id = ?", (r["artifact_id"],)).fetchone()
                if art_row:
                    art_type = art_row["artifact_kind"]
                edges.append(_edge(
                    f"op_art_{node_id}_{r['artifact_id']}_output",
                    op_type,
                    node_id,
                    art_type,
                    r["artifact_id"],
                    "produced",
                    json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                    canonical=canonical,
                ))

        # Search mfdb_edge for non-operation relationships only
        op_rel_types = ("produced", "input_to")
        type_cond = "source_node_type = ?"
        params = [node_type]
        if norm_type == "artifact":
            type_cond = "source_node_type IN ('artifact', 'processed_data', 'raw_data')"
            params = []
        elif norm_type == "operation":
            type_cond = "source_node_type IN ('operation', 'processing_run', 'analysis_run')"
            params = []
        elif norm_type == "parameter":
            type_cond = "source_node_type IN ('analysis_parameter', 'parameter')"
            params = []

        placeholders = ",".join("?" for _ in op_rel_types)
        query = f"""
            SELECT * FROM mfdb_edge
            WHERE {type_cond} AND source_node_id = ?
              AND relationship_type NOT IN ({placeholders})
        """
        rows = conn.execute(query, params + [node_id] + list(op_rel_types)).fetchall()
        for r in rows:
            edges.append(_edge(
                r["edge_id"],
                r["source_node_type"],
                r["source_node_id"],
                r["target_node_type"],
                r["target_node_id"],
                r["relationship_type"],
                json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                canonical=canonical,
            ))

    return edges


def traverse_canonical_graph(
    conn: sqlite3.Connection,
    start_node_type: str,
    start_node_id: str,
    direction: str = "upstream",
    max_depth: int = 100,
    canonical: bool = True,
) -> List[Dict[str, Any]]:
    """Traverse the canonical graph recursively, cycle-safe.

    Parameters
    ----------
    conn : sqlite3.Connection
        The database connection.
    start_node_type : str
        Type of start node.
    start_node_id : str
        Identifier of start node.
    direction : str
        Traversal direction ('upstream' or 'downstream').
    max_depth : int
        Maximum recursion depth.

    Returns
    -------
    List[Dict[str, Any]]
        List of edges visited.
    """
    visited: Set[Tuple[str, str]] = set()
    edges: List[Dict[str, Any]] = []

    queue: List[Tuple[str, str, int]] = [(start_node_type, start_node_id, 0)]
    visited.add((normalize_node_type(start_node_type), start_node_id))

    while queue:
        node_type, node_id, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        neighbors = get_canonical_neighbors(conn, node_type, node_id, direction, canonical=canonical)
        for edge in neighbors:
            edges.append(edge)

            if direction == "upstream":
                next_type = edge["source_node_type"]
                next_id = edge["source_node_id"]
            else:
                next_type = edge["target_node_type"]
                next_id = edge["target_node_id"]

            norm_next = (normalize_node_type(next_type), next_id)
            if norm_next not in visited:
                visited.add(norm_next)
                queue.append((next_type, next_id, depth + 1))

    return edges


def traverse_legacy_provenance_graph(
    conn: sqlite3.Connection,
    start_node_type: str,
    start_node_id: str,
    direction: str = "upstream",
    max_depth: int = 100,
) -> List[sqlite3.Row]:
    """Perform a cycle-safe recursive search for legacy provenance edges.

    Parameters
    ----------
    conn : sqlite3.Connection
        The database connection.
    start_node_type : str
        Type of the starting node.
    start_node_id : str
        Identifier of the starting node.
    direction : str
        Direction of traversal ('upstream' or 'downstream').
    max_depth : int
        Recursion safety depth limit.

    Returns
    -------
    List[sqlite3.Row]
        List of matching legacy fdb_provenance_edge rows.  Returns empty
        list if the legacy table has been dropped.
    """
    try:
        table_check = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fdb_provenance_edge'"
        ).fetchone()
        if not table_check:
            return []
    except sqlite3.OperationalError:
        return []

    visited: Set[Tuple[str, str]] = set()
    edges: List[sqlite3.Row] = []

    def legacy_norm_type(t: str) -> str:
        t_low = t.lower()
        if t_low in ("processed_data", "raw_data", "artifact"):
            return "processed_data"
        if t_low in ("processing_run", "analysis_run", "operation"):
            return "processing_run"
        return t_low

    queue: List[Tuple[str, str, int]] = [(start_node_type, start_node_id, 0)]
    visited.add((legacy_norm_type(start_node_type), start_node_id))

    while queue:
        node_type, node_id, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        type_list = [node_type]
        norm = legacy_norm_type(node_type)
        if norm == "processed_data":
            type_list = ["processed_data", "raw_data", "artifact"]
        elif norm == "processing_run":
            type_list = ["processing_run", "analysis_run", "operation"]

        placeholders = ",".join("?" for _ in type_list)
        if direction == "upstream":
            query = f"""
                SELECT * FROM fdb_provenance_edge
                WHERE target_node_type IN ({placeholders}) AND target_node_id = ?
            """
            params = type_list + [node_id]
        else:
            query = f"""
                SELECT * FROM fdb_provenance_edge
                WHERE source_node_type IN ({placeholders}) AND source_node_id = ?
            """
            params = type_list + [node_id]

        rows = conn.execute(query, params).fetchall()
        for r in rows:
            edges.append(r)
            if direction == "upstream":
                next_type = r["source_node_type"]
                next_id = r["source_node_id"]
            else:
                next_type = r["target_node_type"]
                next_id = r["target_node_id"]

            norm_next = (legacy_norm_type(next_type), next_id)
            if norm_next not in visited:
                visited.add(norm_next)
                queue.append((next_type, next_id, depth + 1))

    return edges
