"""Persist pipeline definitions and runs (PRD-22 Task 3).

A pipeline is a *saveable, shareable document* (the Orange3 ``.ows`` lesson): its
graph structure is stored in the dictionary-declared ``mfdb_pipeline`` /
``mfdb_pipeline_node`` / ``mfdb_pipeline_edge`` tables (structured, not a blob — only
the per-node parameter bag is JSON). A *run* groups the recorded operation chain via
``mfdb_pipeline_run`` / ``mfdb_pipeline_run_operation`` so an execution is queryable
as a unit.

These are thin free functions over an MFDB handle (``db.conn`` + ``db.transaction``),
keeping pipeline persistence in the pipeline package rather than the repository.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from chisurf.core.pipeline.model import Pipeline, PipelineEdge, PipelineNode
from chisurf.core.pipeline.runner import PipelineRun


def _now() -> str:
    from chisurf.core.mfdb.repository import _utc_now

    return _utc_now()


def _default_user() -> str | None:
    from chisurf.core.mfdb.session import configured_default_user_id

    return configured_default_user_id()


def _next_id(db: Any, table: str, column: str) -> int:
    """Next free integer PK for ``table.column`` (globally unique, MAX+1)."""
    return db.conn.execute(f"SELECT COALESCE(MAX({column}), 0) FROM {table}").fetchone()[0] + 1


def save_pipeline(
    db: Any,
    pipeline: Pipeline,
    *,
    is_public: bool = False,
    description: str = "",
    created_by_user_id: str | None = None,
    pipeline_id: str | None = None,
) -> str:
    """Persist a pipeline definition (header + node/edge graph); return its id.

    The definition is replaceable: saving with an existing ``pipeline_id`` soft-clears
    its prior nodes/edges first so the stored graph always matches ``pipeline``.
    """
    if created_by_user_id is None:
        created_by_user_id = _default_user()
    pid = pipeline_id or str(uuid.uuid4())
    now = _now()
    with db.transaction():
        db.conn.execute("DELETE FROM mfdb_pipeline_node WHERE pipeline_id = ?", (pid,))
        db.conn.execute("DELETE FROM mfdb_pipeline_edge WHERE pipeline_id = ?", (pid,))
        db.conn.execute("DELETE FROM mfdb_pipeline WHERE pipeline_id = ?", (pid,))
        db.conn.execute(
            "INSERT INTO mfdb_pipeline (pipeline_id, name, version, description, "
            "created_by_user_id, is_public, created_at, updated_at, deleted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, pipeline.name, pipeline.version, description or None,
             created_by_user_id, 1 if is_public else 0, now, now, None),
        )
        node_base = _next_id(db, "mfdb_pipeline_node", "node_row_id")
        for i, node in enumerate(pipeline.nodes):
            db.conn.execute(
                "INSERT INTO mfdb_pipeline_node (node_row_id, pipeline_id, node_name, "
                "operation_type, parameters_json, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (node_base + i, pid, node.name, node.operation_type,
                 json.dumps(node.parameters or {}), now, now, None),
            )
        edge_base = _next_id(db, "mfdb_pipeline_edge", "edge_row_id")
        for j, edge in enumerate(pipeline.edges):
            db.conn.execute(
                "INSERT INTO mfdb_pipeline_edge (edge_row_id, pipeline_id, source_node, "
                "source_port, target_node, target_port, created_at, updated_at, "
                "deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (edge_base + j, pid, edge.source, edge.source_port, edge.target,
                 edge.target_port, now, now, None),
            )
    return pid


def get_pipeline(db: Any, pipeline_id: str) -> Pipeline | None:
    """Reconstruct a stored :class:`Pipeline` by id, or ``None`` if absent."""
    head = db.conn.execute(
        "SELECT name, version FROM mfdb_pipeline "
        "WHERE pipeline_id = ? AND deleted_at IS NULL",
        (pipeline_id,),
    ).fetchone()
    if head is None:
        return None
    node_rows = db.conn.execute(
        "SELECT node_name, operation_type, parameters_json FROM mfdb_pipeline_node "
        "WHERE pipeline_id = ? AND deleted_at IS NULL ORDER BY node_row_id",
        (pipeline_id,),
    ).fetchall()
    edge_rows = db.conn.execute(
        "SELECT source_node, source_port, target_node, target_port "
        "FROM mfdb_pipeline_edge WHERE pipeline_id = ? AND deleted_at IS NULL "
        "ORDER BY edge_row_id",
        (pipeline_id,),
    ).fetchall()
    nodes = tuple(
        PipelineNode(r[0], r[1], json.loads(r[2]) if r[2] else {}) for r in node_rows
    )
    edges = tuple(PipelineEdge(r[0], r[1], r[2], r[3]) for r in edge_rows)
    return Pipeline(name=head[0], nodes=nodes, edges=edges, version=head[1] or "1.0")


def list_pipelines(
    db: Any, scope: str = "all", owner_id: str | None = None
) -> list[dict[str, Any]]:
    """List stored pipelines scoped ``mine``/``own`` | ``public`` | ``all``."""
    if owner_id is None and scope in ("mine", "own", "all"):
        owner_id = _default_user()
    where = ["deleted_at IS NULL"]
    params: list[Any] = []
    if scope in ("mine", "own"):
        where.append("created_by_user_id = ?")
        params.append(owner_id)
    elif scope == "public":
        where.append("is_public = 1")
    else:
        where.append("(is_public = 1 OR created_by_user_id = ?)")
        params.append(owner_id)
    rows = db.conn.execute(
        f"SELECT * FROM mfdb_pipeline WHERE {' AND '.join(where)} ORDER BY name",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def record_pipeline_run(
    db: Any,
    run: PipelineRun,
    *,
    pipeline_id: str | None = None,
    status: str = "succeeded",
    name: str | None = None,
    created_by_user_id: str | None = None,
) -> str:
    """Persist a run, grouping its operation chain; return the ``pipeline_run_id``.

    Also stamps ``run.pipeline_run_id`` so the in-memory result carries its handle.
    """
    if created_by_user_id is None:
        created_by_user_id = _default_user()
    run_id = str(uuid.uuid4())
    now = _now()
    with db.transaction():
        db.conn.execute(
            "INSERT INTO mfdb_pipeline_run (pipeline_run_id, pipeline_id, name, status, "
            "created_by_user_id, is_public, created_at, updated_at, deleted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, pipeline_id, name or run.pipeline_name, status,
             created_by_user_id, 0, now, now, None),
        )
        row_base = _next_id(db, "mfdb_pipeline_run_operation", "row_id")
        for ordinal, operation_id in enumerate(run.operation_ids):
            db.conn.execute(
                "INSERT INTO mfdb_pipeline_run_operation (row_id, pipeline_run_id, "
                "operation_id, ordinal, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (row_base + ordinal, run_id, operation_id, ordinal, now, now, None),
            )
    run.pipeline_run_id = run_id
    return run_id


def get_pipeline_run(db: Any, pipeline_run_id: str) -> dict[str, Any] | None:
    """Return a run row plus its ordered ``operation_ids``, or ``None``."""
    row = db.conn.execute(
        "SELECT * FROM mfdb_pipeline_run "
        "WHERE pipeline_run_id = ? AND deleted_at IS NULL",
        (pipeline_run_id,),
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["operation_ids"] = [
        r[0]
        for r in db.conn.execute(
            "SELECT operation_id FROM mfdb_pipeline_run_operation "
            "WHERE pipeline_run_id = ? AND deleted_at IS NULL ORDER BY ordinal",
            (pipeline_run_id,),
        ).fetchall()
    ]
    return result


def list_pipeline_runs(
    db: Any, pipeline_id: str | None = None
) -> list[dict[str, Any]]:
    """List pipeline runs (newest first), each with its operation count.

    Filtered to one ``pipeline_id`` when given, else all runs.
    """
    where = ["r.deleted_at IS NULL"]
    params: list[Any] = []
    if pipeline_id is not None:
        where.append("r.pipeline_id = ?")
        params.append(pipeline_id)
    rows = db.conn.execute(
        "SELECT r.pipeline_run_id, r.pipeline_id, r.name, r.status, r.created_at, "
        "(SELECT COUNT(*) FROM mfdb_pipeline_run_operation o "
        " WHERE o.pipeline_run_id = r.pipeline_run_id AND o.deleted_at IS NULL) "
        "AS operation_count "
        f"FROM mfdb_pipeline_run r WHERE {' AND '.join(where)} "
        "ORDER BY r.created_at DESC, r.rowid DESC",
        params,
    ).fetchall()
    return [dict(r) for r in rows]
