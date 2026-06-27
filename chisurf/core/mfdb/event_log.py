"""Append-only MFDB persistence for the operation-history event log (PRD-43).

Each recorded interactive action becomes one row in ``mfdb_event_log``; the
in-memory ``OperationHistory`` is a projection of these rows.  Every function is
best-effort and degrades to a no-op / empty result when no MFDB connection is
available, mirroring :func:`chisurf.core.mfdb.result_registry.register_operation`.
The store is append-only: rows are inserted, never updated in place, and re-inserts
of the same ``event_id`` are ignored (idempotent restore/replay).
"""

from __future__ import annotations

import json
import logging
import typing

logger = logging.getLogger(__name__)

# Columns written, in order. Audit columns (created_at/updated_at/deleted_at) are
# defaulted by the schema and not set here.
_COLUMNS: tuple[str, ...] = (
    "event_id",
    "seq",
    "project_id",
    "action_type",
    "operation_type",
    "summary",
    "payload_json",
    "source_uid",
    "target_uid",
    "operation_id",
    "event_timestamp",
    "replayable",
    "side_effect_class",
    "history_version",
)


def _resolve_conn(db: typing.Any) -> typing.Any:
    """Return a live sqlite connection, or ``None`` when MFDB is unavailable."""
    if db is None:
        from chisurf.core.mfdb.result_registry import _get_global_db

        db = _get_global_db()
    if db is None:
        return None
    return getattr(db, "conn", None)


def _ensure_index(conn: typing.Any) -> None:
    """Guarantee a UNIQUE index on event_id so re-inserts are idempotent."""
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_mfdb_event_log_event_id ON mfdb_event_log(event_id)"
    )


def append_event(
    event: dict[str, typing.Any],
    *,
    project_id: str | None = None,
    operation_type: str | None = None,
    operation_id: str | None = None,
    history_version: str | None = None,
    db: typing.Any = None,
) -> bool:
    """Append one history event to ``mfdb_event_log`` (best-effort, append-only).

    Returns ``True`` if a row was written, ``False`` when no MFDB is available or
    the write failed (history stays authoritative in memory either way). Re-inserts
    of an existing ``event_id`` are ignored.
    """
    conn = _resolve_conn(db)
    if conn is None:
        return False
    event_id = event.get("event_id")
    if not event_id:
        return False
    try:
        _ensure_index(conn)
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM mfdb_event_log").fetchone()[0]
        payload = event.get("payload") or {}
        row = {
            "event_id": str(event_id),
            "seq": int(seq),
            "project_id": project_id,
            "action_type": str(event.get("action_type", "")),
            "operation_type": operation_type,
            "summary": event.get("summary"),
            "payload_json": json.dumps(payload, sort_keys=True) if payload else None,
            "source_uid": event.get("source_uid"),
            "target_uid": event.get("target_uid"),
            "operation_id": operation_id,
            "event_timestamp": event.get("timestamp"),
            "replayable": 1 if payload.get("replayable", True) else 0,
            "side_effect_class": payload.get("side_effect_class"),
            "history_version": history_version,
        }
        placeholders = ", ".join("?" * len(_COLUMNS))
        # Append-only on the event payload, but allow a later call to *stamp* an
        # existing row's project_id / operation_id (e.g. at project-archive time,
        # after the event was first written untagged by the live dual-write).
        # COALESCE keeps any value already set — scoping is only ever filled in,
        # never cleared.
        conn.execute(
            f"INSERT INTO mfdb_event_log ({', '.join(_COLUMNS)}) "
            f"VALUES ({placeholders}) "
            "ON CONFLICT(event_id) DO UPDATE SET "
            "  project_id = COALESCE(excluded.project_id, mfdb_event_log.project_id), "
            "  operation_id = COALESCE(excluded.operation_id, mfdb_event_log.operation_id)",
            tuple(row[c] for c in _COLUMNS),
        )
        conn.commit()
        return True
    except Exception:  # never break the live session on a persistence hiccup
        logger.debug("append_event failed", exc_info=True)
        return False


def read_events(
    *,
    project_id: str | None = None,
    limit: int | None = None,
    db: typing.Any = None,
) -> list[dict[str, typing.Any]]:
    """Read events back in log order, shaped like in-memory history events.

    Returns ``[]`` when no MFDB is available. The returned dicts carry the same
    keys ``OperationHistory.load_events`` expects, so the in-memory history is
    rehydrated as a faithful projection of the durable log.
    """
    conn = _resolve_conn(db)
    if conn is None:
        return []
    try:
        sql = (
            "SELECT event_id, action_type, summary, payload_json, "
            "source_uid, target_uid, event_timestamp "
            "FROM mfdb_event_log WHERE deleted_at IS NULL"
        )
        params: list[typing.Any] = []
        if project_id is not None:
            sql += " AND project_id = ?"
            params.append(project_id)
        sql += " ORDER BY seq ASC"
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        logger.debug("read_events failed", exc_info=True)
        return []

    events: list[dict[str, typing.Any]] = []
    for event_id, action_type, summary, payload_json, source_uid, target_uid, ts in rows:
        events.append(
            {
                "event_id": event_id,
                "action_type": action_type,
                "summary": summary or "",
                "payload": json.loads(payload_json) if payload_json else {},
                "source_uid": source_uid,
                "target_uid": target_uid,
                "timestamp": ts,
            }
        )
    return events
