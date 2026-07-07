"""Lightweight reagent / consumable inventory (PRD-15, LIMS P4).

Track consumable lots — fluorophore/buffer/optical-filter/kit — with lot number and
expiry, and link a lot to whatever *used* it (an operation, a setup, or a sample)
through the orthogonal ``mfdb_reagent_usage`` table, so a result ties to exactly what
was used. Dictionary-declared schema (``mfdb_reagent_lot`` / ``mfdb_reagent_usage``);
these are thin free functions over an MFDB handle.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

#: Extensible reagent classes (mirrors the ``mfdb_reagent_lot.kind`` enum).
REAGENT_KINDS: frozenset[str] = frozenset(
    {"fluorophore", "buffer", "optical_filter", "kit", "other"}
)
#: What a lot can be linked to (mirrors the ``mfdb_reagent_usage.target_type`` enum).
REAGENT_TARGET_TYPES: frozenset[str] = frozenset({"operation", "setup", "sample"})


def _now() -> str:
    from mfdb.repository import _utc_now

    return _utc_now()


def _next_id(db: Any, table: str, column: str) -> int:
    return db.conn.execute(f"SELECT COALESCE(MAX({column}), 0) FROM {table}").fetchone()[0] + 1


def add_reagent_lot(
    db: Any,
    *,
    kind: str,
    name: str,
    lot_number: str = "",
    expiry: str | None = None,
    vendor: str = "",
    catalog_no: str = "",
    concentration: float | None = None,
    concentration_units: str = "",
    received_at: str | None = None,
    opened_at: str | None = None,
    probe_id: str | None = None,
    details: str = "",
    created_by_user_id: str | None = None,
    lot_id: str | None = None,
) -> str:
    """Create a reagent lot; return its ``lot_id``. ``expiry`` is an ISO date string."""
    if kind not in REAGENT_KINDS:
        raise ValueError(f"unknown reagent kind {kind!r}; expected one of {sorted(REAGENT_KINDS)}")
    if not name:
        raise ValueError("reagent lot name is required")
    if created_by_user_id is None:
        from mfdb.security.session import configured_default_user_id

        created_by_user_id = configured_default_user_id()
    lid = lot_id or str(uuid.uuid4())
    now = _now()
    with db.transaction():
        db.conn.execute(
            "INSERT INTO mfdb_reagent_lot (lot_id, kind, name, vendor, catalog_no, "
            "lot_number, concentration, concentration_units, received_at, opened_at, "
            "expiry, probe_id, created_by_user_id, details, created_at, updated_at, "
            "deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (lid, kind, name, vendor or None, catalog_no or None, lot_number or None,
             concentration, concentration_units or None, received_at, opened_at,
             expiry, probe_id, created_by_user_id, details or None, now, now, None),
        )
    return lid


def link_reagent(
    db: Any, lot_id: str, target_type: str, target_id: str, role: str = "used"
) -> None:
    """Link a lot to an operation/setup/sample that used it (idempotent on the triple)."""
    if target_type not in REAGENT_TARGET_TYPES:
        raise ValueError(
            f"unknown target_type {target_type!r}; expected one of {sorted(REAGENT_TARGET_TYPES)}"
        )
    now = _now()
    with db.transaction():
        exists = db.conn.execute(
            "SELECT 1 FROM mfdb_reagent_usage WHERE lot_id = ? AND target_type = ? "
            "AND target_id = ? AND deleted_at IS NULL",
            (lot_id, target_type, target_id),
        ).fetchone()
        if exists:
            return
        db.conn.execute(
            "INSERT INTO mfdb_reagent_usage (usage_row_id, lot_id, target_type, "
            "target_id, role, created_at, updated_at, deleted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_next_id(db, "mfdb_reagent_usage", "usage_row_id"), lot_id, target_type,
             target_id, role or None, now, now, None),
        )


def list_reagents_for(db: Any, target_type: str, target_id: str) -> list[dict[str, Any]]:
    """List the reagent lots used by an operation/setup/sample (the "what was used" query)."""
    rows = db.conn.execute(
        "SELECT l.* FROM mfdb_reagent_lot l "
        "JOIN mfdb_reagent_usage u ON u.lot_id = l.lot_id "
        "WHERE u.target_type = ? AND u.target_id = ? "
        "AND u.deleted_at IS NULL AND l.deleted_at IS NULL "
        "ORDER BY l.kind, l.name",
        (target_type, target_id),
    ).fetchall()
    return [dict(r) for r in rows]


def list_lots(
    db: Any,
    kind: str | None = None,
    *,
    include_expired: bool = False,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """List lots, optionally filtered by ``kind`` and excluding expired ones."""
    where = ["deleted_at IS NULL"]
    params: list[Any] = []
    if kind is not None:
        where.append("kind = ?")
        params.append(kind)
    if not include_expired:
        where.append("(expiry IS NULL OR expiry >= ?)")
        params.append(as_of or date.today().isoformat())
    rows = db.conn.execute(
        f"SELECT * FROM mfdb_reagent_lot WHERE {' AND '.join(where)} ORDER BY kind, name",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def expired_lots(db: Any, as_of: str | None = None) -> list[dict[str, Any]]:
    """Lots whose ``expiry`` is before ``as_of`` (default today) — for QC/warnings."""
    cutoff = as_of or date.today().isoformat()
    rows = db.conn.execute(
        "SELECT * FROM mfdb_reagent_lot "
        "WHERE deleted_at IS NULL AND expiry IS NOT NULL AND expiry < ? "
        "ORDER BY expiry",
        (cutoff,),
    ).fetchall()
    return [dict(r) for r in rows]
