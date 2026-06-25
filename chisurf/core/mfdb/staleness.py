"""Calibration-change impact: link results to calibrations and detect staleness (PRD-05).

A calibration (``register_calibration`` → a ``calibration_data`` artifact produced by a
``calibration`` operation) can be *used by* a downstream result/fit, recorded as a
``calibrated_by`` edge (source = the consumer, target = the calibration artifact — the
standard consumer→resource convention, and a member of the lineage
``USAGE_RELATIONSHIPS`` so ``Lineage.impact_of`` already follows it).

A use is **stale** when a newer calibration of the same ``calibration_type`` exists. This
is the read side of "when a calibration value changes, all downstream fits that used it
can be identified" — complementing the artifact-centric ``Lineage.impact_of``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def record_calibration_use(
    db: Any,
    *,
    used_by_id: str,
    calibration_artifact_id: str,
    used_by_type: str = "operation",
) -> None:
    """Record that ``used_by_id`` used a calibration, via a ``calibrated_by`` edge.

    ``used_by_type`` is ``"operation"`` (a fit operation) or ``"artifact"``. The edge
    runs consumer → calibration artifact, matching the MFDB usage-edge convention.
    """
    db.add_edge(
        source_node_type=used_by_type,
        source_node_id=used_by_id,
        target_node_type="artifact",
        target_node_id=calibration_artifact_id,
        relationship_type="calibrated_by",
    )


@dataclass(frozen=True)
class StaleCalibrationUse:
    """A result that used a calibration superseded by a newer one of the same type."""

    used_by_id: str
    calibration_type: str
    used_artifact_id: str
    latest_artifact_id: str


def _calibration_type(metadata_json: str | None) -> str:
    if not metadata_json:
        return ""
    try:
        meta = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
    except (ValueError, TypeError):
        return ""
    return (meta or {}).get("calibration_type", "") or ""


def _latest_calibration_of_type(db: Any, calibration_type: str) -> str | None:
    """The most recently inserted ``calibration_data`` artifact of a given type.

    Recency is insertion order (``rowid``) — robust to coarse ``created_at`` ties for
    calibrations registered in quick succession; calibration registration is append-only.
    """
    for artifact_id, metadata_json in db.conn.execute(
        "SELECT artifact_id, metadata_json FROM mfdb_artifact "
        "WHERE artifact_kind = 'calibration_data' AND deleted_at IS NULL "
        "ORDER BY rowid DESC"
    ).fetchall():
        if _calibration_type(metadata_json) == calibration_type:
            return artifact_id
    return None


def find_stale_calibration_uses(db: Any) -> list[StaleCalibrationUse]:
    """Find ``calibrated_by`` uses whose calibration is superseded by a newer one.

    For every ``calibrated_by`` edge, read the used calibration's ``calibration_type``
    and compare against the latest calibration of that type; if they differ, the use is
    stale. Returns one :class:`StaleCalibrationUse` per stale edge.
    """
    stale: list[StaleCalibrationUse] = []
    edges = db.conn.execute(
        "SELECT e.source_node_id, e.target_node_id, a.metadata_json "
        "FROM mfdb_edge e JOIN mfdb_artifact a ON a.artifact_id = e.target_node_id "
        "WHERE e.relationship_type = 'calibrated_by' AND e.deleted_at IS NULL "
        "AND a.deleted_at IS NULL"
    ).fetchall()
    for used_by_id, used_artifact_id, metadata_json in edges:
        cal_type = _calibration_type(metadata_json)
        if not cal_type:
            continue
        latest = _latest_calibration_of_type(db, cal_type)
        if latest is not None and latest != used_artifact_id:
            stale.append(
                StaleCalibrationUse(
                    used_by_id=used_by_id,
                    calibration_type=cal_type,
                    used_artifact_id=used_artifact_id,
                    latest_artifact_id=latest,
                )
            )
    return stale
