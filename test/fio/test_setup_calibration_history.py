"""Tests for append-only calibration snapshot history (PRD-04 Sidequest D).

Covers:
- D1: ``mfdb_setup_calibration`` table exists and maps to live columns.
- D2: ``save_setup`` with calibration data creates append-only snapshots.
- D4: ``add_setup_calibration``, ``list_setup_calibration_dates``,
      ``get_setup_calibration`` work correctly.
- D5: Migration backfill creates one snapshot per existing channel.
- D6: Calibrated-at metadata is recorded in operations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chisurf.core.mfdb.store.database_resolver import resolve_database_path
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.schema.schema import migrate_schema, get_schema_version, SCHEMA_VERSION
from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups import (
    setup_id_for_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_db(tmp_path: Path, name: str = "test.db") -> MFDatabase:
    """Open a fresh MFDatabase at a temporary path."""
    db = MFDatabase(str(tmp_path / name))
    return db


# ---------------------------------------------------------------------------
# D1: Table existence and dictionary mapping
# ---------------------------------------------------------------------------


def test_calibration_table_exists(tmp_path: Path) -> None:
    """After fresh schema setup, mfdb_setup_calibration exists and
    has the expected columns."""
    db = _fresh_db(tmp_path)
    try:
        cols = {
            r[1] for r in db.conn.execute(
                "PRAGMA table_info(mfdb_setup_calibration)"
            ).fetchall()
        }
        required = {
            "id", "setup_id", "channel_name",
            "g_factor", "l1", "l2", "g_factor_channels",
            "g_factor_calibration_id", "calibrated_at", "method",
            "created_by_user_id", "created_at", "updated_at", "deleted_at",
        }
        missing = required - cols
        assert not missing, f"Missing columns: {missing}"
    finally:
        db.close()


def test_calibration_dictionary_maps(tmp_path: Path) -> None:
    """All mfdb_setup_calibration dictionary items map to live columns."""
    from chisurf.core.mfdb.schema.dictionary_schema_map import build_dictionary_schema_map

    db_path = os.path.join(tmp_path, "test_dict_cal.db")
    db = MFDatabase(db_path)
    db.close()

    mapper = build_dictionary_schema_map(db_path)
    cat = mapper.dictionary.get_category("mfdb_setup_calibration")
    assert cat is not None, "mfdb_setup_calibration category not found in dictionary"
    failures = []
    for attr, item in cat.items.items():
        full_name = item.name or f"_mfdb_setup_calibration.{attr}"
        mapped = mapper.map_dictionary_item(full_name)
        if mapped is None:
            failures.append(f"{full_name}: no mapping produced")
            continue
        ok, message = mapper.validate_mapping(full_name)
        if not ok:
            failures.append(f"{full_name}: {message}")
    assert not failures, f"Dictionary-schema mapping failures:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# D2/D4: Append-only calibration snapshots
# ---------------------------------------------------------------------------


def test_save_setup_creates_calibration_snapshots(tmp_path: Path) -> None:
    """Saving a setup with g_factor/l1/l2 creates calibration snapshots."""
    db = _fresh_db(tmp_path)
    try:
        setup_name = "Cal Test Setup"
        setup_id = setup_id_for_name(setup_name)
        db.save_setup(
            setup_id=setup_id,
            name=setup_name,
            detectors={
                "green": {
                    "channels": [0, 8],
                    "g_factor": 1.05,
                    "l1": 0.02,
                    "l2": 0.03,
                },
                "red": {
                    "channels": [1, 9],
                    "g_factor": 1.10,
                    "l1": 0.01,
                    "l2": 0.04,
                },
            },
        )

        # Check calibration snapshots exist
        snapshots = db.conn.execute(
            "SELECT * FROM mfdb_setup_calibration WHERE setup_id = ? ORDER BY channel_name",
            (setup_id,),
        ).fetchall()
        assert len(snapshots) == 2
        snap_by_ch = {r["channel_name"]: dict(r) for r in snapshots}
        assert snap_by_ch["green"]["g_factor"] == 1.05
        assert snap_by_ch["green"]["l1"] == 0.02
        assert snap_by_ch["green"]["l2"] == 0.03
        assert snap_by_ch["red"]["g_factor"] == 1.10
        assert snap_by_ch["red"]["l1"] == 0.01
        assert snap_by_ch["red"]["l2"] == 0.04
        assert snap_by_ch["green"]["method"] == "manual"
        assert snap_by_ch["red"]["method"] == "manual"

        # Re-saving the same setup with unchanged factors (e.g. a structural
        # edit) must NOT create duplicate snapshots — only real calibration
        # changes are recorded.
        db.save_setup(
            setup_id=setup_id,
            name=setup_name,
            detectors={
                "green": {"channels": [0, 8], "g_factor": 1.05, "l1": 0.02, "l2": 0.03},
                "red": {"channels": [1, 9], "g_factor": 1.10, "l1": 0.01, "l2": 0.04},
            },
        )
        assert len(db.conn.execute(
            "SELECT 1 FROM mfdb_setup_calibration WHERE setup_id = ?", (setup_id,)
        ).fetchall()) == 2

        # Changing a factor DOES append a new snapshot for that channel.
        db.save_setup(
            setup_id=setup_id,
            name=setup_name,
            detectors={
                "green": {"channels": [0, 8], "g_factor": 1.07, "l1": 0.02, "l2": 0.03},
                "red": {"channels": [1, 9], "g_factor": 1.10, "l1": 0.01, "l2": 0.04},
            },
        )
        green_snaps = db.conn.execute(
            "SELECT g_factor FROM mfdb_setup_calibration "
            "WHERE setup_id = ? AND channel_name = 'green' ORDER BY calibrated_at",
            (setup_id,),
        ).fetchall()
        assert len(green_snaps) == 2
        assert {round(r[0], 2) for r in green_snaps} == {1.05, 1.07}
    finally:
        db.close()


def test_add_setup_calibration_appends_snapshots(tmp_path: Path) -> None:
    """add_setup_calibration creates a new row and does not overwrite."""
    db = _fresh_db(tmp_path)
    try:
        setup_name = "Append Test"
        setup_id = setup_id_for_name(setup_name)
        db.save_setup(setup_id=setup_id, name=setup_name, detectors={"green": {"channels": [0]}})

        # Add first calibration
        snap1 = db.add_setup_calibration(
            setup_id=setup_id,
            channel_name="green",
            g_factor=1.0,
            l1=0.01,
            l2=0.02,
            method="manual",
        )
        assert snap1["id"] is not None

        # Add second calibration (same channel, different values)
        snap2 = db.add_setup_calibration(
            setup_id=setup_id,
            channel_name="green",
            g_factor=2.0,
            l1=0.03,
            l2=0.04,
            method="jordi_g_factor",
        )
        assert snap2["id"] != snap1["id"]

        # Both rows exist
        rows = db.conn.execute(
            "SELECT * FROM mfdb_setup_calibration WHERE setup_id = ? AND channel_name = ? ORDER BY id",
            (setup_id, "green"),
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["g_factor"] == 1.0
        assert rows[1]["g_factor"] == 2.0
    finally:
        db.close()


def test_list_setup_calibration_dates(tmp_path: Path) -> None:
    """list_setup_calibration_dates returns distinct dates newest first."""
    db = _fresh_db(tmp_path)
    try:
        setup_name = "Date Test"
        setup_id = setup_id_for_name(setup_name)
        db.save_setup(setup_id=setup_id, name=setup_name, detectors={"green": {"channels": [0]}})

        db.add_setup_calibration(
            setup_id=setup_id, channel_name="green",
            g_factor=1.0, calibrated_at="2024-01-01T00:00:00",
        )
        db.add_setup_calibration(
            setup_id=setup_id, channel_name="green",
            g_factor=1.1, calibrated_at="2024-06-01T00:00:00",
        )

        dates = db.list_setup_calibration_dates(setup_id)
        assert len(dates) == 2
        assert dates[0] == "2024-06-01T00:00:00"  # newest first
        assert dates[1] == "2024-01-01T00:00:00"
    finally:
        db.close()


def test_get_setup_calibration_returns_latest(tmp_path: Path) -> None:
    """get_setup_calibration with no date returns latest per channel."""
    db = _fresh_db(tmp_path)
    try:
        setup_name = "Latest Test"
        setup_id = setup_id_for_name(setup_name)
        db.save_setup(
            setup_id=setup_id, name=setup_name,
            detectors={"green": {"channels": [0]}, "red": {"channels": [1]}},
        )

        # Green: two snapshots
        db.add_setup_calibration(
            setup_id=setup_id, channel_name="green",
            g_factor=1.0, calibrated_at="2024-01-01T00:00:00",
        )
        db.add_setup_calibration(
            setup_id=setup_id, channel_name="green",
            g_factor=2.0, calibrated_at="2024-06-01T00:00:00",
        )
        # Red: one snapshot
        db.add_setup_calibration(
            setup_id=setup_id, channel_name="red",
            g_factor=1.5, calibrated_at="2024-03-01T00:00:00",
        )

        latest = db.get_setup_calibration(setup_id)
        assert len(latest) == 2
        by_ch = {r["channel_name"]: r for r in latest}
        assert by_ch["green"]["g_factor"] == 2.0  # latest
        assert by_ch["red"]["g_factor"] == 1.5
    finally:
        db.close()


def test_get_setup_calibration_at_date(tmp_path: Path) -> None:
    """get_setup_calibration with calibrated_at returns that snapshot."""
    db = _fresh_db(tmp_path)
    try:
        setup_name = "Snapshot Test"
        setup_id = setup_id_for_name(setup_name)
        db.save_setup(setup_id=setup_id, name=setup_name, detectors={"green": {"channels": [0]}})

        db.add_setup_calibration(
            setup_id=setup_id, channel_name="green",
            g_factor=1.0, calibrated_at="2024-01-01T00:00:00",
        )
        db.add_setup_calibration(
            setup_id=setup_id, channel_name="green",
            g_factor=2.0, calibrated_at="2024-06-01T00:00:00",
        )

        snap = db.get_setup_calibration(setup_id, calibrated_at="2024-01-01T00:00:00")
        assert len(snap) == 1
        assert snap[0]["g_factor"] == 1.0
    finally:
        db.close()


def test_get_setup_includes_calibration(tmp_path: Path) -> None:
    """get_setup returns calibration and calibration_dates keys."""
    db = _fresh_db(tmp_path)
    try:
        setup_name = "Full Setup Test"
        setup_id = setup_id_for_name(setup_name)
        db.save_setup(
            setup_id=setup_id, name=setup_name,
            detectors={"green": {"channels": [0], "g_factor": 1.5}},
        )

        full = db.get_setup(setup_id)
        assert full is not None
        assert "calibration_dates" in full
        assert "calibration" in full
        assert len(full["calibration"]) >= 1
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Note: the former test_v35_migration_backfills_existing_channels test exercised the
# removed version-chain migration (mfdb_schema_version stepping v34→v35 with a calibration
# backfill, method="migrated"). PRD-19 deleted the version chain (pre-PRD-19 DBs are
# disposable). Calibration-snapshot creation on the current path is covered by
# test_save_setup_creates_calibration_snapshots / _appends_snapshots above.
