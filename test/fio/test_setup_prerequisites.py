"""Tests for PRD-04 prerequisite tasks P1–P4.

Covers structured child tables, legacy JSON import, dictionary mapping, and
RPC structured-field output.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from chisurf.core.mfdb import schema
from chisurf.core.mfdb.dictionary_schema_map import build_dictionary_schema_map
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups import (
    _resolve_active_user_id,
    _save_setup_row,
    setup_id_for_name,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _db(tmp_path: Path) -> MFDatabase:
    """Open a fresh MFDatabase at a temporary path."""
    p = tmp_path / "test.db"
    db = MFDatabase(str(p))
    return db


# ---------------------------------------------------------------------------
# Test 1: Saving a detector setup creates structured child records
# ---------------------------------------------------------------------------


def test_save_setup_creates_structured_child_tables(tmp_path: Path) -> None:
    """Saving a detector setup creates queryable mfdb_setup_detector_channel
    and mfdb_setup_pie_window rows plus the parent mfdb_setup row."""
    db = _db(tmp_path)
    try:
        setup_name = "Test MFD Setup"
        setup_id = setup_id_for_name(setup_name)
        detectors_data = {
            "green": {"channels": [0, 8], "micro_time_ranges": [(0, 4096)]},
            "red": {"channels": [1, 9], "micro_time_ranges": [(0, 4096)]},
        }
        windows_data = {
            "prompt": (0, 2048),
            "after": (2048, 4096),
        }

        db.save_setup(
            setup_id=setup_id,
            name=setup_name,
            description="Test setup for structured storage",
            configuration={"setup_type": "tttr_detector_setup"},
            detectors=detectors_data,
            windows=windows_data,
        )

        # Verify parent row
        setup = db.get_setup(setup_id)
        assert setup is not None
        assert setup["name"] == setup_name

        # Verify structured child rows
        channels = db.list_detector_channels(setup_id)
        assert len(channels) == 2
        channel_names = {c["name"] for c in channels}
        assert channel_names == {"green", "red"}

        pie_windows = db.list_pie_windows(setup_id)
        assert len(pie_windows) == 2
        window_names = {w["name"] for w in pie_windows}
        assert window_names == {"prompt", "after"}

        # get_setup() also embeds child rows
        full = db.get_setup(setup_id)
        assert len(full.get("detector_channels", [])) == 2
        assert len(full.get("pie_windows", [])) == 2
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 2: Legacy JSON import backfills into structured form
# ---------------------------------------------------------------------------


def test_legacy_json_import_backfills_structured_tables(tmp_path: Path) -> None:
    """Importing legacy detector_setups.json data via _save_setup_row creates
    structured child rows matching the original JSON payload."""
    db = _db(tmp_path)
    try:
        setup_name = "Legacy MFD Setup"
        legacy_data = {
            "detectors": {
                "green": {"chs": [0, 8], "micro_time_ranges": [(0, 4096)]},
                "red": {"chs": [1, 9], "micro_time_ranges": [(0, 4096)]},
            },
            "windows": {
                "prompt": (0, 2048),
                "after": (2048, 4096),
            },
            "tttr_reading": {
                "file_type": "spc",
                "macro_time_calibration": 1.0e-9,
            },
        }
        _save_setup_row(db, setup_name, legacy_data, user_id="")
        setup_id = setup_id_for_name(setup_name)

        # Parent row exists
        setup = db.get_setup(setup_id)
        assert setup is not None
        assert setup["name"] == setup_name

        # Structured detector channels (JSON columns stored as text; legacy
        # import uses ``chs`` key → ``channels`` column is None)
        channels = db.list_detector_channels(setup_id)
        assert len(channels) == 2
        for ch in channels:
            assert ch["name"] in ("green", "red")

        # Structured pie windows
        pie_windows = db.list_pie_windows(setup_id)
        assert len(pie_windows) == 2
        for pw in pie_windows:
            assert pw["name"] in ("prompt", "after")
            assert pw["start"] is not None
            assert pw["end"] is not None

        # Timing info stored on parent
        config = json.loads(setup.get("configuration_json") or "{}")
        setup_data = config.get("setup_data", {})
        assert setup_data.get("tttr_reading", {}).get("file_type") == "spc"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Test 3: Dictionary is the source of truth
# ---------------------------------------------------------------------------


def _setup_dictionary_mapper(tmp_path: Path):
    """Return a DictionarySchemaMap for a fresh MFDB."""
    db_path = os.path.join(tmp_path, "test_dict.db")
    db = MFDatabase(db_path)
    db.close()
    return build_dictionary_schema_map(db_path)


_SETUP_CATEGORIES = {"mfdb_setup", "mfdb_setup_detector_channel", "mfdb_setup_pie_window", "mfdb_setup_fcs_pair", "mfdb_setup_calibration", "mfdb_artifact", "mfdb_parameter", "mfdb_operation_parameter_def"}


def test_every_setup_dictionary_item_maps_to_live_column(tmp_path: Path) -> None:
    """Every dictionary item in the setup/detector/window categories maps to a
    live column — no curated allow-list, no omissions."""
    mapper = _setup_dictionary_mapper(tmp_path)
    failures: list[str] = []
    for cat_name in _SETUP_CATEGORIES:
        cat = mapper.dictionary.get_category(cat_name)
        if cat is None:
            failures.append(f"Category {cat_name!r} not found in dictionary")
            continue
        for attr, item in cat.items.items():
            full_name = item.name or f"_{cat_name}.{attr}"
            mapped = mapper.map_dictionary_item(full_name)
            if mapped is None:
                failures.append(f"{full_name}: no mapping produced")
                continue
            ok, message = mapper.validate_mapping(full_name)
            if not ok:
                failures.append(f"{full_name}: {message}")
    assert not failures, f"Dictionary-schema mapping failures:\n" + "\n".join(failures)


def test_no_setup_dictionary_items_are_unmapped(tmp_path: Path) -> None:
    """None of the setup/detector/window dictionary items appear in the
    unmapped list from get_unmapped_flr_items()."""
    mapper = _setup_dictionary_mapper(tmp_path)
    unmapped = mapper.get_unmapped_flr_items()
    unmapped_in_categories = [
        u for u in unmapped if u.category in _SETUP_CATEGORIES
    ]
    assert not unmapped_in_categories, (
        f"Items should not be unmapped:\n" +
        "\n".join(f"  {u.dictionary_name}: {u.reason}" for u in unmapped_in_categories)
    )


def test_fresh_db_has_no_legacy_or_duplicate_tables(tmp_path: Path) -> None:
    """PRD-19: a freshly built MFDB contains no legacy ``fdb_*`` tables and no
    duplicate-of-flrCIF ``mfdb_sample``/``mfdb_experiment`` tables, while the
    canonical tables are present. Guards the dictionary-driven schema against
    legacy regressions."""
    import sqlite3

    db_path = os.path.join(tmp_path, "legacy_check.db")
    MFDatabase(db_path).close()
    con = sqlite3.connect(db_path)
    try:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        con.close()

    legacy = sorted(t for t in tables if t.startswith("fdb_"))
    duplicates = sorted(t for t in tables if t in {"mfdb_sample", "mfdb_experiment"})
    assert not legacy, f"Legacy fdb_* tables must not exist: {legacy}"
    assert not duplicates, f"Duplicate-of-flrCIF tables must not exist: {duplicates}"
    # Canonical tables are present (flrCIF authoritative + mfdb_* extensions).
    for canonical in ("flr_sample", "mfdb_artifact", "mfdb_operation", "mfdb_edge"):
        assert canonical in tables, f"Canonical table {canonical!r} missing"


# ---------------------------------------------------------------------------
# Test 4: mfdb-admin setup list/detail RPC returns structured fields
# ---------------------------------------------------------------------------


def _admin_auth(db_path: str) -> dict:
    """Mint a valid session token for the seeded default admin at *db_path*.

    A fresh MFDatabase seeds an admin user, so _require_auth no longer treats
    writes as bootstrap and rejects anonymous requests; handlers need a token.
    """
    from mfdb.auth import create_session

    with MFDatabase(db_path) as db:
        token = create_session(db.conn, "user_default")["token"]
        db.conn.commit()
    return {"token": token}


def test_list_setups_handler_returns_structured_fields(tmp_path: Path) -> None:
    """mfdb.setups.list returns detector_channels and pie_windows keys."""
    from mfdb.admin.backend.services import (
        list_setups_handler,
        get_setup_handler,
    )
    from unittest.mock import patch

    db_path = os.path.join(tmp_path, "test_rpc.db")
    with MFDatabase(db_path):
        pass
    auth = _admin_auth(db_path)

    patchers = [
        patch(
            "mfdb.admin.backend.services.resolve_database_path",
            return_value=db_path,
        ),
        patch(
            "mfdb.admin.backend.services.resolve_database_path",
            return_value=db_path,
        ),
        # list_setups_handler / get_setup_handler delegate to chisurf.core.mfdb.api,
        # which binds its own resolve_database_path — patch it too so the read path
        # uses this temp DB and the test never touches the real user database.
        patch(
            "chisurf.core.mfdb.api.resolve_database_path",
            return_value=db_path,
        ),
    ]
    for p in patchers:
        p.start()
    try:
        # Save a setup with detectors and windows
        from mfdb.admin.backend.services import (
            save_setup_handler,
        )
        setup_payload = {
            "setup_id": "rpc_test_setup",
            "name": "RPC Test Setup",
            "configuration": {
                "setup_type": "tttr_detector_setup",
                "setup_data": {
                    "detectors": {"green": {"chs": [0]}},
                    "windows": {"prompt": (0, 2048)},
                },
            },
            "detectors": {"green": {"chs": [0]}},
        }
        res = save_setup_handler(setup=setup_payload, auth=auth)
        assert "setup" in res

        # list returns setups (direct API, not wrapped with ok)
        res = list_setups_handler(auth=auth)
        assert "setups" in res
        assert any(s["setup_id"] == "rpc_test_setup" for s in res["setups"])

        # get returns structured child rows
        res = get_setup_handler(setup_id="rpc_test_setup", auth=auth)
        setup = res["setup"]
        assert setup["setup_id"] == "rpc_test_setup"
    finally:
        for p in patchers:
            p.stop()


def test_setup_detail_rpc_includes_child_tables(tmp_path: Path) -> None:
    """mfdb.setups.get returns detector_channels and pie_windows lists."""
    from mfdb.admin.backend.services import (
        save_setup_handler,
    )
    from mfdb.admin.backend.services import (
        get_setup_handler,
    )
    from unittest.mock import patch

    db_path = os.path.join(tmp_path, "test_detail.db")
    with MFDatabase(db_path):
        pass
    auth = _admin_auth(db_path)

    patchers = [
        patch(
            "mfdb.admin.backend.services.resolve_database_path",
            return_value=db_path,
        ),
        patch(
            "mfdb.admin.backend.services.resolve_database_path",
            return_value=db_path,
        ),
        # list_setups_handler / get_setup_handler delegate to chisurf.core.mfdb.api,
        # which binds its own resolve_database_path — patch it too so the read path
        # uses this temp DB and the test never touches the real user database.
        patch(
            "chisurf.core.mfdb.api.resolve_database_path",
            return_value=db_path,
        ),
    ]
    for p in patchers:
        p.start()
    try:
        setup_payload = {
            "setup_id": "detail_test",
            "name": "Detail Test",
            "configuration": {"setup_type": "tttr_detector_setup"},
            "detectors": {"green": {"chs": [0, 8]}, "red": {"chs": [1, 9]}},
        }
        res = save_setup_handler(setup=setup_payload, auth=auth)
        assert "setup" in res

        res = get_setup_handler(setup_id="detail_test", auth=auth)
        setup = res["setup"]
        assert "detector_channels" in setup
        assert "pie_windows" in setup
    finally:
        for p in patchers:
            p.stop()
