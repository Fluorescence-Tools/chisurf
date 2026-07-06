"""Tests for FCS channel setup MFDB integration (mirrors detector setup tests)."""

import os
import json
import pathlib
from pathlib import Path
from unittest.mock import patch

import pytest

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.fluorescence.fcs.channel_setups import (
    FCS_CHANNEL_SETUPS_FILE,
    FCS_SETUP_TYPE,
    _fcs_config,
    load_fcs_channel_setups,
    save_fcs_channel_setups,
    _save_setup_row,
    _fcs_row_to_data,
    _use_mfdb,
    build_channels_from_setup,
)
FCS_CONFIG = _fcs_config()
from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils import (
    SetupTypeConfig,
    get_db,
    setup_id_for_name as _sifn_shared,
    resolve_active_user_id,
    load_mfdb_setups,
    set_last_used,
)


def _fresh_db(tmp_path: Path) -> MFDatabase:
    db_path = os.path.join(str(tmp_path), "test_fcs.db")
    with MFDatabase(db_path) as db:
        pass
    return MFDatabase(db_path)


def _write_legacy_json(path: Path, setups: dict) -> None:
    data = {"version": 1, "setups": setups, "last_used_setup": None}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def setup_id_for_name(name: str, user_id: str = "") -> str:
    return _sifn_shared(name, user_id, "fcs_channel_setup")


# ---------------------------------------------------------------------------
# S-B1: Save and load FCS channel setups in MFDB
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Saving an FCS channel setup then loading returns the same data."""
    db = _fresh_db(tmp_path)
    try:
        user_id = "user_alice"
        db.conn.execute(
            "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
            "VALUES (?, ?, ?)",
            (user_id, "00000000-0000-0000-0000-000000000001", user_id),
        )

        corr = {"n_bins": 4, "n_casc": 32, "make_fine": True}
        pairs = [
            {"name": "GG", "channel_a": "GG", "channel_b": "GG", "kind": "ACF"},
            {"name": "RR", "channel_a": "RR", "channel_b": "RR", "kind": "ACF"},
        ]
        data = {"correlator": corr, "pairs": pairs}

        _save_setup_row(db, "TestSetup", data, user_id=user_id)

        loaded = load_mfdb_setups(db, FCS_CONFIG, user_id, row_to_data=_fcs_row_to_data)
        assert "TestSetup" in loaded.get("setups", {})
        sd = loaded["setups"]["TestSetup"]
        assert sd.get("correlator", {}).get("n_bins") == 4
        assert sd.get("correlator", {}).get("n_casc") == 32
        assert sd.get("correlator", {}).get("make_fine") is True
        assert len(sd.get("pairs", [])) == 2
    finally:
        db.close()


def test_fcs_pairs_child_table_written(tmp_path: Path) -> None:
    """FCS pairs are stored in the mfdb_setup_fcs_pair child table."""
    db = _fresh_db(tmp_path)
    try:
        user_id = "user_alice"
        db.conn.execute(
            "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
            "VALUES (?, ?, ?)",
            (user_id, "00000000-0000-0000-0000-000000000001", user_id),
        )

        pairs = [
            {"name": "GG", "channel_a": "GG", "channel_b": "GG", "kind": "ACF"},
            {"name": "GR", "channel_a": "GG", "channel_b": "RR", "kind": "CCF"},
        ]
        data = {"correlator": {"n_bins": 2, "n_casc": 25, "make_fine": False}, "pairs": pairs}

        _save_setup_row(db, "PairTest", data, user_id=user_id)

        sid = setup_id_for_name("PairTest", user_id)
        rows = db.conn.execute(
            "SELECT * FROM mfdb_setup_fcs_pair WHERE setup_id = ? AND deleted_at IS NULL ORDER BY id",
            (sid,)
        ).fetchall()
        assert len(rows) == 2
        assert rows[0]["name"] == "GG"
        assert rows[0]["channel_a"] == "GG"
        assert rows[0]["channel_b"] == "GG"
        assert rows[1]["name"] == "GR"
        assert rows[1]["channel_a"] == "GG"
        assert rows[1]["channel_b"] == "RR"
        assert rows[1]["kind"] == "CCF"
    finally:
        db.close()


def test_correlator_typed_columns(tmp_path: Path) -> None:
    """Correlator scalars n_bins, n_casc, make_fine are stored as typed columns."""
    db = _fresh_db(tmp_path)
    try:
        user_id = "user_alice"
        db.conn.execute(
            "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
            "VALUES (?, ?, ?)",
            (user_id, "00000000-0000-0000-0000-000000000001", user_id),
        )

        data = {"correlator": {"n_bins": 8, "n_casc": 50, "make_fine": False}, "pairs": []}
        _save_setup_row(db, "CorrTest", data, user_id=user_id)

        sid = setup_id_for_name("CorrTest", user_id)
        row = db.conn.execute(
            "SELECT n_bins, n_casc, make_fine FROM mfdb_setup WHERE setup_id = ?",
            (sid,)
        ).fetchone()
        assert row is not None
        assert row["n_bins"] == 8
        assert row["n_casc"] == 50
        assert row["make_fine"] == 0  # stored as 0/1 int
    finally:
        db.close()


# ---------------------------------------------------------------------------
# S-B2: User-scoped IDs and ownership
# ---------------------------------------------------------------------------


def test_fcs_id_is_namespaced(tmp_path: Path) -> None:
    """FCS setup IDs include the user slug."""
    sid = setup_id_for_name("My FCS Setup", user_id="user_alice")
    assert sid.startswith("fcs_channel_setup:")
    assert "user_alice" in sid
    assert "my_fcs_setup" in sid


def test_fcs_same_name_different_users_no_collision(tmp_path: Path) -> None:
    """Two users can each have an FCS setup with the same name."""
    db = _fresh_db(tmp_path)
    try:
        alice = "user_alice"
        bob = "user_bob"
        for uid in (alice, bob):
            db.conn.execute(
                "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
                "VALUES (?, ?, ?)",
                (uid, f"00000000-0000-0000-0000-{uid.replace('user_', '')}", uid),
            )

        data = {"correlator": {"n_bins": 2, "n_casc": 25, "make_fine": False}, "pairs": []}
        _save_setup_row(db, "MySetup", data, user_id=alice)
        _save_setup_row(db, "MySetup", data, user_id=bob)

        alice_loaded = load_mfdb_setups(db, FCS_CONFIG, alice, row_to_data=_fcs_row_to_data)
        bob_loaded = load_mfdb_setups(db, FCS_CONFIG, bob, row_to_data=_fcs_row_to_data)

        assert "MySetup" in alice_loaded["setups"]
        assert "MySetup" in bob_loaded["setups"]
    finally:
        db.close()


def test_fcs_save_records_owner(tmp_path: Path) -> None:
    """FCS setup saved for a user stores created_by_user_id."""
    db = _fresh_db(tmp_path)
    try:
        user_id = "user_alice"
        db.conn.execute(
            "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
            "VALUES (?, ?, ?)",
            (user_id, "00000000-0000-0000-0000-000000000001", user_id),
        )

        import chisurf.core.fluorescence.fcs.channel_setups as fcs_mod
        import chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils as utils

        orig_active = utils.resolve_active_user_id
        utils.resolve_active_user_id = lambda: user_id
        orig_get_db = utils.get_db
        utils.get_db = lambda: db
        try:
            data = {
                "setups": {
                    "FcsTest": {
                        "correlator": {"n_bins": 2, "n_casc": 25, "make_fine": False},
                        "pairs": [],
                        "_is_public": False,
                    },
                },
                "last_used": "FcsTest",
            }
            ok = save_fcs_channel_setups(data)
            assert ok

            sid = setup_id_for_name("FcsTest", user_id)
            row = db.get_setup(sid)
            assert row is not None
            assert row["created_by_user_id"] == user_id
        finally:
            utils.resolve_active_user_id = orig_active
            utils.get_db = orig_get_db
    finally:
        db.close()


def test_fcs_save_no_login_still_works(tmp_path: Path) -> None:
    """Shared (no-login) FCS setups are saved without an owner."""
    db = _fresh_db(tmp_path)
    try:
        data = {
            "setups": {
                "NoLoginSetup": {
                    "correlator": {"n_bins": 2, "n_casc": 25, "make_fine": False},
                    "pairs": [],
                },
            },
        }
        _save_setup_row(db, "NoLoginSetup", data, user_id="")
        sid = setup_id_for_name("NoLoginSetup", user_id="")
        row = db.get_setup(sid)
        assert row is not None
        assert row["created_by_user_id"] is None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# S-B3: Visibility — public vs private, ownerless = shared
# ---------------------------------------------------------------------------


def test_fcs_load_excludes_other_users_private_setups(tmp_path: Path) -> None:
    """User A does not see user B's private FCS setups."""
    db = _fresh_db(tmp_path)
    try:
        alice = "user_alice"
        bob = "user_bob"
        for uid in (alice, bob):
            db.conn.execute(
                "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
                "VALUES (?, ?, ?)",
                (uid, f"00000000-0000-0000-0000-{uid.replace('user_', '')}", uid),
            )

        data = {"correlator": {"n_bins": 2, "n_casc": 25, "make_fine": False}, "pairs": []}
        _save_setup_row(db, "AliceOnly", data, user_id=alice, is_public=False)
        _save_setup_row(db, "BobOnly", data, user_id=bob, is_public=False)

        alice_s = load_mfdb_setups(db, FCS_CONFIG, alice, row_to_data=_fcs_row_to_data)
        bob_s = load_mfdb_setups(db, FCS_CONFIG, bob, row_to_data=_fcs_row_to_data)

        assert "AliceOnly" in alice_s["setups"]
        assert "BobOnly" not in alice_s["setups"]
        assert "BobOnly" in bob_s["setups"]
        assert "AliceOnly" not in bob_s["setups"]
    finally:
        db.close()


def test_fcs_load_includes_shared_and_public_setups(tmp_path: Path) -> None:
    """Both users see shared and public FCS setups."""
    db = _fresh_db(tmp_path)
    try:
        alice = "user_alice"
        bob = "user_bob"
        for uid in (alice, bob):
            db.conn.execute(
                "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
                "VALUES (?, ?, ?)",
                (uid, f"00000000-0000-0000-0000-{uid.replace('user_', '')}", uid),
            )

        data = {"correlator": {"n_bins": 2, "n_casc": 25, "make_fine": False}, "pairs": []}
        _save_setup_row(db, "Shared", data, user_id="")  # ownerless = shared
        _save_setup_row(db, "AlicePub", data, user_id=alice, is_public=True)
        _save_setup_row(db, "AlicePriv", data, user_id=alice, is_public=False)

        alice_s = load_mfdb_setups(db, FCS_CONFIG, alice, row_to_data=_fcs_row_to_data)
        bob_s = load_mfdb_setups(db, FCS_CONFIG, bob, row_to_data=_fcs_row_to_data)

        assert "Shared" in alice_s["setups"]
        assert "Shared" in bob_s["setups"]
        assert "AlicePub" in alice_s["setups"]
        assert "AlicePub" in bob_s["setups"]
        assert "AlicePriv" in alice_s["setups"]
        assert "AlicePriv" not in bob_s["setups"]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# S-B4: JSON migration
# ---------------------------------------------------------------------------


def test_fcs_migration_idempotent(tmp_path: Path) -> None:
    """Migrating twice adds no extra rows."""
    db = _fresh_db(tmp_path)
    try:
        user_id = "user_default"
        db.conn.execute(
            "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
            "VALUES (?, ?, ?)",
            (user_id, "00000000-0000-0000-0000-000000000000", user_id),
        )

        legacy = tmp_path / "fcs_channel_setups.json"
        _write_legacy_json(legacy, {"FCS1": {
            "correlator": {"n_bins": 2, "n_casc": 25, "make_fine": False},
            "pairs": [{"name": "GG", "channel_a": "GG", "channel_b": "GG"}],
        }})

        from chisurf.core.fluorescence.fcs.channel_setups import _migrate_json_to_mfdb
        _migrate_json_to_mfdb(db, legacy, user_id=user_id)
        count_after_first = len(db.list_setups())

        _migrate_json_to_mfdb(db, legacy, user_id=user_id)
        count_after_second = len(db.list_setups())

        assert count_after_second == count_after_first
        assert count_after_first >= 1
    finally:
        db.close()


# ---------------------------------------------------------------------------
# S-B5: JSON export preserved (custom file path → JSON)
# ---------------------------------------------------------------------------


def test_fcs_custom_file_uses_json(tmp_path: Path) -> None:
    """A custom file path falls back to JSON, not MFDB."""
    custom = tmp_path / "custom_fcs.json"
    data = {"version": 1, "setups": {}, "last_used_setup": None}
    ok = save_fcs_channel_setups(data, file_path=str(custom))
    assert ok
    assert custom.exists()
    loaded = json.loads(custom.read_text())
    assert loaded.get("version") == 1


# ---------------------------------------------------------------------------
# S-B6: Build channels from setup (legacy helper still works)
# ---------------------------------------------------------------------------


def test_build_channels_basic() -> None:
    """build_channels_from_setup returns the expected channel structure."""
    windows = {"prompt": (0, 1024)}
    detectors = {"SPAD1": {"chs": [0], "micro_time_ranges": [(0, 100)]}}
    result = build_channels_from_setup(windows, detectors)
    assert "SPAD1" in result
    assert len(result["SPAD1"]) == 1
    seg = result["SPAD1"][0]
    assert seg["window_range"] == (0, 1024)
    assert seg["detector_chs"] == [0]
    assert seg["micro_time_range"] == (0, 100)


def test_build_channels_no_windows() -> None:
    """Without windows, segments have window_range=None."""
    detectors = {"SPAD1": {"chs": [0], "micro_time_ranges": [(0, 100)]}}
    result = build_channels_from_setup({}, detectors)
    assert "SPAD1" in result
    assert result["SPAD1"][0]["window_range"] is None


# ---------------------------------------------------------------------------
# Sidequest B addendum: per-pair correlator columns
# ---------------------------------------------------------------------------


def test_per_pair_correlator_roundtrip(tmp_path: Path) -> None:
    """Per-pair n_bins/n_casc/make_fine survive save/load through the
    structured child-table columns."""
    db = _fresh_db(tmp_path)
    try:
        user_id = "user_alice"
        db.conn.execute(
            "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
            "VALUES (?, ?, ?)",
            (user_id, "00000000-0000-0000-0000-000000000001", user_id),
        )

        pairs = [
            {
                "name": "GG", "channel_a": "GG", "channel_b": "GG", "kind": "ACF",
                "n_bins": 4, "n_casc": 32, "make_fine": True,
            },
            {
                "name": "RR", "channel_a": "RR", "channel_b": "RR", "kind": "ACF",
                "n_bins": 2, "n_casc": 16, "make_fine": False,
            },
        ]
        # Default setup-level correlator values
        corr = {"n_bins": 8, "n_casc": 64, "make_fine": True}
        data = {"correlator": corr, "pairs": pairs}
        _save_setup_row(db, "PerPairTest", data, user_id=user_id)

        # Load back and verify per-pair values
        loaded = load_mfdb_setups(db, FCS_CONFIG, user_id, row_to_data=_fcs_row_to_data)
        sd = loaded["setups"]["PerPairTest"]
        loaded_pairs = sd.get("pairs", [])
        assert len(loaded_pairs) == 2

        gg = next(p for p in loaded_pairs if p["name"] == "GG")
        assert gg["n_bins"] == 4
        assert gg["n_casc"] == 32
        assert gg["make_fine"] is True

        rr = next(p for p in loaded_pairs if p["name"] == "RR")
        assert rr["n_bins"] == 2
        assert rr["n_casc"] == 16
        assert rr["make_fine"] is False
    finally:
        db.close()


def test_per_pair_correlator_stored_in_child_table(tmp_path: Path) -> None:
    """Per-pair correlator values are stored in the mfdb_setup_fcs_pair
    child-table columns, not only on the parent row."""
    db = _fresh_db(tmp_path)
    try:
        user_id = "user_alice"
        db.conn.execute(
            "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
            "VALUES (?, ?, ?)",
            (user_id, "00000000-0000-0000-0000-000000000001", user_id),
        )

        pairs = [
            {
                "name": "GR", "channel_a": "GG", "channel_b": "RR", "kind": "CCF",
                "n_bins": 6, "n_casc": 48, "make_fine": True,
            },
        ]
        data = {"correlator": {"n_bins": 2, "n_casc": 25, "make_fine": False}, "pairs": pairs}
        _save_setup_row(db, "ChildColTest", data, user_id=user_id)

        sid = setup_id_for_name("ChildColTest", user_id)
        row = db.conn.execute(
            "SELECT n_bins, n_casc, make_fine FROM mfdb_setup_fcs_pair "
            "WHERE setup_id = ? AND deleted_at IS NULL",
            (sid,),
        ).fetchone()
        assert row is not None
        assert row["n_bins"] == 6
        assert row["n_casc"] == 48
        assert row["make_fine"] == 1
    finally:
        db.close()


# NOTE: the former ``test_v36_migration_backfills_fcs_pair_correlator`` was
# removed. It validated a per-version v35->v36 backfill migration that no longer
# exists after the PRD-19 migration collapse (MIGRATIONS is now {1, 40} with
# reconcile-to-.dic as the additive upgrade path). Worse, it mutated the module
# global ``schema.SCHEMA_VERSION = 35`` without restoring it, which made every
# subsequent test's fresh MFDatabase skip the version-40 reconcile step and get
# a partial (pre-.dic) schema — the root cause of ~150 cascading suite failures.


# ---------------------------------------------------------------------------
# Sidequest B addendum 2: MFDB-only default save; migration removes legacy
# ---------------------------------------------------------------------------


def test_default_fcs_save_uses_mfdb_no_json(tmp_path: Path) -> None:
    """Default save (no file_path) writes to MFDB and does NOT create a JSON
    side-file at the canonical path."""
    import chisurf.core.fluorescence.fcs.channel_setups as fcs_mod
    import chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils as utils

    # Point canonical file to temp location so we don't depend on real config
    orig_file = fcs_mod.FCS_CHANNEL_SETUPS_FILE
    fcs_mod.FCS_CHANNEL_SETUPS_FILE = tmp_path / "fcs_channel_setups.json"
    fcs_mod._FCS_CONFIG_CACHE = None

    db = _fresh_db(tmp_path)
    user_id = "user_alice"
    db.conn.execute(
        "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
        "VALUES (?, ?, ?)",
        (user_id, "00000000-0000-0000-0000-000000000001", user_id),
    )
    db.conn.commit()

    orig_get_db = utils.get_db
    utils.get_db = lambda *a, **kw: db
    orig_active = utils.resolve_active_user_id
    utils.resolve_active_user_id = lambda: user_id

    try:
        data = {
            "setups": {
                "MfdbOnlySetup": {
                    "correlator": {"n_bins": 4, "n_casc": 32, "make_fine": True},
                    "pairs": [],
                },
            },
            "last_used_setup": "MfdbOnlySetup",
        }

        ok = save_fcs_channel_setups(data)
        assert ok, "save_fcs_channel_setups should succeed"

        # No JSON should have been written to the canonical (temp) path
        assert not fcs_mod.FCS_CHANNEL_SETUPS_FILE.exists(), (
            "Default save must not create JSON at the canonical path"
        )

        # Data must be loadable from MFDB
        loaded = load_fcs_channel_setups(
            db_path=db.db_path, user_id=user_id, skip_migration=True,
        )
        assert "MfdbOnlySetup" in loaded["setups"]
        assert loaded["last_used_setup"] == "MfdbOnlySetup"
        sd = loaded["setups"]["MfdbOnlySetup"]
        assert sd["correlator"]["n_bins"] == 4
    finally:
        fcs_mod.FCS_CHANNEL_SETUPS_FILE = orig_file
        fcs_mod._FCS_CONFIG_CACHE = None
        utils.get_db = orig_get_db
        utils.resolve_active_user_id = orig_active
        db.close()


def test_fcs_migration_removes_legacy_file(tmp_path: Path) -> None:
    """After a verified migration, the legacy JSON file is deleted."""
    import chisurf.core.fluorescence.fcs.channel_setups as fcs_mod
    import chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils as utils

    # Point the canonical file to our temp directory
    orig_file = fcs_mod.FCS_CHANNEL_SETUPS_FILE
    fcs_mod.FCS_CHANNEL_SETUPS_FILE = tmp_path / "fcs_channel_setups.json"
    fcs_mod._FCS_CONFIG_CACHE = None

    db_path = str(tmp_path / "mfdb.sqlite")
    with MFDatabase(db_path):
        pass
    db = MFDatabase(db_path)
    user_id = "user_alice"
    db.conn.execute(
        "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
        "VALUES (?, ?, ?)",
        (user_id, "00000000-0000-0000-0000-000000000001", user_id),
    )
    db.conn.commit()
    db.close()

    legacy_path = fcs_mod.FCS_CHANNEL_SETUPS_FILE
    _write_legacy_json(legacy_path, {"FCS1": {
        "correlator": {"n_bins": 5, "n_casc": 40, "make_fine": False},
        "pairs": [{"name": "GG", "channel_a": "GG", "channel_b": "GG", "kind": "ACF"}],
    }})
    assert legacy_path.exists()

    try:
        # Run load with migration — this triggers migration + file deletion
        loaded = load_fcs_channel_setups(
            db_path=db_path, user_id=user_id, skip_migration=False,
        )
        # After successful migration, the legacy file should be gone
        assert not legacy_path.exists(), (
            "Legacy JSON file must be deleted after verified migration"
        )
        # Data from the legacy file must be present in the result
        assert "FCS1" in loaded["setups"]
    finally:
        fcs_mod.FCS_CHANNEL_SETUPS_FILE = orig_file
        fcs_mod._FCS_CONFIG_CACHE = None


def test_fcs_explicit_export_still_roundtrips(tmp_path: Path) -> None:
    """Explicit export via custom file_path still writes JSON and can be
    read back."""
    custom = tmp_path / "my_export.json"

    data = {
        "version": 1,
        "setups": {
            "Exported": {
                "correlator": {"n_bins": 3, "n_casc": 24, "make_fine": False},
                "pairs": [{"name": "GG", "channel_a": "GG", "channel_b": "GG", "kind": "ACF"}],
            },
        },
        "last_used_setup": "Exported",
    }

    ok = save_fcs_channel_setups(data, file_path=str(custom))
    assert ok
    assert custom.exists(), "Explicit export should create JSON file"

    raw = json.loads(custom.read_text())
    assert raw.get("version") == 1
    assert "Exported" in raw.get("setups", {})

    # Load it back — should go through JSON fallback path
    loaded = load_fcs_channel_setups(file_path=str(custom))
    assert "Exported" in loaded["setups"]
    assert loaded["setups"]["Exported"]["correlator"]["n_bins"] == 3
