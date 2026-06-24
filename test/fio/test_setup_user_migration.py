"""Tests for per-user detector-setup migration and ownership (PRD-04 Sidequest S1-S5).

Ensures:
- S5-1: Two users with same-named setups keep distinct, non-colliding rows.
- S5-2: Per-user migration from ``detector_setups.json`` is idempotent.
- S5-3: ``load_detector_setups()`` for user A excludes user B's private setups.
- S5-4: A setup saved by a logged-in user records that user as owner.
- S5-5: No-login/default-user path still migrates and loads setups.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from chisurf.core.mfdb.database_resolver import resolve_database_path
from chisurf.core.mfdb.repository import MFDatabase
from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups import (
    _load_mfdb_detector_setups,
    _migrate_json_setups_to_mfdb,
    _resolve_active_user_id,
    _save_setup_row,
    load_detector_setups,
    save_detector_setups,
    setup_id_for_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_db(tmp_path: Path, name: str = "test.db") -> MFDatabase:
    """Open a fresh MFDatabase at a temporary path."""
    db = MFDatabase(str(tmp_path / name))
    return db


def _write_legacy_json(path: Path, setups: dict, last_used: str = "") -> None:
    """Write a fake ``detector_setups.json`` file."""
    data: dict = {"setups": setups}
    if last_used:
        data["last_used"] = last_used
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# S5-1: Two users with same-named setups don't collide
# ---------------------------------------------------------------------------


def test_same_name_different_users_no_collision(tmp_path: Path) -> None:
    """Two users can each have a setup named 'MFD' without overwriting."""
    db = _fresh_db(tmp_path)
    try:
        alice = "user_alice"
        bob = "user_bob"
        # Bootstrap test users
        for uid in (alice, bob):
            db.conn.execute(
                "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
                "VALUES (?, ?, ?)",
                (uid, f"00000000-0000-0000-0000-{uid.replace('user_', '')}", uid),
            )

        alice_data = {
            "detectors": {"green": {"chs": [0, 8]}},
            "windows": {"prompt": (0, 2048)},
        }
        bob_data = {
            "detectors": {"red": {"chs": [1, 9]}},
            "windows": {"delayed": (2048, 4096)},
        }

        _save_setup_row(db, "MFD", alice_data, user_id=alice)
        _save_setup_row(db, "MFD", bob_data, user_id=bob)

        alice_id = setup_id_for_name("MFD", user_id=alice)
        bob_id = setup_id_for_name("MFD", user_id=bob)

        assert alice_id != bob_id, "Setup IDs should differ across users"

        alice_row = db.get_setup(alice_id)
        bob_row = db.get_setup(bob_id)

        assert alice_row is not None
        assert bob_row is not None
        assert alice_row["created_by_user_id"] == alice
        assert bob_row["created_by_user_id"] == bob
        assert alice_row["setup_id"] != bob_row["setup_id"]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# S5-2: Per-user migration is idempotent
# ---------------------------------------------------------------------------


def test_per_user_migration_idempotent(tmp_path: Path) -> None:
    """Re-running migration for the same user adds nothing."""
    db = _fresh_db(tmp_path)
    try:
        user_id = "user_default"
        legacy = tmp_path / "detector_setups.json"
        _write_legacy_json(
            legacy,
            {"My Setup": {"detectors": {"green": {"chs": [0]}}, "windows": {"p": (0, 1)}}},
            last_used="My Setup",
        )

        # First migration
        _migrate_json_setups_to_mfdb(db, legacy, user_id=user_id)
        count_after_first = len(db.list_setups())

        # Second migration — should be idempotent
        _migrate_json_setups_to_mfdb(db, legacy, user_id=user_id)
        count_after_second = len(db.list_setups())

        assert count_after_first == count_after_second, (
            f"Migration added {count_after_second - count_after_first} extra rows"
        )
        assert count_after_first >= 1
    finally:
        db.close()


def test_different_user_gets_separate_migration(tmp_path: Path) -> None:
    """Migrating for user B when user A already imported adds user B's setups."""
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

        legacy = tmp_path / "detector_setups.json"
        _write_legacy_json(legacy, {"Shared": {"detectors": {}, "windows": {}}})

        _migrate_json_setups_to_mfdb(db, legacy, user_id=alice)
        alice_count = len(
            [s for s in db.list_setups() if s.get("created_by_user_id") == alice]
        )

        _migrate_json_setups_to_mfdb(db, legacy, user_id=bob)
        bob_count = len(
            [s for s in db.list_setups() if s.get("created_by_user_id") == bob]
        )
        total = len(db.list_setups())

        assert alice_count >= 1, "Alice should have imported setups"
        assert bob_count >= 1, "Bob should have imported setups"
        assert total >= 2, "Both users' setups should coexist"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# S5-3: load_detector_setups for user A excludes user B's private setups
# ---------------------------------------------------------------------------


def test_load_excludes_other_users_private_setups(tmp_path: Path) -> None:
    """``load_detector_setups()`` scoped to user A does not return user B's."""
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

        # Private (non-public) setups — only visible to the owning user
        _save_setup_row(db, "AliceOnly", {"detectors": {}, "windows": {}}, user_id=alice, is_public=False)
        _save_setup_row(db, "BobOnly", {"detectors": {}, "windows": {}}, user_id=bob, is_public=False)

        alice_setups = _load_mfdb_detector_setups(db, user_id=alice)["setups"]
        bob_setups = _load_mfdb_detector_setups(db, user_id=bob)["setups"]

        assert "AliceOnly" in alice_setups
        assert "BobOnly" not in alice_setups, "Alice sees Bob's private setup"
        assert "BobOnly" in bob_setups
        assert "AliceOnly" not in bob_setups, "Bob sees Alice's private setup"
    finally:
        db.close()


def test_load_includes_shared_and_public_setups(tmp_path: Path) -> None:
    """Both users see shared/builtin and public setups."""
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

        # Shared setup (no owner) — save with empty user_id so created_by_user_id is NULL
        _save_setup_row(db, "Shared", {"detectors": {}, "windows": {}}, user_id="")
        # Alice's public setup
        _save_setup_row(db, "AlicePublic", {"detectors": {}, "windows": {}}, user_id=alice, is_public=True)
        # Alice's private setup
        _save_setup_row(db, "AlicePrivate", {"detectors": {}, "windows": {}}, user_id=alice, is_public=False)

        alice_setups = _load_mfdb_detector_setups(db, user_id=alice)["setups"]
        bob_setups = _load_mfdb_detector_setups(db, user_id=bob)["setups"]

        assert "Shared" in alice_setups
        assert "Shared" in bob_setups
        assert "AlicePublic" in alice_setups
        assert "AlicePublic" in bob_setups, "Bob should see Alice's public setup"
        assert "AlicePrivate" in alice_setups
        assert "AlicePrivate" not in bob_setups, "Bob should not see Alice's private setup"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# S5-4: A setup saved by a logged-in user records the owner
# ---------------------------------------------------------------------------


def test_save_records_owner(tmp_path: Path) -> None:
    """``save_detector_setups`` called for a user stores ``created_by_user_id``."""
    db = _fresh_db(tmp_path)
    try:
        user_id = "user_default"
        import chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups as dsu
        import chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_setup_utils as utils

        original_resolve = dsu._resolve_active_user_id
        dsu._resolve_active_user_id = lambda: user_id
        utils.resolve_active_user_id = lambda: user_id
        original_db = dsu._db
        dsu._db = lambda: db
        original_get_db = utils.get_db
        utils.get_db = lambda: db
        try:
            data = {
                "setups": {
                    "MySetup": {"detectors": {"green": {"chs": [0]}}, "windows": {}},
                },
                "last_used": "MySetup",
            }
            save_detector_setups(data)

            setup_id = setup_id_for_name("MySetup", user_id=user_id)
            row = db.get_setup(setup_id)
            assert row is not None
            assert row["created_by_user_id"] == user_id
        finally:
            dsu._resolve_active_user_id = original_resolve
            dsu._db = original_db
            utils.resolve_active_user_id = original_resolve
            utils.get_db = original_get_db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# S5-5: No-login / default-user path still works
# ---------------------------------------------------------------------------


def test_default_user_migration_and_load(tmp_path: Path) -> None:
    """Single-user / no-login install still migrates and loads setups."""
    db = _fresh_db(tmp_path)
    try:
        user_id = "user_default"
        legacy = tmp_path / "detector_setups.json"
        _write_legacy_json(
            legacy,
            {"Default": {"detectors": {}, "windows": {"p": (0, 1)}}},
        )

        # Migrate via the internal function (simulates the default-user path)
        _migrate_json_setups_to_mfdb(db, legacy, user_id=user_id)
        loaded = _load_mfdb_detector_setups(db, user_id=user_id)

        assert "Default" in loaded.get("setups", {}), (
            "Default user should be able to load setups after migration"
        )

        # Verify owner is recorded as user_default
        setup_id = setup_id_for_name("Default", user_id=user_id)
        row = db.get_setup(setup_id)
        assert row is not None
        assert row["created_by_user_id"] == user_id
    finally:
        db.close()


def test_save_no_login_still_works(tmp_path: Path) -> None:
    """Saving setups without a logged-in user succeeds and creates shared rows."""
    db = _fresh_db(tmp_path)
    try:
        _save_setup_row(db, "NoLogin", {"detectors": {}, "windows": {"w": (0, 1)}}, user_id="")

        setup_id = setup_id_for_name("NoLogin", user_id="")
        row = db.get_setup(setup_id)
        assert row is not None, "Setup should exist after save"
        assert row["created_by_user_id"] is None, (
            "No-login save should leave created_by_user_id NULL"
        )
    finally:
        db.close()
