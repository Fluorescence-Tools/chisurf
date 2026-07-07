"""Tests for MFDB dataset browser (PRD-10).

Asserts browsing, scoping, pagination, open, shifter round-trip, and GUI
construction.  Uses DI via in-process RPC client with a temp database path.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from chisurf.core.mfdb.security.auth import _hash_token
from chisurf.core.mfdb.repository import MFDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user_alice_token() -> str:
    return "alice-session-token"


@pytest.fixture
def user_bob_token() -> str:
    return "bob-session-token"


@pytest.fixture
def temp_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    user_alice_token: str,
    user_bob_token: str,
) -> Path:
    """Create a temp MFDB with Alice (admin, not admin) and Bob users."""
    db_path = tmp_path / "test_mfdb.db"
    db = MFDatabase(db_path)

    alice_hash = _hash_token(user_alice_token)
    bob_hash = _hash_token(user_bob_token)

    db.conn.execute(
        "INSERT INTO flr_sample_users (user_id, display_name, is_admin) VALUES (?, ?, ?)",
        ("alice", "Alice", 0),
    )
    db.conn.execute(
        "INSERT INTO flr_sample_users (user_id, display_name, is_admin) VALUES (?, ?, ?)",
        ("bob", "Bob", 0),
    )
    db.conn.execute(
        "INSERT INTO mfdb_session (session_id, user_id, token_hash, expires_at) VALUES (?, ?, ?, ?)",
        ("sess_alice", "alice", alice_hash, "2099-12-31T23:59:59"),
    )
    db.conn.execute(
        "INSERT INTO mfdb_session (session_id, user_id, token_hash, expires_at) VALUES (?, ?, ?, ?)",
        ("sess_bob", "bob", bob_hash, "2099-12-31T23:59:59"),
    )
    db.conn.commit()

    # Helper: register a raw_measurement artifact via direct repository call
    def _register_artifact(
        artifact_id: str,
        kind: str = "raw_measurement",
        data_format: str = "ptu",
        user_id: str = "alice",
        is_public: bool = False,
    ) -> None:
        db.register_artifact(
            artifact_id=artifact_id,
            artifact_kind=kind,
            data_format=data_format,
            storage_mode="local_file",
            file_path="/tmp/fake.ptu",
            created_by_user_id=user_id,
            is_public=is_public,
        )

    # Register test artifacts
    _register_artifact("art_alice_priv_01", user_id="alice", is_public=False)
    _register_artifact("art_alice_pub_01", user_id="alice", is_public=True)
    _register_artifact("art_alice_pub_02", user_id="alice", is_public=True)
    _register_artifact("art_bob_priv_01", user_id="bob", is_public=False)
    _register_artifact("art_bob_pub_01", user_id="bob", is_public=True)

    # Register an artifact with a different kind/format for filtering tests
    _register_artifact(
        "art_calib_01",
        kind="calibration_data",
        data_format="json",
        user_id="alice",
        is_public=True,
    )

    db.close()

    # Patch resolve_database_path in the services module to point to our temp db
    monkeypatch.setattr(
        "mfdb.admin.backend.services.resolve_database_path",
        lambda: db_path,
    )
    monkeypatch.setattr(
        "chisurf.core.mfdb.store.database_resolver.resolve_database_path",
        lambda: db_path,
    )

    return db_path


def _alice_auth(user_alice_token: str) -> dict:
    return {"token": user_alice_token}


def _bob_auth(user_bob_token: str) -> dict:
    return {"token": user_bob_token}


# ---------------------------------------------------------------------------
# datasets.browse — scope / pagination / filtering
# ---------------------------------------------------------------------------


def test_browse_mine_returns_only_active_user_datasets(
    temp_db: Path,
    user_alice_token: str,
) -> None:
    """'Mine' scope returns only artifacts owned by the authenticated user."""
    from mfdb.admin.backend.services import (
        datasets_browse_handler,
    )

    result = datasets_browse_handler(
        scope="own",
        auth=_alice_auth(user_alice_token),
    )
    datasets = result.get("datasets", [])
    ids = {d["artifact_id"] for d in datasets}
    assert "art_alice_priv_01" in ids
    assert "art_alice_pub_01" in ids
    assert "art_alice_pub_02" in ids
    assert "art_bob_priv_01" not in ids
    assert "art_bob_pub_01" not in ids


def test_browse_public_includes_public_datasets(
    temp_db: Path,
    user_alice_token: str,
) -> None:
    """'Public' scope returns only public artifacts regardless of owner."""
    from mfdb.admin.backend.services import (
        datasets_browse_handler,
    )

    result = datasets_browse_handler(
        scope="public",
        auth=_alice_auth(user_alice_token),
    )
    datasets = result.get("datasets", [])
    ids = {d["artifact_id"] for d in datasets}
    assert "art_alice_pub_01" in ids
    assert "art_alice_pub_02" in ids
    assert "art_bob_pub_01" in ids
    assert "art_alice_priv_01" not in ids
    assert "art_bob_priv_01" not in ids


def test_browse_all_includes_public_and_own(
    temp_db: Path,
    user_alice_token: str,
) -> None:
    """'All' scope returns public datasets + those owned by the user."""
    from mfdb.admin.backend.services import (
        datasets_browse_handler,
    )

    result = datasets_browse_handler(
        scope="all",
        auth=_alice_auth(user_alice_token),
    )
    datasets = result.get("datasets", [])
    ids = {d["artifact_id"] for d in datasets}
    assert "art_alice_priv_01" in ids  # owned
    assert "art_alice_pub_01" in ids  # owned + public
    assert "art_alice_pub_02" in ids
    assert "art_bob_pub_01" in ids  # public (not owned)
    assert "art_bob_priv_01" not in ids  # private and not owned


def test_browse_kinds_filter(
    temp_db: Path,
    user_alice_token: str,
) -> None:
    """kinds filter narrows results to matching artifact_kind."""
    from mfdb.admin.backend.services import (
        datasets_browse_handler,
    )

    result = datasets_browse_handler(
        scope="public",
        kinds=["calibration_data"],
        auth=_alice_auth(user_alice_token),
    )
    datasets = result.get("datasets", [])
    ids = {d["artifact_id"] for d in datasets}
    assert "art_calib_01" in ids
    assert "art_alice_pub_01" not in ids


def test_browse_formats_filter(
    temp_db: Path,
    user_alice_token: str,
) -> None:
    """formats filter narrows results to matching data_format."""
    from mfdb.admin.backend.services import (
        datasets_browse_handler,
    )

    result = datasets_browse_handler(
        scope="public",
        formats=["json"],
        auth=_alice_auth(user_alice_token),
    )
    datasets = result.get("datasets", [])
    ids = {d["artifact_id"] for d in datasets}
    assert "art_calib_01" in ids
    assert "art_alice_pub_01" not in ids


def test_browse_query_filter(
    temp_db: Path,
    user_alice_token: str,
) -> None:
    """query filters by artifact_id substring."""
    from mfdb.admin.backend.services import (
        datasets_browse_handler,
    )

    result = datasets_browse_handler(
        scope="public",
        query="pub_01",
        auth=_alice_auth(user_alice_token),
    )
    datasets = result.get("datasets", [])
    ids = {d["artifact_id"] for d in datasets}
    assert "art_alice_pub_01" in ids
    assert "art_alice_pub_02" not in ids


def test_browse_pagination(
    temp_db: Path,
    user_alice_token: str,
) -> None:
    """Pagination produces non-overlapping, correctly bounded pages."""
    from mfdb.admin.backend.services import (
        datasets_browse_handler,
    )

    # Page 1 (limit=3)
    r1 = datasets_browse_handler(
        scope="public",
        limit=3,
        offset=0,
        auth=_alice_auth(user_alice_token),
    )
    # Page 2 (limit=3)
    r2 = datasets_browse_handler(
        scope="public",
        limit=3,
        offset=3,
        auth=_alice_auth(user_alice_token),
    )

    d1 = r1.get("datasets", [])
    d2 = r2.get("datasets", [])
    total = r1.get("total", 0)

    # Pages do not overlap
    ids1 = {d["artifact_id"] for d in d1}
    ids2 = {d["artifact_id"] for d in d2}
    assert ids1.isdisjoint(ids2)

    # Total matches expected public count (4 public artifacts)
    assert total == 4

    # Combined pages cover all public datasets
    all_ids = ids1 | ids2
    assert "art_alice_pub_01" in all_ids
    assert "art_alice_pub_02" in all_ids
    assert "art_bob_pub_01" in all_ids
    assert "art_calib_01" in all_ids

    # Total returned on both pages
    assert r2.get("total") == total


# ---------------------------------------------------------------------------
# Two-user scoping
# ---------------------------------------------------------------------------


def test_user_b_cannot_see_a_private_dataset(
    temp_db: Path,
    user_bob_token: str,
) -> None:
    """Bob cannot see Alice's private dataset via 'all' scope."""
    from mfdb.admin.backend.services import (
        datasets_browse_handler,
    )

    result = datasets_browse_handler(
        scope="all",
        auth=_bob_auth(user_bob_token),
    )
    datasets = result.get("datasets", [])
    ids = {d["artifact_id"] for d in datasets}
    assert "art_alice_priv_01" not in ids
    assert "art_alice_pub_01" in ids  # public is visible


def test_user_b_can_see_a_public_dataset(
    temp_db: Path,
    user_bob_token: str,
) -> None:
    """Bob can see Alice's public dataset via 'public' scope."""
    from mfdb.admin.backend.services import (
        datasets_browse_handler,
    )

    result = datasets_browse_handler(
        scope="public",
        auth=_bob_auth(user_bob_token),
    )
    datasets = result.get("datasets", [])
    ids = {d["artifact_id"] for d in datasets}
    assert "art_alice_pub_01" in ids
    assert "art_bob_pub_01" in ids  # own datasets are also public


# ---------------------------------------------------------------------------
# datasets.open
# ---------------------------------------------------------------------------


def test_open_dataset_returns_local_path_for_object_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """datasets.open returns a readable local path for a registered object."""
    from mfdb.admin.backend.services import (
        datasets_open_handler,
    )

    obj_root = tmp_path / "obj_store"
    obj_root.mkdir()
    monkeypatch.setattr(
        "chisurf.core.mfdb.store.database_resolver.object_store_root",
        lambda: obj_root,
    )

    db_path = tmp_path / "open_test.db"
    db = MFDatabase(db_path)

    monkeypatch.setattr(
        "mfdb.admin.backend.services.resolve_database_path",
        lambda: db_path,
    )
    monkeypatch.setattr(
        "chisurf.core.mfdb.store.database_resolver.resolve_database_path",
        lambda: db_path,
    )

    # Register a session + user
    tok = "open-test-token"
    tok_hash = _hash_token(tok)
    db.conn.execute(
        "INSERT INTO flr_sample_users (user_id, display_name, is_admin) VALUES (?, ?, ?)",
        ("open_user", "Open User", 0),
    )
    db.conn.execute(
        "INSERT INTO mfdb_session (session_id, user_id, token_hash, expires_at) VALUES (?, ?, ?, ?)",
        ("sess_open", "open_user", tok_hash, "2099-12-31T23:59:59"),
    )
    db.conn.commit()

    # Store a real file in the object store
    content = b"fake tttr data for open test"
    store = db._get_object_store()
    ref = store.put_bytes(content, filename="test.ptu")
    db.conn.execute(
        "INSERT INTO mfdb_object (object_uuid, content_md5, original_filename, "
        "size_bytes, storage_path, refcount) VALUES (?, ?, ?, ?, ?, 1)",
        (ref.uuid, ref.md5, "test.ptu", ref.size, ref.storage_path),
    )
    db.conn.commit()

    # Register an artifact pointing to this object
    art_id = "art_open_01"
    db.register_artifact(
        artifact_id=art_id,
        artifact_kind="raw_measurement",
        data_format="ptu",
        storage_mode="managed_archive",
        object_uuid=ref.uuid,
        created_by_user_id="open_user",
        is_public=False,
    )
    db.close()

    result = datasets_open_handler(
        artifact_id=art_id,
        auth={"token": tok},
    )
    local_path = result.get("local_path")
    assert local_path is not None
    assert Path(local_path).exists()
    assert Path(local_path).read_bytes() == content


# ---------------------------------------------------------------------------
# Shifter round-trip
# ---------------------------------------------------------------------------


def test_shifter_round_trip(
    tmp_path: Path,
    user_alice_token: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register a TTTR raw_measurement, browse lists it, open reads it."""
    from mfdb.admin.backend.services import (
        datasets_browse_handler,
        datasets_open_handler,
    )

    obj_root = tmp_path / "obj_store"
    obj_root.mkdir()
    monkeypatch.setattr(
        "chisurf.core.mfdb.store.database_resolver.object_store_root",
        lambda: obj_root,
    )

    db_path = tmp_path / "shifter_rt.db"
    db = MFDatabase(db_path)

    monkeypatch.setattr(
        "mfdb.admin.backend.services.resolve_database_path",
        lambda: db_path,
    )
    monkeypatch.setattr(
        "chisurf.core.mfdb.store.database_resolver.resolve_database_path",
        lambda: db_path,
    )

    # Register Alice (user_default already bootstrapped by ensure_schema)
    tok_hash = _hash_token(user_alice_token)
    db.conn.execute(
        "INSERT INTO flr_sample_users (user_id, display_name, is_admin) VALUES (?, ?, ?)",
        ("alice", "Alice", 0),
    )
    db.conn.execute(
        "INSERT INTO mfdb_session (session_id, user_id, token_hash, expires_at) VALUES (?, ?, ?, ?)",
        ("sess_alice2", "alice", tok_hash, "2099-12-31T23:59:59"),
    )
    db.conn.commit()

    # Patch _resolve_active_user_id to return alice for this test
    monkeypatch.setattr(
        "chisurf.core.mfdb.provenance.result_registry._resolve_active_user_id",
        lambda: "alice",
    )

    # Create a real file and register as raw_measurement
    content = b"fake TTTR data for shifter round trip"
    src_file = tmp_path / "test_input.ptu"
    src_file.write_bytes(content)

    from chisurf.core.mfdb.provenance.result_registry import register_raw_measurement
    art_id = register_raw_measurement(
        file_path=str(src_file),
        db=db,
        is_public=False,
    )
    assert art_id, "register_raw_measurement should return an artifact_id"

    # Verify browse (mine) lists it
    browse_result = datasets_browse_handler(
        scope="own",
        auth=_alice_auth(user_alice_token),
    )
    ds_ids = {d["artifact_id"] for d in browse_result.get("datasets", [])}
    assert art_id in ds_ids

    # Verify open returns readable content
    open_result = datasets_open_handler(
        artifact_id=art_id,
        auth=_alice_auth(user_alice_token),
    )
    local_path = open_result.get("local_path")
    assert local_path is not None
    assert Path(local_path).exists()
    assert Path(local_path).read_bytes() == content

    db.close()


# ---------------------------------------------------------------------------
# Regression: processed_data via register_result appears in browse
# ---------------------------------------------------------------------------


def test_processed_data_appears_in_browse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register a processed_data artifact via register_result with a
    non-default user and verify it appears in browse_datasets.

    This guards against silent registration failures (Bug C).
    """
    from mfdb.admin.backend.services import (
        datasets_browse_handler,
    )
    from chisurf.core.mfdb.provenance.result_registry import register_result

    obj_root = tmp_path / "obj_store"
    obj_root.mkdir()
    monkeypatch.setattr(
        "chisurf.core.mfdb.store.database_resolver.object_store_root",
        lambda: obj_root,
    )

    db_path = tmp_path / "regr_processed.db"
    db = MFDatabase(db_path)

    monkeypatch.setattr(
        "mfdb.admin.backend.services.resolve_database_path",
        lambda: db_path,
    )
    monkeypatch.setattr(
        "chisurf.core.mfdb.store.database_resolver.resolve_database_path",
        lambda: db_path,
    )

    # Create a non-default user (not "user_default", not "guest")
    tok = "regr-user-token"
    tok_hash = _hash_token(tok)
    db.conn.execute(
        "INSERT INTO flr_sample_users (user_id, display_name, is_admin) VALUES (?, ?, ?)",
        ("regr_user", "Regression User", 0),
    )
    db.conn.execute(
        "INSERT INTO mfdb_session (session_id, user_id, token_hash, expires_at) VALUES (?, ?, ?, ?)",
        ("sess_regr", "regr_user", tok_hash, "2099-12-31T23:59:59"),
    )
    db.conn.commit()

    # Patch _resolve_active_user_id to return our test user
    monkeypatch.setattr(
        "chisurf.core.mfdb.provenance.result_registry._resolve_active_user_id",
        lambda: "regr_user",
    )

    # Register processed_data with operation_type="microtime_shift"
    content = b'{"processed": true}'
    art_id = register_result(
        kind="processed_data",
        data=content,
        operation_type="microtime_shift",
        db=db,
    )
    assert art_id, "register_result should return a non-empty artifact_id"

    # Verify it appears in browse for the owning user
    auth = {"token": tok}
    browse_result = datasets_browse_handler(scope="own", auth=auth)
    ds_ids = {d["artifact_id"] for d in browse_result.get("datasets", [])}
    assert art_id in ds_ids, (
        f"processed_data artifact {art_id} should appear in browse(scope='own') "
        f"for user regr_user; got {ds_ids}"
    )

    # Also verify it appears in scope="all" for the same user
    browse_all = datasets_browse_handler(scope="all", auth=auth)
    all_ids = {d["artifact_id"] for d in browse_all.get("datasets", [])}
    assert art_id in all_ids

    db.close()


def test_register_result_fails_loud_on_real_error(tmp_path: Path) -> None:
    """register_result raises when the DB is present but a real error occurs.

    A vocabulary violation (bad operation_type) must propagate, not silently
    return "".
    """
    from chisurf.core.mfdb.provenance.result_registry import register_result

    db_path = tmp_path / "loud_fail.db"
    db = MFDatabase(db_path)

    # Register with a bogus operation_type that is NOT in OPERATION_TYPES
    with pytest.raises(Exception, match="operation_type"):
        register_result(
            kind="processed_data",
            data=b"test",
            operation_type="__nonexistent_optype__",
            db=db,
        )

    db.close()


# ---------------------------------------------------------------------------
# GUI construction smoke test (mirrors test/gui/test_detector_wizard_page.py)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    True,
    reason="Qt smoke test disabled by default; "
    "run manually with QT_QPA_PLATFORM=offscreen pytest ...",
)
def test_mfdb_dataset_browser_constructs() -> None:
    """MfdbDatasetBrowser constructs without crashing (no client → disabled)."""
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    from chisurf.gui.widgets.mfdb.dataset_browser import MfdbDatasetBrowser

    browser = MfdbDatasetBrowser(client=None)
    assert browser is not None
    assert browser._is_connected() is False
    assert browser.status_label.text() == "MFDB not connected"
    browser.close()


@pytest.mark.skipif(
    True,
    reason="Qt smoke test disabled by default; "
    "run manually with QT_QPA_PLATFORM=offscreen pytest ...",
)
def test_mfdb_dataset_picker_dialog_returns_none_when_no_client() -> None:
    """pick_dataset returns None when client is None."""
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    from chisurf.gui.widgets.mfdb.dataset_browser import (
        MfdbDatasetPickerDialog,
    )

    result = MfdbDatasetPickerDialog.pick_dataset(client=None)
    assert result is None


def test_processed_dataset_with_unseeded_user_registers_and_browses(tmp_path, monkeypatch):
    """Regression (PRD-10 bug): a processed dataset must register and appear in
    the browser even when the configured active user was never pre-seeded and the
    operation type is plugin-specific.

    Covers two silent-failure bugs:
      * Bug A: operation_type='microtime_shift' must be a valid vocab value.
      * Bug B: a configured default_user_id with no flr_sample_users row must not
        fail the created_by_user_id foreign key (ensure_user bootstraps it).
    """
    import chisurf.core.settings
    from chisurf.core.mfdb.provenance import result_registry as rr

    # Active user that is NOT pre-seeded in flr_sample_users (config injection).
    monkeypatch.setitem(
        chisurf.core.settings.cs_settings, "mfdb", {"default_user_id": "scientist_x"}
    )

    db = MFDatabase(str(tmp_path / "reg.db"))
    src = tmp_path / "in.ptu"  # a valid raw data_format; the bug under test is the
    src.write_text("hello")     # unseeded user + plugin operation_type, not the format

    raw = rr.register_raw_measurement(file_path=str(src), db=db)
    assert raw, "raw must register even when the active user is not pre-seeded"

    proc = rr.register_result(
        kind="processed_data",
        data={"x": [1, 2, 3]},
        parent_artifact_id=raw,
        operation_type="microtime_shift",
        db=db,
    )
    assert proc, "processed_data with operation_type='microtime_shift' must register"

    kinds = [
        d["artifact_kind"]
        for d in db.browse_datasets(scope="own", owner_id="scientist_x")["datasets"]
    ]
    assert "processed_data" in kinds
    assert "raw_measurement" in kinds


def test_browse_datasets_format_filter_normalizes_dot(tmp_path):
    """Regression: browse_datasets must match the dot-less stored data_format
    whether the caller passes 'ptu', '.ptu', or '.PTU'. The shifter's MFDB
    picker passed dotted formats, so it always returned zero datasets and the
    load-from-MFDB roundtrip was broken."""
    from chisurf.core.mfdb.provenance import result_registry as rr

    db = MFDatabase(str(tmp_path / "fmt.db"))
    f = tmp_path / "meas.ptu"
    f.write_bytes(b"PQTTTRdata")
    raw = rr.register_raw_measurement(file_path=str(f), db=db)
    assert raw
    uid = rr._resolve_active_user_id()

    row = db.conn.execute(
        "SELECT data_format FROM mfdb_artifact WHERE artifact_id=?", (raw,)
    ).fetchone()
    assert row[0] == "ptu"  # stored without the dot

    for fmts in (["ptu"], [".ptu"], [".PTU"], ["PTU"]):
        n = len(db.browse_datasets(scope="own", owner_id=uid, formats=fmts)["datasets"])
        assert n == 1, f"formats={fmts} should match the stored 'ptu' (got {n})"


def test_browse_handler_own_scope_uses_default_user_when_anonymous(tmp_path, monkeypatch):
    """Regression: with no auth session (in-process GUI client), the 'own' scope
    must fall back to the configured default_user_id so it matches the owner that
    registration stamps. Otherwise 'Mine' shows nothing despite registered data."""
    import chisurf.core.settings
    from chisurf.core.mfdb.provenance import result_registry as rr
    from mfdb.admin.backend import services as svc

    monkeypatch.setitem(
        chisurf.core.settings.cs_settings, "mfdb", {"default_user_id": "tpeulen"}
    )
    dbp = str(tmp_path / "own.db")
    db = MFDatabase(dbp)
    f = tmp_path / "a.spc"
    f.write_bytes(b"spc")
    assert rr.register_raw_measurement(file_path=str(f), db=db)
    db.close()

    monkeypatch.setattr(svc, "resolve_database_path", lambda: dbp)
    # auth=None -> anonymous principal -> must fall back to default_user_id
    r = svc.datasets_browse_handler(scope="own", kinds=["raw_measurement"], auth=None)
    assert r["total"] == 1, "own scope must match the registration owner when anonymous"


def test_real_mfdbclient_call_browses_datasets(tmp_path, monkeypatch):
    """Integration regression: the real MFDBClient must expose ``call`` and
    return datasets through the in-process dispatcher.

    The browser widget and shifter use ``client.call(...)``; MFDBClient only
    had ``_call``, so every browse raised AttributeError (swallowed) and the
    picker showed 0 — even though the backend handler worked. This exercises the
    real client end to end.
    """
    import chisurf.core.settings
    from chisurf.core.mfdb.provenance import result_registry as rr
    import chisurf.core.mfdb.store.database_resolver as dr
    from mfdb.admin.backend import services as svc
    from mfdb.admin.gui.client import MFDBClient

    monkeypatch.setitem(
        chisurf.core.settings.cs_settings, "mfdb", {"default_user_id": "tpeulen"}
    )
    dbp = str(tmp_path / "client.db")
    db = MFDatabase(dbp)
    f = tmp_path / "m.ptu"
    f.write_bytes(b"data")
    assert rr.register_raw_measurement(file_path=str(f), db=db)
    db.close()
    monkeypatch.setattr(dr, "resolve_database_path", lambda: dbp)
    monkeypatch.setattr(svc, "resolve_database_path", lambda: dbp)

    client = MFDBClient(inprocess=True)
    assert hasattr(client, "call"), "MFDBClient must expose a public call()"
    res = client.call(
        "mfdb.datasets.browse",
        {"scope": "own", "kinds": ["raw_measurement"], "formats": ["ptu"]},
    )
    assert isinstance(res, dict)
    assert res.get("total") == 1
    assert len(res.get("datasets", [])) == 1


def test_datasets_open_allows_anonymous_with_default_user(tmp_path, monkeypatch):
    """Regression: datasets.open must not require an authenticated session when a
    default user is configured (the in-process GUI client is anonymous). It
    previously failed with 'Authentication required', breaking the load."""
    import chisurf.core.settings
    from chisurf.core.mfdb.provenance import result_registry as rr
    from mfdb.admin.backend import services as svc

    monkeypatch.setitem(
        chisurf.core.settings.cs_settings, "mfdb", {"default_user_id": "tpeulen"}
    )
    dbp = str(tmp_path / "open.db")
    db = MFDatabase(dbp)
    f = tmp_path / "m.ptu"
    f.write_bytes(b"payload-bytes")
    art = rr.register_raw_measurement(file_path=str(f), db=db)
    db.close()
    monkeypatch.setattr(svc, "resolve_database_path", lambda: dbp)

    res = svc.datasets_open_handler(artifact_id=art, auth=None)  # anonymous
    assert res.get("local_path"), "anonymous open must return a local path"


def test_browse_datasets_joins_sample_name_and_refcount(
    temp_db: Path,
) -> None:
    """browse_datasets returns sample_name and object_refcount if associated."""
    db = MFDatabase(temp_db)

    # 1. Insert a sample. flr_sample is the flrCIF/pdbx-canonical table the
    #    browser joins; the name lives in its ``description`` field.
    db.conn.execute(
        "INSERT INTO flr_sample (sample_id, description) VALUES (?, ?)",
        ("sample_xxx", "Super Cool Protein Sample"),
    )

    # 2. Insert an object (for refcount and original_filename)
    db.conn.execute(
        "INSERT INTO mfdb_object (object_uuid, content_md5, storage_path, refcount, original_filename) VALUES (?, ?, ?, ?, ?)",
        ("obj_uuid_123", "md5_content_123", "/tmp/nonexistent", 5, "my_original_file.ptu"),
    )

    # 3. Register artifact associated with the object
    db.register_artifact(
        artifact_id="art_with_sample_01",
        artifact_kind="raw_measurement",
        data_format="ptu",
        storage_mode="local_file",
        file_path="/tmp/fake.ptu",
        created_by_user_id="alice",
        is_public=True,
    )
    db.conn.execute(
        "UPDATE mfdb_artifact SET object_uuid = ? WHERE artifact_id = ?",
        ("obj_uuid_123", "art_with_sample_01"),
    )

    # 4. Link artifact to sample via edge
    db.conn.execute(
        "INSERT INTO mfdb_edge (source_node_type, source_node_id, target_node_type, target_node_id, relationship_type) VALUES (?, ?, ?, ?, ?)",
        ("artifact", "art_with_sample_01", "sample", "sample_xxx", "measured_sample"),
    )
    db.conn.commit()

    # Now query browse_datasets
    result = db.browse_datasets(scope="all")
    datasets = result.get("datasets", [])

    # Find our artifact
    target_ds = None
    for ds in datasets:
        if ds.get("artifact_id") == "art_with_sample_01":
            target_ds = ds
            break

    assert target_ds is not None
    assert target_ds.get("sample_name") == "Super Cool Protein Sample"
    assert target_ds.get("object_refcount") == 5
    assert target_ds.get("original_filename") == "my_original_file.ptu"

    # Check that another artifact without associations has NA/None or default
    other_ds = None
    for ds in datasets:
        if ds.get("artifact_id") == "art_alice_pub_01":
            other_ds = ds
            break

    assert other_ds is not None
    assert other_ds.get("sample_name") is None
    assert other_ds.get("object_refcount") is None



def test_name_only_sample_appears_in_flr_sample_and_list(tmp_path):
    """Regression: a sample created with only a name (no description) must carry
    that name into flr_sample.description (the flrCIF/pdbx-canonical table), so it
    is not nameless in list_samples / search / browse. Previously the name lived
    only in mfdb_sample.display_name and flr_sample.description was empty."""
    from chisurf.core.mfdb.samples.sample_manager import create_sample
    from chisurf.core.mfdb.samples.sample_requests import SampleDefinition

    db = MFDatabase(str(tmp_path / "s.db"))
    sid = create_sample(db, SampleDefinition(name="DNA-Al488-Cy5"))

    row = db.conn.execute(
        "SELECT description FROM flr_sample WHERE sample_id = ?", (sid,)
    ).fetchone()
    assert row["description"] == "DNA-Al488-Cy5"
    names = [r["description"] for r in db.list_samples()]
    assert "DNA-Al488-Cy5" in names


# Note: the former test_backfill_fills_empty_flr_sample_description test simulated the
# legacy state by inserting into mfdb_sample and backfilling flr_sample from it. PRD-19
# collapsed that duplicate — mfdb_sample no longer exists (flr_sample is the single
# source of truth) — so the backfill-from-mfdb_sample path is gone.


def test_browse_datasets_excludes_grouped_members(
    temp_db: Path,
) -> None:
    """browse_datasets only returns the group artifact, not its member files."""
    db = MFDatabase(temp_db)

    # 1. Register a group artifact
    db.register_artifact(
        artifact_id="art_group_01",
        artifact_kind="raw_measurement",
        data_format="ptu",
        storage_mode="local_file",
        file_path="/tmp/group.ptu",
        created_by_user_id="alice",
        is_public=True,
    )

    # 2. Register member artifacts
    db.register_artifact(
        artifact_id="art_member_01",
        artifact_kind="raw_measurement",
        data_format="ptu",
        storage_mode="local_file",
        file_path="/tmp/member1.ptu",
        created_by_user_id="alice",
        is_public=True,
    )
    db.register_artifact(
        artifact_id="art_member_02",
        artifact_kind="raw_measurement",
        data_format="ptu",
        storage_mode="local_file",
        file_path="/tmp/member2.ptu",
        created_by_user_id="alice",
        is_public=True,
    )

    # 3. Create grouped_in edges: member -> grouped_in -> group
    db.conn.execute(
        "INSERT INTO mfdb_edge (source_node_type, source_node_id, target_node_type, target_node_id, relationship_type) VALUES (?, ?, ?, ?, ?)",
        ("artifact", "art_member_01", "artifact", "art_group_01", "grouped_in"),
    )
    db.conn.execute(
        "INSERT INTO mfdb_edge (source_node_type, source_node_id, target_node_type, target_node_id, relationship_type) VALUES (?, ?, ?, ?, ?)",
        ("artifact", "art_member_02", "artifact", "art_group_01", "grouped_in"),
    )
    db.conn.commit()

    # Now query browse_datasets
    result = db.browse_datasets(scope="all")
    datasets = result.get("datasets", [])
    ids = {d["artifact_id"] for d in datasets}

    # Verify group is present, but members are excluded
    assert "art_group_01" in ids
    assert "art_member_01" not in ids
    assert "art_member_02" not in ids



def test_multi_owner_browse_and_dict_mapping(tmp_path, monkeypatch):
    """A dataset can be co-owned: each owner sees it under 'own'; non-owners do
    not. The mfdb_artifact_owner items map to live columns (dict-driven)."""
    import chisurf.core.settings
    from chisurf.core.mfdb.provenance import result_registry as rr
    from chisurf.core.mfdb.schema.dictionary_schema_map import build_dictionary_schema_map

    monkeypatch.setitem(
        chisurf.core.settings.cs_settings, "mfdb", {"default_user_id": "alice"}
    )
    dbp = str(tmp_path / "mo.db")
    db = MFDatabase(dbp)
    f = tmp_path / "x.ptu"
    f.write_bytes(b"shared")
    art = rr.register_raw_measurement(file_path=str(f), db=db)

    assert db.list_artifact_owners(art) == ["alice"]            # creator owns
    assert db.browse_datasets(scope="own", owner_id="alice")["total"] == 1
    assert db.browse_datasets(scope="own", owner_id="bob")["total"] == 0

    db.add_artifact_owner(art, "bob")                            # co-own
    assert set(db.list_artifact_owners(art)) == {"alice", "bob"}
    assert db.browse_datasets(scope="own", owner_id="bob")["total"] == 1
    assert db.browse_datasets(scope="own", owner_id="carol")["total"] == 0

    # add_artifact_owner is idempotent
    db.add_artifact_owner(art, "bob")
    assert db.list_artifact_owners(art).count("bob") == 1

    mapper = build_dictionary_schema_map(dbp)
    unmapped = [u.dictionary_name for u in mapper.get_unmapped_flr_items()
                if u.category == "mfdb_artifact_owner"]
    assert unmapped == []


# Note: the former test_v39_backfills_artifact_owner_from_creator test drove the removed
# version-chain migration (set_schema_version(38) + migrate_schema → v39 owner backfill).
# PRD-19 deleted the version chain (pre-PRD-19 DBs are disposable). On the current path
# register_result records the owner in mfdb_artifact_owner directly (covered by the
# scope/own browse tests above), so no migration backfill is needed.
