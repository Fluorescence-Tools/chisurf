"""Tests for MFDB authentication and authorization."""

from __future__ import annotations

from pathlib import Path

import pytest

from chisurf.core.mfdb.auth import (
    PERM_MANAGE,
    PERM_READ,
    PERM_WRITE,
    AnonymousPrincipal,
    AuthError,
    AuthenticatedPrincipal,
    PermissionDenied,
    authenticate_token,
    can_access,
    chgrp,
    chmod,
    chown,
    create_default_acl_for_object,
    create_session,
    filter_readable,
    grant_acl,
    inherit_acl_from_parent,
    is_throttled,
    list_sessions,
    principal_from_rpc_auth,
    record_auth_attempt,
    require_access,
    require_authenticated,
    revoke_acl,
    revoke_session,
    revoke_session_by_token,
)
from chisurf.core.mfdb.repository import MFDatabase


@pytest.fixture
def db(tmp_path: Path):
    """Create a fresh MFDatabase with auth tables."""
    path = tmp_path / "auth_test.db"
    db = MFDatabase(path)
    yield db
    db.close()


@pytest.fixture
def patch_db(monkeypatch, db):
    """Monkeypatch auth_services to use the test database."""
    from mfdb.admin.backend import auth_services
    from contextlib import contextmanager

    @contextmanager
    def _fake_get_db():
        yield db

    monkeypatch.setattr(auth_services, '_get_db', _fake_get_db)


@pytest.fixture
def admin_user(db):
    """Create an admin user."""
    db.add_user("admin_user", display_name="Admin User", is_admin=1)
    return "admin_user"


@pytest.fixture
def normal_user(db):
    """Create a normal user."""
    db.add_user("normal_user", display_name="Normal User", is_admin=0)
    return "normal_user"


def _make_user(db, user_id, display_name=None, is_admin=0):
    db.conn.execute(
        "INSERT OR IGNORE INTO flr_sample_users (user_id, display_name, is_admin) VALUES (?, ?, ?)",
        (user_id, display_name or user_id, is_admin),
    )
    db.conn.commit()
    return user_id


# ---- Fresh schema tests ----


def test_fresh_schema_creates_auth_tables(tmp_path: Path) -> None:
    path = tmp_path / "fresh_auth.db"
    with MFDatabase(path) as db:
        tables = {
            r[0] for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for t in ("mfdb_group", "mfdb_group_member", "mfdb_object_acl",
                   "mfdb_acl_entry", "mfdb_session", "mfdb_auth_attempt"):
            assert t in tables, f"Missing table: {t}"


def test_fresh_schema_bootstraps_groups(tmp_path: Path) -> None:
    path = tmp_path / "fresh_bootstrap.db"
    with MFDatabase(path) as db:
        groups = db.conn.execute(
            "SELECT group_id FROM mfdb_group WHERE deleted_at IS NULL"
        ).fetchall()
        group_ids = {r[0] for r in groups}
        assert "admins" in group_ids
        assert "users" in group_ids
        assert "public" in group_ids


def test_fresh_schema_adds_existing_users_to_groups(tmp_path: Path) -> None:
    path = tmp_path / "fresh_users.db"
    with MFDatabase(path) as db:
        members = db.conn.execute(
            "SELECT user_id FROM mfdb_group_member WHERE group_id = 'users' AND deleted_at IS NULL"
        ).fetchall()
        member_ids = {r[0] for r in members}
        assert "user_default" in member_ids


# ---- Principal tests ----


def test_anonymous_principal() -> None:
    p = AnonymousPrincipal()
    assert p.user_id is None
    assert not p.is_authenticated
    assert not p.is_admin


def test_authenticated_principal() -> None:
    p = AuthenticatedPrincipal("test_user")
    assert p.user_id == "test_user"
    assert p.is_authenticated
    assert not p.is_admin


def test_admin_principal() -> None:
    p = AuthenticatedPrincipal("admin_user", is_admin=True)
    assert p.is_admin


def test_require_authenticated_rejects_anonymous() -> None:
    p = AnonymousPrincipal()
    with pytest.raises(AuthError):
        require_authenticated(p)


def test_require_authenticated_allows_authenticated() -> None:
    p = AuthenticatedPrincipal("test_user")
    require_authenticated(p)


# ---- Token authentication tests ----


def test_authenticate_token_valid(db, normal_user):
    session = create_session(db.conn, normal_user)
    token = session["token"]
    principal = authenticate_token(db.conn, token)
    assert principal.is_authenticated
    assert principal.user_id == normal_user


def test_authenticate_token_invalid(db):
    principal = authenticate_token(db.conn, "invalid_token")
    assert not principal.is_authenticated


def test_authenticate_token_empty(db):
    principal = authenticate_token(db.conn, "")
    assert not principal.is_authenticated


def test_authenticate_token_none(db):
    principal = authenticate_token(db.conn, None)
    assert not principal.is_authenticated


def test_authenticate_token_revoked(db, normal_user):
    session = create_session(db.conn, normal_user)
    token = session["token"]
    revoke_session_by_token(db.conn, token)
    principal = authenticate_token(db.conn, token)
    assert not principal.is_authenticated


def test_principal_from_rpc_auth_none(db):
    p = principal_from_rpc_auth(db.conn, None)
    assert isinstance(p, AnonymousPrincipal)


def test_principal_from_rpc_auth_missing_token(db):
    p = principal_from_rpc_auth(db.conn, {})
    assert isinstance(p, AnonymousPrincipal)


def test_principal_from_rpc_auth_valid(db, normal_user):
    session = create_session(db.conn, normal_user)
    p = principal_from_rpc_auth(db.conn, {"token": session["token"]})
    assert p.is_authenticated
    assert p.user_id == normal_user


# ---- Session tests ----


def test_create_session_returns_token_once(db, normal_user):
    session = create_session(db.conn, normal_user)
    assert "token" in session
    assert "expires_at" in session
    assert "user" in session
    assert session["user"]["user_id"] == normal_user


def test_session_stores_hash_not_token(db, normal_user):
    session = create_session(db.conn, normal_user)
    token = session["token"]
    rows = db.conn.execute("SELECT token_hash FROM mfdb_session").fetchall()
    token_hashes = {r[0] for r in rows}
    assert token not in token_hashes


def test_list_sessions(db, normal_user):
    create_session(db.conn, normal_user)
    sessions = list_sessions(db.conn)
    assert len(sessions) >= 1


def test_revoke_session(db, normal_user):
    session = create_session(db.conn, normal_user)
    srow = db.conn.execute(
        "SELECT session_id FROM mfdb_session WHERE user_id = ? AND revoked_at IS NULL",
        (normal_user,),
    ).fetchone()
    session_id = srow[0]
    revoke_session(db.conn, session_id)
    srow = db.conn.execute(
        "SELECT revoked_at FROM mfdb_session WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    assert srow[0] is not None


def test_session_last_used_updates(db, normal_user):
    session = create_session(db.conn, normal_user)
    token = session["token"]
    db.conn.execute("UPDATE mfdb_session SET last_used_at = NULL WHERE token_hash = ?",
                    (__import__("hashlib").sha256(token.encode()).hexdigest(),))
    db.conn.commit()
    authenticate_token(db.conn, token)
    row = db.conn.execute("SELECT last_used_at FROM mfdb_session WHERE token_hash = ?",
                          (__import__("hashlib").sha256(token.encode()).hexdigest(),)).fetchone()
    assert row[0] is not None


# ---- Auth attempt / throttling tests ----


def test_record_auth_attempt(db, normal_user):
    record_auth_attempt(db.conn, normal_user, False, "wrong_password")
    rows = db.conn.execute(
        "SELECT success, reason FROM mfdb_auth_attempt WHERE user_id = ?",
        (normal_user,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 0
    assert rows[0][1] == "wrong_password"


def test_is_throttled(db, normal_user):
    assert not is_throttled(db.conn, normal_user)
    for _ in range(6):
        record_auth_attempt(db.conn, normal_user, False, "test")
    assert is_throttled(db.conn, normal_user)


# ---- Permission denial tests ----


def test_require_access_raises_permission_denied(db):
    p = AnonymousPrincipal()
    with pytest.raises(PermissionDenied):
        require_access(db.conn, p, "test_type", "test_id", PERM_READ)


def test_require_access_admin_bypass(db, admin_user):
    p = AuthenticatedPrincipal(admin_user, is_admin=True)
    require_access(db.conn, p, "test_type", "test_id", PERM_READ)


# ---- ACL / mode tests ----


def test_create_default_acl(db, normal_user):
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user)
    row = db.conn.execute(
        "SELECT mode, owner_user_id FROM mfdb_object_acl WHERE object_type = ? AND object_id = ?",
        ("sample", "sample_1"),
    ).fetchone()
    assert row is not None
    assert row[0] == 0o700
    assert row[1] == normal_user


def test_owner_can_read_private_object(db, normal_user):
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user)
    p = AuthenticatedPrincipal(normal_user)
    assert can_access(db.conn, p, "sample", "sample_1", PERM_READ)


def test_anonymous_cannot_read_private_object(db, normal_user):
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user)
    p = AnonymousPrincipal()
    assert not can_access(db.conn, p, "sample", "sample_1", PERM_READ)


def test_anonymous_can_read_public_object(db, normal_user):
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user, mode=0o704)
    p = AnonymousPrincipal()
    assert can_access(db.conn, p, "sample", "sample_1", PERM_READ)


def test_anonymous_cannot_write_public_object(db, normal_user):
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user, mode=0o704)
    p = AnonymousPrincipal()
    assert not can_access(db.conn, p, "sample", "sample_1", PERM_WRITE)


def test_owner_without_write_bit_cannot_write(db, normal_user):
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user, mode=0o500)
    p = AuthenticatedPrincipal(normal_user)
    assert not can_access(db.conn, p, "sample", "sample_1", PERM_WRITE)


def test_owner_with_manage_bit_can_manage(db, normal_user):
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user, mode=0o700)
    p = AuthenticatedPrincipal(normal_user)
    assert can_access(db.conn, p, "sample", "sample_1", PERM_MANAGE)


def test_owner_without_manage_bit_cannot_manage(db, normal_user):
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user, mode=0o600)
    p = AuthenticatedPrincipal(normal_user)
    assert not can_access(db.conn, p, "sample", "sample_1", PERM_MANAGE)


def test_admin_can_read_everything(db, admin_user):
    p = AuthenticatedPrincipal(admin_user, is_admin=True)
    assert can_access(db.conn, p, "nonexistent", "nobody", PERM_READ)


def test_admin_can_write_everything(db, admin_user):
    p = AuthenticatedPrincipal(admin_user, is_admin=True)
    assert can_access(db.conn, p, "nonexistent", "nobody", PERM_WRITE)


def test_admin_can_manage_everything(db, admin_user):
    p = AuthenticatedPrincipal(admin_user, is_admin=True)
    assert can_access(db.conn, p, "nonexistent", "nobody", PERM_MANAGE)


def test_explicit_deny_blocks_read(db, normal_user):
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user, mode=0o704)
    grant_acl(
        db.conn, AuthenticatedPrincipal(normal_user, is_admin=True),
        "sample", "sample_1", "user", normal_user, PERM_READ, effect="deny",
    )
    p = AuthenticatedPrincipal(normal_user)
    assert not can_access(db.conn, p, "sample", "sample_1", PERM_READ)


def test_explicit_allow_grants_read(db, normal_user):
    other_user = _make_user(db, "other_user")
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user, mode=0o700)
    grant_acl(
        db.conn, AuthenticatedPrincipal(normal_user, is_admin=True),
        "sample", "sample_1", "user", other_user, PERM_READ, effect="allow",
    )
    p = AuthenticatedPrincipal(other_user)
    assert can_access(db.conn, p, "sample", "sample_1", PERM_READ)


def test_group_readable_object(db, normal_user):
    other_user = _make_user(db, "other_user")
    db.conn.execute(
        "INSERT OR IGNORE INTO mfdb_group_member (group_id, user_id, role) VALUES ('users', ?, 'member')",
        (other_user,),
    )
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user,
                                   owner_group_id="users", mode=0o740)
    p = AuthenticatedPrincipal(other_user)
    assert can_access(db.conn, p, "sample", "sample_1", PERM_READ)
    assert not can_access(db.conn, p, "sample", "sample_1", PERM_WRITE)


def test_group_writable_object(db, normal_user):
    other_user = _make_user(db, "other_user")
    db.conn.execute(
        "INSERT OR IGNORE INTO mfdb_group_member (group_id, user_id, role) VALUES ('users', ?, 'member')",
        (other_user,),
    )
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user,
                                   owner_group_id="users", mode=0o770)
    p = AuthenticatedPrincipal(other_user)
    assert can_access(db.conn, p, "sample", "sample_1", PERM_WRITE)


def test_list_excludes_unreadable(db, normal_user):
    other_user = _make_user(db, "other_user")
    create_default_acl_for_object(db.conn, "sample", "public_sample", normal_user, mode=0o704)
    create_default_acl_for_object(db.conn, "sample", "private_sample", normal_user, mode=0o700)

    rows = [
        {"id": "public_sample"},
        {"id": "private_sample"},
    ]
    p = AnonymousPrincipal()
    readable = filter_readable(db.conn, p, "sample", rows, id_key="id")
    readable_ids = {r["id"] for r in readable}
    assert "public_sample" in readable_ids
    assert "private_sample" not in readable_ids


def test_chmod_updates_mode(db, normal_user):
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user)
    p = AuthenticatedPrincipal(normal_user)
    chmod(db.conn, p, "sample", "sample_1", 0o704)
    row = db.conn.execute(
        "SELECT mode FROM mfdb_object_acl WHERE object_type = 'sample' AND object_id = 'sample_1'"
    ).fetchone()
    assert row[0] == 0o704


def test_chown_updates_owner(db, normal_user):
    other_user = _make_user(db, "other_user")
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user)
    p = AuthenticatedPrincipal(normal_user)
    chown(db.conn, p, "sample", "sample_1", other_user)
    row = db.conn.execute(
        "SELECT owner_user_id FROM mfdb_object_acl WHERE object_type = 'sample' AND object_id = 'sample_1'"
    ).fetchone()
    assert row[0] == other_user


def test_chgrp_updates_group(db, normal_user):
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user)
    p = AuthenticatedPrincipal(normal_user)
    chgrp(db.conn, p, "sample", "sample_1", "users")
    row = db.conn.execute(
        "SELECT owner_group_id FROM mfdb_object_acl WHERE object_type = 'sample' AND object_id = 'sample_1'"
    ).fetchone()
    assert row[0] == "users"


def test_inherit_acl_from_parent(db, normal_user):
    create_default_acl_for_object(db.conn, "experiment", "exp_1", normal_user, mode=0o750)
    inherit_acl_from_parent(db.conn, "artifact", "art_1", "experiment", "exp_1", owner_user_id=normal_user)
    row = db.conn.execute(
        "SELECT mode, inherits_from_type, inherits_from_id FROM mfdb_object_acl WHERE object_type = 'artifact' AND object_id = 'art_1'"
    ).fetchone()
    assert row is not None
    assert row[0] == 0o750
    assert row[1] == "experiment"
    assert row[2] == "exp_1"


def test_revoke_acl_soft_deletes(db, normal_user):
    other_user = _make_user(db, "other_user")
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user, mode=0o700)
    grant_acl(
        db.conn, AuthenticatedPrincipal(normal_user, is_admin=True),
        "sample", "sample_1", "user", other_user, PERM_READ, effect="allow",
    )
    row = db.conn.execute(
        "SELECT entry_id FROM mfdb_acl_entry WHERE object_type = 'sample' AND object_id = 'sample_1' AND deleted_at IS NULL"
    ).fetchone()
    entry_id = row[0]
    p = AuthenticatedPrincipal(normal_user)
    revoke_acl(db.conn, p, entry_id)
    row = db.conn.execute(
        "SELECT deleted_at FROM mfdb_acl_entry WHERE entry_id = ?",
        (entry_id,),
    ).fetchone()
    assert row[0] is not None


def test_login_handler_creates_session(db, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import login_handler
    result = login_handler(user_id=normal_user)
    assert result["ok"] is True
    assert "token" in result
    assert result["user"]["user_id"] == normal_user


def test_login_handler_admin_passwordless_flag_denied(db, patch_db):
    """Admin accounts can never log in without a password, even when the
    allow_passwordless_login flag is set."""
    from mfdb.admin.backend.auth_services import login_handler
    from mfdb.admin.backend.password_services import hash_password

    db.add_user(
        "passwordless_admin",
        display_name="Passwordless Admin",
        is_admin=1,
        password_hash=hash_password("correct_password"),
        allow_passwordless_login=1,
    )

    # No password -> denied despite the passwordless flag.
    with pytest.raises(AuthError):
        login_handler(user_id="passwordless_admin")

    # Correct password -> allowed.
    result = login_handler(user_id="passwordless_admin", password="correct_password")
    assert result["ok"] is True
    assert result["user"]["user_id"] == "passwordless_admin"
    assert result["user"]["is_admin"] is True


def test_login_handler_admin_empty_password_denied(db, patch_db):
    """An admin with a password hash cannot log in with an empty password."""
    from mfdb.admin.backend.auth_services import login_handler
    from mfdb.admin.backend.password_services import hash_password

    db.add_user(
        "some_admin",
        display_name="Some Admin",
        is_admin=1,
        password_hash=hash_password("Password123"),
    )
    with pytest.raises(AuthError):
        login_handler(user_id="some_admin", password="")


def test_login_handler_non_admin_passwordless_allowed(db, patch_db):
    """Non-admin users with the passwordless flag (e.g. guest) can still log in
    without a password."""
    from mfdb.admin.backend.auth_services import login_handler

    db.add_user("kiosk", display_name="Kiosk", is_admin=0, allow_passwordless_login=1)
    result = login_handler(user_id="kiosk")
    assert result["ok"] is True
    assert result["user"]["user_id"] == "kiosk"


def test_login_handler_wrong_password_fails(db, normal_user, patch_db):
    from mfdb.admin.backend.password_services import hash_password
    from mfdb.admin.backend.auth_services import login_handler
    db.conn.execute(
        "UPDATE flr_sample_users SET password_hash = ? WHERE user_id = ?",
        (hash_password("correct_password"), normal_user),
    )
    db.conn.commit()
    with pytest.raises(AuthError):
        login_handler(user_id=normal_user, password="wrong_password")


def test_me_handler_returns_user(db, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import me_handler
    session = create_session(db.conn, normal_user)
    result = me_handler(auth={"token": session["token"]})
    assert result["ok"] is True
    assert result["user"]["user_id"] == normal_user


def test_me_handler_rejects_anonymous(db, patch_db):
    from mfdb.admin.backend.auth_services import me_handler
    with pytest.raises(AuthError):
        me_handler(auth=None)


def test_groups_list(db, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import groups_list_handler
    session = create_session(db.conn, normal_user)
    result = groups_list_handler(auth={"token": session["token"]})
    assert result["ok"] is True
    groups = result["groups"]
    group_ids = {g["group_id"] for g in groups}
    assert "admins" in group_ids
    assert "users" in group_ids


def test_groups_create_admin_only(db, admin_user, patch_db):
    from mfdb.admin.backend.auth_services import groups_create_handler
    session = create_session(db.conn, admin_user)
    result = groups_create_handler(
        auth={"token": session["token"]},
        group={"group_id": "test_group", "display_name": "Test Group"},
    )
    assert result["ok"] is True


def test_groups_create_non_admin_fails(db, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import groups_create_handler
    session = create_session(db.conn, normal_user)
    with pytest.raises(PermissionDenied):
        groups_create_handler(
            auth={"token": session["token"]},
            group={"group_id": "test_group", "display_name": "Test Group"},
        )


def test_members_list(db, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import members_list_handler
    session = create_session(db.conn, normal_user)
    result = members_list_handler(auth={"token": session["token"]}, group_id="users")
    assert result["ok"] is True
    assert len(result["members"]) >= 1


def test_members_add(db, admin_user, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import members_add_handler
    session = create_session(db.conn, admin_user)
    result = members_add_handler(
        auth={"token": session["token"]},
        group_id="admins",
        user_id=normal_user,
    )
    assert result["ok"] is True
    rows = db.conn.execute(
        "SELECT role FROM mfdb_group_member WHERE group_id = 'admins' AND user_id = ? AND deleted_at IS NULL",
        (normal_user,),
    ).fetchall()
    assert len(rows) == 1


def test_members_remove(db, admin_user, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import members_add_handler, members_remove_handler
    session = create_session(db.conn, admin_user)

    members_add_handler(
        auth={"token": session["token"]},
        group_id="admins",
        user_id=normal_user,
    )

    result = members_remove_handler(
        auth={"token": session["token"]},
        group_id="admins",
        user_id=normal_user,
    )
    assert result["ok"] is True
    rows = db.conn.execute(
        "SELECT deleted_at FROM mfdb_group_member WHERE group_id = 'admins' AND user_id = ?",
        (normal_user,),
    ).fetchall()
    assert rows[0][0] is not None


def test_permissions_get(db, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import permissions_get_handler
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user)
    session = create_session(db.conn, normal_user)
    result = permissions_get_handler(
        auth={"token": session["token"]},
        object_type="sample",
        object_id="sample_1",
    )
    assert result["ok"] is True
    assert result["acl"] is not None
    assert result["acl"]["owner_user_id"] == normal_user


def test_logout_revokes_token(db, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import login_handler, logout_handler
    login_result = login_handler(user_id=normal_user)
    token = login_result["token"]
    logout_result = logout_handler(auth={"token": token})
    assert logout_result["ok"] is True
    principal = authenticate_token(db.conn, token)
    assert not principal.is_authenticated


def test_sessions_list(db, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import sessions_list_handler
    create_session(db.conn, normal_user)
    session = create_session(db.conn, normal_user)
    result = sessions_list_handler(auth={"token": session["token"]})
    assert result["ok"] is True
    assert len(result["sessions"]) >= 1


def test_sessions_revoke(db, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import sessions_revoke_handler
    session = create_session(db.conn, normal_user)
    srow = db.conn.execute(
        "SELECT session_id FROM mfdb_session WHERE user_id = ? AND revoked_at IS NULL",
        (normal_user,),
    ).fetchone()
    session_id = srow[0]
    session2 = create_session(db.conn, normal_user)
    result = sessions_revoke_handler(
        auth={"token": session2["token"]},
        session_id=session_id,
    )
    assert result["ok"] is True
    srow = db.conn.execute(
        "SELECT revoked_at FROM mfdb_session WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    assert srow[0] is not None


def test_permissions_chmod_via_rpc(db, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import permissions_chmod_handler
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user)
    session = create_session(db.conn, normal_user)
    result = permissions_chmod_handler(
        auth={"token": session["token"]},
        object_type="sample",
        object_id="sample_1",
        mode=0o704,
    )
    assert result["ok"] is True
    row = db.conn.execute(
        "SELECT mode FROM mfdb_object_acl WHERE object_type = 'sample' AND object_id = 'sample_1'"
    ).fetchone()
    assert row[0] == 0o704


def test_permissions_grant_via_rpc(db, normal_user, patch_db):
    from mfdb.admin.backend.auth_services import permissions_grant_handler
    other_user = _make_user(db, "other_user")
    create_default_acl_for_object(db.conn, "sample", "sample_1", normal_user)
    session = create_session(db.conn, normal_user)
    result = permissions_grant_handler(
        auth={"token": session["token"]},
        object_type="sample",
        object_id="sample_1",
        subject_type="user",
        subject_id=other_user,
        permissions=PERM_READ,
        effect="allow",
    )
    assert result["ok"] is True
    p = AuthenticatedPrincipal(other_user)
    assert can_access(db.conn, p, "sample", "sample_1", PERM_READ)
