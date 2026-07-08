"""Tests for the pluggable auth-provider framework + Local provider (PRD-49)."""

from __future__ import annotations

from pathlib import Path

import pytest
from mfdb.admin.backend.password_services import hash_password
from mfdb.repository import MFDatabase
from mfdb.security.auth import AuthError, authenticate_token, is_throttled
from mfdb.security.auth_providers import AuthIdentity, LocalAuthProvider
from mfdb.security.login import (
    login,
    resolve_or_provision_user,
    resolve_provider,
    sync_groups,
)


def _db(tmp_path: Path, name: str = "auth.db") -> MFDatabase:
    return MFDatabase(tmp_path / name)


def _add_user(db, user_id, *, password=None, is_admin=0, allow_passwordless=0, email=None):
    db.conn.execute(
        "INSERT INTO flr_sample_users "
        "(user_id, display_name, email, is_admin, allow_passwordless_login, password_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, user_id, email, is_admin, allow_passwordless,
         hash_password(password) if password else None),
    )
    db.conn.commit()
    return user_id


# ---- schema ----

def test_user_table_has_provider_columns(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        cols = {r[1]: r for r in db.conn.execute("PRAGMA table_info(flr_sample_users)")}
        assert "auth_provider" in cols
        assert "external_id" in cols
        # default provider is 'local'
        _add_user(db, "u1")
        row = db.conn.execute(
            "SELECT auth_provider, external_id FROM flr_sample_users WHERE user_id='u1'"
        ).fetchone()
        assert row["auth_provider"] == "local"
        assert row["external_id"] is None
        # reverse-lookup index exists
        idx = {r[1] for r in db.conn.execute("PRAGMA index_list(flr_sample_users)")}
        assert "idx_users_external" in idx


def test_bootstrap_admin_uses_random_salt(tmp_path: Path) -> None:
    # Two fresh DBs should not share the same admin hash (no fixed salt), yet
    # 'admin' still verifies against each.
    from mfdb.admin.backend.password_services import verify_password

    with _db(tmp_path, "a.db") as a, _db(tmp_path, "b.db") as b:
        ha = a.conn.execute(
            "SELECT password_hash FROM flr_sample_users WHERE user_id='user_default'"
        ).fetchone()[0]
        hb = b.conn.execute(
            "SELECT password_hash FROM flr_sample_users WHERE user_id='user_default'"
        ).fetchone()[0]
        assert ha != hb
        assert verify_password("admin", ha)
        assert verify_password("admin", hb)


# ---- LocalAuthProvider ----

def test_local_provider_password_rules(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        _add_user(db, "pw", password="secret", email="pw@example.org")
        _add_user(db, "free", allow_passwordless=1)
        _add_user(db, "nohash")
        _add_user(db, "boss", password="admin-pass", is_admin=1)
        prov = LocalAuthProvider(db.conn)

        # correct password → identity
        ident = prov.authenticate(user_id="pw", password="secret")
        assert isinstance(ident, AuthIdentity)
        assert ident.provider == "local" and ident.external_id == "pw"
        assert ident.email == "pw@example.org" and ident.is_admin is False

        assert prov.authenticate(user_id="pw", password="wrong") is None
        assert prov.authenticate(user_id="pw", password="") is None  # has hash, no pw
        assert prov.authenticate(user_id="missing", password="x") is None

        # passwordless
        assert prov.authenticate(user_id="free", password="") is not None

        # no hash: ok only when no password supplied
        assert prov.authenticate(user_id="nohash", password="") is not None
        assert prov.authenticate(user_id="nohash", password="anything") is None

        # admin always needs a password
        assert prov.authenticate(user_id="boss", password="") is None
        assert prov.authenticate(user_id="boss", password="admin-pass").is_admin is True


# ---- login() orchestrator (local) ----

def test_login_local_mints_resolvable_session(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        _add_user(db, "alice", password="pw")
        result = login(db.conn, user_id="alice", password="pw")
        assert result["ok"] and result["authenticated"]
        token = result["token"]
        principal = authenticate_token(db.conn, token)
        assert principal.is_authenticated
        assert principal.user_id == "alice"


def test_login_bad_password_raises(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        _add_user(db, "bob", password="pw")
        with pytest.raises(AuthError):
            login(db.conn, user_id="bob", password="nope")


def test_login_throttles_after_repeated_failures(tmp_path: Path) -> None:
    from mfdb.security.auth import MAX_FAILED_ATTEMPTS

    with _db(tmp_path) as db:
        _add_user(db, "carol", password="pw")
        for _ in range(MAX_FAILED_ATTEMPTS):
            with pytest.raises(AuthError):
                login(db.conn, user_id="carol", password="wrong")
        assert is_throttled(db.conn, "carol")
        with pytest.raises(AuthError, match="Too many"):
            login(db.conn, user_id="carol", password="pw")


def test_resolve_provider_unknown_raises(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        with pytest.raises(AuthError, match="Unknown auth provider"):
            resolve_provider("saml", conn=db.conn)


# ---- external-identity resolution / JIT provisioning ----

def _ext(external_id, *, email=None, is_admin=False, groups=()):
    return AuthIdentity(
        provider="ldap", external_id=external_id, email=email,
        display_name=f"Ext {external_id}", is_admin=is_admin, groups=tuple(groups),
    )


def test_jit_provision_creates_and_links_user(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        uid = resolve_or_provision_user(db.conn, _ext("jdoe", email="jdoe@lab.org"))
        assert uid == "jdoe"
        row = db.conn.execute(
            "SELECT auth_provider, external_id, email FROM flr_sample_users WHERE user_id='jdoe'"
        ).fetchone()
        assert row["auth_provider"] == "ldap"
        assert row["external_id"] == "jdoe"
        assert row["email"] == "jdoe@lab.org"
        # base 'users' group membership
        gm = db.conn.execute(
            "SELECT 1 FROM mfdb_group_member WHERE group_id='users' AND user_id='jdoe'"
        ).fetchone()
        assert gm is not None
        # second login matches the existing link (no duplicate)
        uid2 = resolve_or_provision_user(db.conn, _ext("jdoe", email="jdoe@lab.org"))
        assert uid2 == "jdoe"
        n = db.conn.execute(
            "SELECT COUNT(*) FROM flr_sample_users WHERE external_id='jdoe'"
        ).fetchone()[0]
        assert n == 1


def test_email_fallback_links_existing_user(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        _add_user(db, "local_kate", email="kate@lab.org")
        uid = resolve_or_provision_user(db.conn, _ext("kate", email="kate@lab.org"))
        assert uid == "local_kate"  # matched by email, not JIT-created
        row = db.conn.execute(
            "SELECT auth_provider, external_id FROM flr_sample_users WHERE user_id='local_kate'"
        ).fetchone()
        assert row["auth_provider"] == "ldap"
        assert row["external_id"] == "kate"


def test_match_only_unmatched_returns_none(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        assert resolve_or_provision_user(db.conn, _ext("ghost"), jit=False) is None


def test_jit_admin_joins_admins_group(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        resolve_or_provision_user(db.conn, _ext("root", is_admin=True))
        gm = db.conn.execute(
            "SELECT 1 FROM mfdb_group_member WHERE group_id='admins' AND user_id='root'"
        ).fetchone()
        assert gm is not None


def test_sync_groups_is_additive_and_idempotent(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        _add_user(db, "guser")
        sync_groups(db.conn, "guser", ("users", "public"))
        sync_groups(db.conn, "guser", ("users", "public"))  # idempotent
        rows = {
            r["group_id"]
            for r in db.conn.execute(
                "SELECT group_id FROM mfdb_group_member WHERE user_id='guser'"
            )
        }
        assert {"users", "public"} <= rows


def test_login_via_ldap_identity_end_to_end(tmp_path: Path, monkeypatch) -> None:
    # Drive login() through a stub 'ldap' provider to prove the external path
    # (authenticate → JIT provision → group sync → session) without a directory.
    import mfdb.security.login as login_mod

    class _StubLdap:
        name = "ldap"

        def __init__(self, cfg):
            self.cfg = cfg

        def authenticate(self, *, user_id, password=""):
            if password != "dirpass":
                return None
            return AuthIdentity(
                provider="ldap", external_id=user_id, email=f"{user_id}@lab.org",
                display_name=user_id.title(), is_admin=False, groups=("public",),
            )

    real = login_mod.resolve_provider

    def _resolve(name, *, conn, config=None):
        if (name or "").lower() == "ldap":
            return _StubLdap(config)
        return real(name, conn=conn, config=config)

    monkeypatch.setattr(login_mod, "resolve_provider", _resolve)

    with _db(tmp_path) as db:
        result = login(db.conn, provider="ldap", user_id="mallory", password="dirpass")
        assert result["ok"]
        principal = authenticate_token(db.conn, result["token"])
        assert principal.user_id == "mallory"
        # provisioned + mapped group
        assert db.conn.execute(
            "SELECT 1 FROM mfdb_group_member WHERE group_id='public' AND user_id='mallory'"
        ).fetchone() is not None
        with pytest.raises(AuthError):
            login(db.conn, provider="ldap", user_id="mallory", password="badpass")
