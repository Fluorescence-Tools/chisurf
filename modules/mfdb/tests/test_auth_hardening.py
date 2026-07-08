"""Auth hardening: provider registry, throttle persistence, reconciliation (PRD-59 Phase 4)."""

from __future__ import annotations

from pathlib import Path

import mfdb.config as config
import mfdb.security.auth_providers as ap
import mfdb.security.login as login_mod
import pytest
from mfdb.repository import MFDatabase
from mfdb.security.auth import MAX_FAILED_ATTEMPTS, AuthError, authenticate_token, is_throttled
from mfdb.security.auth_providers import (
    AuthIdentity,
    LdapAuthProvider,
    ProviderContext,
    available_providers,
    build_provider,
    register_provider,
)


def _db(tmp_path: Path, name: str = "hard.db") -> MFDatabase:
    return MFDatabase(tmp_path / name)


# ---- extensible provider registry ----

def test_builtin_providers_registered() -> None:
    assert "local" in available_providers()
    assert "ldap" in available_providers()


def test_register_custom_provider_and_build(tmp_path: Path) -> None:
    class _Custom:
        name = "custom"

        def __init__(self, ctx):
            self.ctx = ctx

        def authenticate(self, *, user_id, password=""):
            if password != "ok":
                return None
            return AuthIdentity(provider="custom", external_id=user_id, email=f"{user_id}@x")

    # A new backend is added by registration alone — no core edit.
    register_provider("custom", _Custom)
    try:
        assert "custom" in available_providers()
        with _db(tmp_path) as db:
            prov = build_provider("custom", ProviderContext(conn=db.conn, config=None))
            assert prov.authenticate(user_id="z", password="ok").provider == "custom"
            assert prov.authenticate(user_id="z", password="bad") is None

            # login() dispatches through the registry to the custom provider (JIT).
            result = login_mod.login(db.conn, provider="custom", user_id="newbie", password="ok")
            assert result["ok"]
            assert authenticate_token(db.conn, result["token"]).user_id == "newbie"
    finally:
        ap._PROVIDER_FACTORIES.pop("custom", None)


def test_build_unknown_provider_raises(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        with pytest.raises(AuthError, match="Unknown auth provider"):
            build_provider("nope", ProviderContext(conn=db.conn))


# ---- throttle persistence across RPC connections (the security fix) ----

def test_throttle_persists_across_rpc_logins(tmp_path: Path, monkeypatch) -> None:
    import contextlib

    from mfdb.admin.backend import auth_services

    db_path = tmp_path / "throttle.db"
    MFDatabase(db_path).close()
    monkeypatch.setattr(config, "_AUTH_CONFIG_RESOLVER", None)
    monkeypatch.delenv("MFDB_AUTH_PROVIDER", raising=False)

    # Each RPC login opens its OWN fresh connection on the same DB file.
    @contextlib.contextmanager
    def _fresh_db():
        db = MFDatabase(db_path)
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(auth_services, "_get_db", _fresh_db)
    login_handler = auth_services.login_handler

    # Each call opens its OWN MFDatabase (like a real RPC request).
    for _ in range(MAX_FAILED_ATTEMPTS):
        with pytest.raises(AuthError):
            login_handler("user_default", "wrongpw")

    # A fresh connection must now see the accumulated failures → throttled.
    db = MFDatabase(db_path)
    try:
        n = db.conn.execute("SELECT COUNT(*) FROM mfdb_auth_attempt WHERE success=0").fetchone()[0]
        assert n >= MAX_FAILED_ATTEMPTS
        assert is_throttled(db.conn, "user_default")
    finally:
        db.close()


# ---- directory-authoritative attribute + group reconciliation ----

def _stub_resolver(state):
    class _Stub:
        name = "ldap"

        def authenticate(self, *, user_id, password=""):
            return AuthIdentity(
                provider="ldap",
                external_id=user_id,
                email=state["email"],
                display_name=state["name"],
                is_admin=state["admin"],
                groups=state["groups"],
                managed_groups=("g_a", "g_b"),
            )

    def _resolve(name, *, conn, config=None):
        return _Stub()

    return _resolve


def _groups(db, user_id):
    return {
        r["group_id"]
        for r in db.conn.execute(
            "SELECT group_id FROM mfdb_group_member WHERE user_id=? AND deleted_at IS NULL",
            (user_id,),
        )
    }


def test_login_reconciles_attributes_and_groups(tmp_path: Path, monkeypatch) -> None:
    state = {"admin": True, "groups": ("g_a",), "email": "u@lab.org", "name": "User One"}
    monkeypatch.setattr(login_mod, "resolve_provider", _stub_resolver(state))

    with _db(tmp_path) as db:
        for g in ("g_a", "g_b"):
            db.conn.execute(
                "INSERT OR IGNORE INTO mfdb_group (group_id, display_name) VALUES (?, ?)", (g, g)
            )
        db.conn.commit()

        # First login: admin, in g_a.
        login_mod.login(db.conn, provider="ldap", user_id="u", password="pw")
        row = db.conn.execute(
            "SELECT is_admin, email, display_name FROM flr_sample_users WHERE user_id='u'"
        ).fetchone()
        assert row["is_admin"] == 1
        assert row["email"] == "u@lab.org"
        assert row["display_name"] == "User One"
        g1 = _groups(db, "u")
        assert {"g_a", "admins", "users"} <= g1
        assert "g_b" not in g1

        # Directory changes: no longer admin, now in g_b instead of g_a.
        state.update(admin=False, groups=("g_b",), email="u2@lab.org", name="User Renamed")
        login_mod.login(db.conn, provider="ldap", user_id="u", password="pw")
        row = db.conn.execute(
            "SELECT is_admin, email, display_name FROM flr_sample_users WHERE user_id='u'"
        ).fetchone()
        assert row["is_admin"] == 0
        assert row["email"] == "u2@lab.org"
        assert row["display_name"] == "User Renamed"
        g2 = _groups(db, "u")
        assert "g_b" in g2
        assert "g_a" not in g2       # managed group removed when absent
        assert "admins" not in g2    # de-escalated
        assert "users" in g2         # base membership retained


def test_reconcile_does_not_touch_unmanaged_groups(tmp_path: Path, monkeypatch) -> None:
    state = {"admin": False, "groups": ("g_a",), "email": None, "name": None}
    monkeypatch.setattr(login_mod, "resolve_provider", _stub_resolver(state))
    with _db(tmp_path) as db:
        for g in ("g_a", "g_b", "local_only"):
            db.conn.execute(
                "INSERT OR IGNORE INTO mfdb_group (group_id, display_name) VALUES (?, ?)", (g, g)
            )
        db.conn.commit()
        login_mod.login(db.conn, provider="ldap", user_id="u", password="pw")
        # A locally-added group outside managed_groups must survive reconciliation.
        db.conn.execute(
            "INSERT INTO mfdb_group_member (group_id, user_id, role) VALUES ('local_only','u','member')"
        )
        db.conn.commit()
        login_mod.login(db.conn, provider="ldap", user_id="u", password="pw")
        assert "local_only" in _groups(db, "u")


# ---- LDAP robustness ----

def test_ldap_missing_base_dn_raises() -> None:
    prov = LdapAuthProvider({"bind_dn": "cn=svc"}, connection_factory=lambda u, p: None)
    with pytest.raises(AuthError, match="base_dn"):
        prov.authenticate(user_id="x", password="pw")


def test_ldap_directory_unavailable_raises() -> None:
    class _Raising:
        def bind(self):
            raise RuntimeError("connection refused")

        def unbind(self):
            pass

    prov = LdapAuthProvider(
        {"base_dn": "dc=lab", "bind_dn": "cn=svc"},
        connection_factory=lambda u, p: _Raising(),
    )
    with pytest.raises(AuthError, match="unavailable"):
        prov.authenticate(user_id="u", password="pw")


def test_ldap_managed_groups_from_config() -> None:
    prov = LdapAuthProvider({"base_dn": "dc=lab", "group_map": {"cn=a": "g_a", "cn=b": "g_b"}})
    assert set(prov._managed_groups()) == {"g_a", "g_b"}
