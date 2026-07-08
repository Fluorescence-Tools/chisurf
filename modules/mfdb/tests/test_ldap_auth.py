"""LDAP auth provider tests, driven fully offline via ldap3 MOCK_SYNC (PRD-49)."""

from __future__ import annotations

from pathlib import Path

import pytest

ldap3 = pytest.importorskip("ldap3")

from mfdb.repository import MFDatabase  # noqa: E402
from mfdb.security.auth import AuthError, authenticate_token  # noqa: E402
from mfdb.security.auth_providers import AuthIdentity, LdapAuthProvider  # noqa: E402

# A tiny fake directory seeded into every MOCK_SYNC connection.
_DIRECTORY = {
    "cn=svc,dc=lab": {"objectClass": ["person"], "userPassword": "svcpw"},
    "uid=jdoe,ou=people,dc=lab": {
        "objectClass": ["inetOrgPerson"],
        "userPassword": "secret",
        "uid": "jdoe",
        "mail": "jdoe@lab.org",
        "cn": "John Doe",
        "memberOf": ["cn=fret,ou=groups,dc=lab", "cn=admins,ou=groups,dc=lab"],
    },
    "uid=kate,ou=people,dc=lab": {
        "objectClass": ["inetOrgPerson"],
        "userPassword": "katepw",
        "uid": "kate",
        "mail": "kate@lab.org",
        "cn": "Kate Smith",
    },
}

_CFG = {
    "base_dn": "ou=people,dc=lab",
    "bind_dn": "cn=svc,dc=lab",
    "bind_password": "svcpw",
    "user_filter": "(uid={login})",
    "group_map": {"cn=fret,ou=groups,dc=lab": "users"},
    "admin_groups": ["cn=admins,ou=groups,dc=lab"],
}


def _factory(entries=None):
    server = ldap3.Server("fake")
    directory = entries if entries is not None else _DIRECTORY

    def make(user, password):
        conn = ldap3.Connection(
            server, user=user, password=password, client_strategy=ldap3.MOCK_SYNC
        )
        for dn, attrs in directory.items():
            conn.strategy.add_entry(dn, attrs)
        return conn

    return make


def _provider(cfg=None, entries=None):
    return LdapAuthProvider(cfg or _CFG, connection_factory=_factory(entries))


# ---- provider.authenticate ----

def test_ldap_success_maps_attrs_and_groups() -> None:
    ident = _provider().authenticate(user_id="jdoe", password="secret")
    assert isinstance(ident, AuthIdentity)
    assert ident.provider == "ldap"
    assert ident.external_id == "jdoe"
    assert ident.email == "jdoe@lab.org"
    assert ident.display_name == "John Doe"
    assert ident.is_admin is True  # member of admin_groups
    assert ident.groups == ("users",)  # cn=fret mapped; cn=admins not in group_map


def test_ldap_wrong_password_returns_none() -> None:
    assert _provider().authenticate(user_id="jdoe", password="nope") is None


def test_ldap_unknown_user_returns_none() -> None:
    assert _provider().authenticate(user_id="ghost", password="x") is None


def test_ldap_empty_password_returns_none() -> None:
    assert _provider().authenticate(user_id="jdoe", password="") is None


def test_ldap_non_admin_no_group_mapping() -> None:
    ident = _provider().authenticate(user_id="kate", password="katepw")
    assert ident is not None
    assert ident.is_admin is False
    assert ident.groups == ()


def test_ldap_service_bind_failure_raises() -> None:
    cfg = dict(_CFG, bind_password="wrong")
    with pytest.raises(AuthError, match="service bind"):
        _provider(cfg).authenticate(user_id="jdoe", password="secret")


def test_ldap_filter_injection_is_escaped() -> None:
    # A wildcard login must not match every entry — the filter value is escaped.
    assert _provider().authenticate(user_id="*", password="secret") is None


def test_ldap_missing_dependency_raises(monkeypatch) -> None:
    import mfdb.security.auth_providers as ap

    def _boom():
        raise ImportError("no ldap3")

    monkeypatch.setattr(ap, "_require_ldap3", _boom)
    # No injected factory → _connect falls back to the real (now-missing) dep.
    with pytest.raises(ImportError):
        LdapAuthProvider(_CFG).authenticate(user_id="jdoe", password="secret")


# ---- login() end-to-end via the LDAP provider ----

def test_login_ldap_jit_provisions_and_maps_groups(tmp_path: Path, monkeypatch) -> None:
    import mfdb.security.login as login_mod

    def _resolve(name, *, conn, config=None):
        assert (name or "").lower() == "ldap"
        return _provider()

    monkeypatch.setattr(login_mod, "resolve_provider", _resolve)

    with MFDatabase(tmp_path / "ldap.db") as db:
        result = login_mod.login(db.conn, provider="ldap", user_id="jdoe", password="secret")
        assert result["ok"] and result["authenticated"]
        principal = authenticate_token(db.conn, result["token"])
        assert principal.user_id == "jdoe"
        assert principal.is_admin is True

        row = db.conn.execute(
            "SELECT auth_provider, external_id, email FROM flr_sample_users WHERE user_id='jdoe'"
        ).fetchone()
        assert row["auth_provider"] == "ldap"
        assert row["external_id"] == "jdoe"
        assert row["email"] == "jdoe@lab.org"

        groups = {
            r["group_id"]
            for r in db.conn.execute(
                "SELECT group_id FROM mfdb_group_member WHERE user_id='jdoe'"
            )
        }
        assert "users" in groups   # mapped from cn=fret
        assert "admins" in groups  # admin identity joins admins

        with pytest.raises(AuthError):
            login_mod.login(db.conn, provider="ldap", user_id="jdoe", password="bad")


# ---- config wiring ----

def test_configured_auth_config_env_and_resolver(monkeypatch) -> None:
    import mfdb.config as config

    monkeypatch.setattr(config, "_AUTH_CONFIG_RESOLVER", None)
    monkeypatch.setenv("MFDB_AUTH_PROVIDER", "ldap")
    monkeypatch.setenv("MFDB_LDAP_HOST", "ldap.lab.org")
    monkeypatch.setenv("MFDB_LDAP_BASE_DN", "ou=people,dc=lab")
    cfg = config.configured_auth_config()
    assert cfg["auth_provider"] == "ldap"
    assert cfg["ldap"]["host"] == "ldap.lab.org"
    assert cfg["ldap"]["base_dn"] == "ou=people,dc=lab"

    # A host-injected resolver takes precedence over the env.
    config.set_auth_config_resolver(lambda: {"auth_provider": "local"})
    try:
        assert config.configured_auth_config() == {"auth_provider": "local"}
    finally:
        config.set_auth_config_resolver(None)
