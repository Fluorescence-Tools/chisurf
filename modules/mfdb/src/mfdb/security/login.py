"""Authentication orchestration: provider → MFDB user → session (PRD-59).

``login()`` is the single entry point behind the ``mfdb.security.auth.login`` RPC.
It selects an :class:`~mfdb.security.auth_providers.AuthProvider`, authenticates,
resolves the provider identity onto an MFDB ``flr_sample_users`` row (matching by
``(auth_provider, external_id)`` then ``email``, else JIT-provisioning for
external providers), syncs mapped groups into ``mfdb_group_member``, and mints a
session via the existing :func:`~mfdb.security.auth.create_session`.

The ``local`` provider is always available (bootstrap admin, offline/standalone);
a configured default provider (e.g. ``ldap``) handles the rest.
"""

from __future__ import annotations

import sqlite3
import uuid as _uuid
from typing import Any

from mfdb.security.auth import (
    AuthError,
    _dao,
    create_session,
    is_throttled,
    record_auth_attempt,
)
from mfdb.security.auth_providers import AuthIdentity, AuthProvider, LocalAuthProvider

#: Default active branch every user is attached to (the "main" branch).
_MAIN_BRANCH_UUID = "00000000-0000-0000-0000-000000000000"


def resolve_provider(
    name: str | None,
    *,
    conn: sqlite3.Connection,
    config: dict[str, Any] | None = None,
) -> AuthProvider:
    """Return the :class:`AuthProvider` for *name* (or the configured default).

    ``"local"`` is always constructible; ``"ldap"`` is loaded lazily (its
    optional ``ldap3`` dependency is only imported when selected).
    """
    prov = (name or (config or {}).get("auth_provider") or "local").lower()
    if prov == "local":
        return LocalAuthProvider(conn)
    if prov == "ldap":
        from mfdb.security.auth_providers import LdapAuthProvider  # Phase 2

        ldap_cfg = (config or {}).get("ldap") if config else None
        return LdapAuthProvider(ldap_cfg or {})
    raise AuthError(f"Unknown auth provider: {prov!r}")


def _jit_enabled(config: dict[str, Any] | None) -> bool:
    """Whether external providers may JIT-provision an MFDB user (default True)."""
    if not config:
        return True
    ldap_cfg = config.get("ldap") or {}
    return bool(ldap_cfg.get("jit_provision", True))


def resolve_or_provision_user(
    conn: sqlite3.Connection,
    identity: AuthIdentity,
    *,
    jit: bool = True,
) -> str | None:
    """Map *identity* to an MFDB ``user_id``.

    ``local`` identities are their own MFDB user. External identities are matched
    by ``(auth_provider, external_id)`` then ``email`` (linking the row), else
    JIT-provisioned when *jit* is set (``None`` otherwise — match-only).
    """
    if identity.provider == "local":
        return identity.external_id

    dao = _dao(conn)
    rows = dao.list(
        "flr_sample_users",
        filters={"auth_provider": identity.provider, "external_id": identity.external_id},
        limit=1,
    )
    if rows:
        return rows[0]["user_id"]

    if identity.email:
        rows = dao.list("flr_sample_users", filters={"email": identity.email}, limit=1)
        if rows:
            user_id = rows[0]["user_id"]
            dao.update(
                "flr_sample_users",
                user_id,
                {"auth_provider": identity.provider, "external_id": identity.external_id},
            )
            return user_id

    if not jit:
        return None
    return _provision_user(conn, identity)


def _provision_user(conn: sqlite3.Connection, identity: AuthIdentity) -> str:
    """Create a new MFDB user row from an external *identity* and base groups."""
    dao = _dao(conn)
    user_id = identity.external_id
    dao.insert(
        "flr_sample_users",
        {
            "user_id": user_id,
            "user_uuid": str(_uuid.uuid4()),
            "display_name": identity.display_name or user_id,
            "email": identity.email,
            "is_admin": 1 if identity.is_admin else 0,
            "allow_passwordless_login": 0,
            "auth_provider": identity.provider,
            "external_id": identity.external_id,
            "active_branch_uuid": _MAIN_BRANCH_UUID,
        },
    )
    # Base membership mirrors bootstrap_auth_groups: everyone in "users",
    # admins also in "admins".
    _ensure_group_member(conn, "users", user_id)
    if identity.is_admin:
        _ensure_group_member(conn, "admins", user_id)
    return user_id


def _ensure_group_member(
    conn: sqlite3.Connection, group_id: str, user_id: str, role: str = "member"
) -> None:
    """Idempotently add *user_id* to *group_id* (UNIQUE(group_id, user_id))."""
    dao = _dao(conn)
    if not dao.list(
        "mfdb_group_member",
        filters={"group_id": group_id, "user_id": user_id},
        include_deleted=True,
        limit=1,
    ):
        dao.insert(
            "mfdb_group_member",
            {"group_id": group_id, "user_id": user_id, "role": role},
        )


def sync_groups(conn: sqlite3.Connection, user_id: str, group_ids: tuple[str, ...]) -> None:
    """Ensure *user_id* belongs to each mapped MFDB group id (additive)."""
    for group_id in group_ids:
        if group_id:
            _ensure_group_member(conn, group_id, user_id)


def login(
    conn: sqlite3.Connection,
    *,
    provider: str | None = None,
    user_id: str,
    password: str = "",
    client_metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    jit: bool | None = None,
) -> dict[str, Any]:
    """Authenticate and mint a session; returns ``{ok, authenticated, token, ...}``.

    Raises :class:`~mfdb.security.auth.AuthError` on throttle or bad credentials.
    Commits only on success (matching the prior login path); failed attempts are
    recorded within the caller's transaction.
    """
    if is_throttled(conn, user_id):
        raise AuthError("Too many failed login attempts. Try again later.")

    prov = resolve_provider(provider, conn=conn, config=config)
    identity = prov.authenticate(user_id=user_id, password=password)
    if identity is None:
        record_auth_attempt(conn, user_id, False, reason="invalid_credentials")
        raise AuthError("Invalid credentials")

    jit_enabled = _jit_enabled(config) if jit is None else jit
    mfdb_user_id = resolve_or_provision_user(conn, identity, jit=jit_enabled)
    if mfdb_user_id is None:
        record_auth_attempt(conn, user_id, False, reason="unmatched_external_user")
        raise AuthError("Invalid credentials")

    sync_groups(conn, mfdb_user_id, identity.groups)

    client_host = client_name = None
    if client_metadata:
        client_host = client_metadata.get("host")
        client_name = client_metadata.get("name")
    session = create_session(
        conn,
        user_id=mfdb_user_id,
        client_host=client_host,
        client_name=client_name,
        client_metadata=client_metadata,
    )
    record_auth_attempt(conn, mfdb_user_id, True)
    conn.commit()
    return {"ok": True, "authenticated": True, **session}
