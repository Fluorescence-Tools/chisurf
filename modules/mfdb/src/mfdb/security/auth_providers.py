"""Pluggable authentication providers for MFDB (PRD-49).

An :class:`AuthProvider` turns a credential (user id + password) into a validated
:class:`AuthIdentity` — the provider-native identity — or ``None`` for bad
credentials. :mod:`mfdb.security.login` orchestrates the rest: mapping the
identity onto an MFDB user (matching / JIT-provisioning), syncing groups, and
minting a session.

Providers hold their backend context — a live DB connection for
:class:`LocalAuthProvider`, a directory config for the LDAP provider — and keep
any optional third-party import lazy so ``mfdb`` still imports with only ``src``
on the path.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class AuthIdentity:
    """A provider-native identity resolved from a successful authentication.

    Parameters
    ----------
    provider : str
        Provider id (``"local"``, ``"ldap"``, ...).
    external_id : str
        Provider-native user identifier. For ``local`` this is the MFDB
        ``user_id``; for LDAP the directory ``uid`` (used as the MFDB user id on
        JIT provisioning and stored in ``flr_sample_users.external_id``).
    email : str or None
        Email address, used as the secondary match key.
    display_name : str or None
        Human-readable name for the (possibly JIT-provisioned) MFDB user.
    is_admin : bool
        Whether the provider considers this identity an administrator.
    groups : tuple of str
        MFDB group ids this identity should belong to (already mapped from the
        provider's native groups). Empty for ``local`` — local group membership
        is managed inside MFDB and is not re-synced on login.
    raw : dict
        Provider-native attributes, for diagnostics.
    """

    provider: str
    external_id: str
    email: str | None = None
    display_name: str | None = None
    is_admin: bool = False
    groups: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AuthProvider(Protocol):
    """Contract every authentication backend implements."""

    #: Provider id, e.g. ``"local"`` / ``"ldap"``.
    name: str

    def authenticate(self, *, user_id: str, password: str = "") -> AuthIdentity | None:
        """Return an :class:`AuthIdentity` on success, or ``None`` for bad credentials.

        Implementations raise only on misconfiguration (e.g. an unreachable
        directory), never for a wrong password.
        """
        ...


class LocalAuthProvider:
    """Authenticate against ``flr_sample_users.password_hash`` (PBKDF2-SHA256).

    Behaviour is identical to the pre-existing token-minting login path: admin
    accounts always require a password; ``allow_passwordless_login`` users log in
    without one; a user with no stored hash logs in only when no password is
    supplied.
    """

    name = "local"

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def authenticate(self, *, user_id: str, password: str = "") -> AuthIdentity | None:
        """Verify *password* for *user_id*; return an identity or ``None``."""
        from mfdb.admin.backend.password_services import verify_password

        row = self._conn.execute(
            "SELECT display_name, email, is_admin, password_hash, allow_passwordless_login "
            "FROM flr_sample_users WHERE user_id = ? AND deleted_at IS NULL",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        display_name = row["display_name"]
        email = row["email"]
        is_admin = row["is_admin"] == 1
        password_hash = row["password_hash"]
        allow_passwordless = row["allow_passwordless_login"] == 1

        # Admin accounts must always supply a password.
        if is_admin and not password:
            return None
        if allow_passwordless and not password:
            ok = True
        elif password_hash:
            ok = verify_password(password, password_hash)
        else:
            # No stored hash: accept only when no password is supplied.
            ok = not password
        if not ok:
            return None
        return AuthIdentity(
            provider="local",
            external_id=user_id,
            email=email,
            display_name=display_name,
            is_admin=is_admin,
        )
