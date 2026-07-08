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
from collections.abc import Callable
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


def _require_ldap3():
    """Import and return the optional ``ldap3`` dependency, or raise clearly."""
    try:
        import ldap3
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via monkeypatch
        if exc.name and exc.name.split(".")[0] != "ldap3":
            raise
        raise ImportError(
            "Optional dependency 'ldap3' is required for LDAP authentication. "
            "Install it (e.g. `pip install ldap3` or the mfdb '[ldap]' extra)."
        ) from exc
    return ldap3


def _entry_value(entry: Any, attr: str) -> str | None:
    """Return a single string value of *attr* on an ldap3 entry, or ``None``."""
    try:
        value = entry[attr].value
    except Exception:
        return None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value) if value is not None else None


def _entry_values(entry: Any, attr: str) -> list[str]:
    """Return all string values of *attr* on an ldap3 entry (empty if absent)."""
    try:
        values = entry[attr].values
    except Exception:
        return []
    return [str(v) for v in values]


class LdapAuthProvider:
    """Authenticate against an LDAP / Active Directory directory (PRD-49).

    Search+bind: bind as the configured service account, search for the login
    under ``base_dn`` with ``user_filter``, then re-bind as the located user DN
    with the supplied password to verify it. Directory group memberships
    (``memberOf``) are mapped to MFDB group ids via ``group_map`` and to admin
    status via ``admin_groups``.

    The ``ldap3`` dependency is optional and imported lazily. A
    ``connection_factory(user, password) -> Connection`` may be injected for
    offline testing (e.g. an ``ldap3`` ``MOCK_SYNC`` connection); when omitted a
    real connection is built from *config*.

    Config keys: ``host``, ``port``, ``use_ssl`` (default True), ``base_dn``,
    ``bind_dn``, ``bind_password``, ``user_filter`` (default ``"(uid={login})"``),
    ``uid_attr``/``mail_attr``/``name_attr``/``memberof_attr``, ``group_map``
    (ldap group → mfdb group id), ``admin_groups`` (ldap groups granting admin).
    """

    name = "ldap"

    def __init__(
        self,
        config: dict[str, Any],
        *,
        connection_factory: Callable[[str | None, str], Any] | None = None,
    ):
        self._cfg = dict(config or {})
        self._factory = connection_factory

    def _connect(self, user: str | None, password: str) -> Any:
        if self._factory is not None:
            return self._factory(user, password)
        ldap3 = _require_ldap3()
        cfg = self._cfg
        server = ldap3.Server(
            cfg.get("host"),
            port=cfg.get("port"),
            use_ssl=bool(cfg.get("use_ssl", True)),
            get_info=ldap3.NONE,
        )
        conn = ldap3.Connection(server, user=user, password=password)
        if cfg.get("start_tls"):
            conn.open()
            conn.start_tls()
        return conn

    def _map_groups(self, memberships: list[str]) -> tuple[tuple[str, ...], bool]:
        group_map = self._cfg.get("group_map") or {}
        admin_groups = set(self._cfg.get("admin_groups") or ())
        mfdb_groups: list[str] = []
        is_admin = False
        for member in memberships:
            mapped = group_map.get(member)
            if mapped and mapped not in mfdb_groups:
                mfdb_groups.append(mapped)
            if member in admin_groups:
                is_admin = True
        return tuple(mfdb_groups), is_admin

    def authenticate(self, *, user_id: str, password: str = "") -> AuthIdentity | None:
        """Search+bind *user_id* against the directory; return an identity or ``None``.

        Returns ``None`` for an unknown user or a wrong password. Raises
        :class:`~mfdb.security.auth.AuthError` only on misconfiguration (a failed
        service bind).
        """
        from mfdb.security.auth import AuthError

        if not password:
            # LDAP login requires a password (no anonymous/unauthenticated bind).
            return None

        cfg = self._cfg
        uid_attr = cfg.get("uid_attr", "uid")
        mail_attr = cfg.get("mail_attr", "mail")
        name_attr = cfg.get("name_attr", "cn")
        memberof_attr = cfg.get("memberof_attr", "memberOf")

        # 1. Service-account bind + search for the login.
        svc = self._connect(cfg.get("bind_dn"), cfg.get("bind_password") or "")
        if not svc.bind():
            raise AuthError("LDAP service bind failed (check bind_dn / bind_password)")
        login_filter = cfg.get("user_filter", "(uid={login})").format(
            login=_escape_filter(user_id)
        )
        svc.search(
            cfg["base_dn"],
            login_filter,
            attributes=[uid_attr, mail_attr, name_attr, memberof_attr],
        )
        if not svc.entries:
            return None
        entry = svc.entries[0]
        user_dn = entry.entry_dn

        # 2. Re-bind as the located user to verify the password.
        usr = self._connect(user_dn, password)
        if not usr.bind():
            return None

        # 3. Extract attributes + map groups.
        uid = _entry_value(entry, uid_attr) or user_id
        memberships = _entry_values(entry, memberof_attr)
        groups, is_admin = self._map_groups(memberships)
        return AuthIdentity(
            provider="ldap",
            external_id=uid,
            email=_entry_value(entry, mail_attr),
            display_name=_entry_value(entry, name_attr),
            is_admin=is_admin,
            groups=groups,
            raw={"dn": user_dn, "memberOf": memberships},
        )


def _escape_filter(value: str) -> str:
    """Escape LDAP filter special characters in a user-supplied login value."""
    try:
        from ldap3.utils.conv import escape_filter_chars

        return escape_filter_chars(value)
    except Exception:
        # Minimal fallback if ldap3's helper is unavailable.
        for ch, rep in (("\\", "\\5c"), ("*", "\\2a"), ("(", "\\28"), (")", "\\29"), ("\0", "\\00")):
            value = value.replace(ch, rep)
        return value
