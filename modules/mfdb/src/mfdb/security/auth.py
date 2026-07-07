"""MFDB authorization core: principals, token auth, permission evaluation, ACL mutation."""

from __future__ import annotations

import datetime
import hashlib
import logging
import secrets
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

SESSION_DURATION_HOURS = 12
MAX_FAILED_ATTEMPTS = 5
THROTTLE_WINDOW_MINUTES = 15


class AuthError(Exception):
    """Base auth exception — generic message to avoid leaking existence."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message)


class PermissionDenied(Exception):
    """Raised when access is denied — generic message, no existence leak."""

    def __init__(self, message: str = "Permission denied"):
        super().__init__(message)


class Principal:
    """Base principal — anonymous by default."""

    def __init__(self, user_id: str | None = None, is_admin: bool = False):
        self._user_id = user_id
        self._is_admin = is_admin

    @property
    def user_id(self) -> str | None:
        return self._user_id

    @property
    def is_admin(self) -> bool:
        return self._is_admin

    @property
    def is_authenticated(self) -> bool:
        return self._user_id is not None

    def __repr__(self) -> str:
        if self.is_authenticated:
            return f"Principal(user_id={self._user_id}, admin={self._is_admin})"
        return "Principal(anonymous)"


class AnonymousPrincipal(Principal):
    """Anonymous/unauthenticated principal."""

    def __init__(self):
        super().__init__(user_id=None, is_admin=False)


class AuthenticatedPrincipal(Principal):
    """Authenticated user principal."""

    def __init__(self, user_id: str, is_admin: bool = False):
        super().__init__(user_id=user_id, is_admin=is_admin)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _is_row_dict(row) -> bool:
    """Check if a row supports key-based access."""
    return hasattr(row, 'keys')


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


_DAO_SCHEMA_CACHE: dict[tuple[str, ...], Any] = {}


def _dao(conn: sqlite3.Connection):
    """Return a dictionary DAO over ``conn`` for schema-checked single-table CRUD.

    These security helpers take a bare connection (a transport-agnostic boundary
    that must not depend on ``MFDatabase``), so they build the DAO on demand.
    Bespoke reads (ACL-precedence joins, throttle aggregates) stay hand-written.

    The introspected schema is cached process-wide keyed by the table-name set
    (all MFDB databases share the dictionary-generated schema), so the per-call
    cost is one cheap ``sqlite_master`` read even when ``can_access`` loops over
    many rows — no repeated full ``PRAGMA`` introspection.
    """
    from mfdb.schema.dao import DictionaryDao

    names = tuple(
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    )
    schema = _DAO_SCHEMA_CACHE.get(names)
    if schema is None:
        schema = DictionaryDao.from_connection(conn)._schema
        _DAO_SCHEMA_CACHE[names] = schema
    return DictionaryDao(conn, schema)


def hash_token(token: str) -> str:
    """Return the storage hash for a session token.

    Public alias for the token-hashing helper so callers do not depend on the
    private ``_hash_token`` name.

    Parameters
    ----------
    token : str
        The raw session token.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest used to look the token up in storage.
    """
    return _hash_token(token)


def generate_session_token() -> str:
    """Generate a cryptographically secure session token."""
    return secrets.token_urlsafe(48)


def authenticate_token(conn: sqlite3.Connection, token: str) -> Principal:
    """Authenticate a session token, returning a Principal.

    Returns AnonymousPrincipal for invalid/expired/revoked tokens.
    Never raises — returns anonymous principal on failure.
    """
    if not token:
        return AnonymousPrincipal()

    token_hash = _hash_token(token)
    now = _utc_now_iso()

    row = conn.execute(
        """SELECT s.user_id, s.expires_at, s.revoked_at, u.is_admin
           FROM mfdb_session s
           JOIN flr_sample_users u ON u.user_id = s.user_id
           WHERE s.token_hash = ?""",
        (token_hash,),
    ).fetchone()

    if not row:
        return AnonymousPrincipal()

    user_id = row["user_id"] if _is_row_dict(row) else row[0]
    expires_at = row["expires_at"] if _is_row_dict(row) else row[1]
    revoked_at = row["revoked_at"] if _is_row_dict(row) else row[2]
    is_admin = row["is_admin"] if _is_row_dict(row) else row[3]

    if revoked_at is not None and revoked_at != "":
        return AnonymousPrincipal()
    if expires_at and expires_at < now:
        return AnonymousPrincipal()

    _dao(conn).update(
        "mfdb_session", token_hash, {"last_used_at": now}, pk_column="token_hash"
    )
    conn.commit()

    return AuthenticatedPrincipal(
        user_id=user_id,
        is_admin=bool(is_admin),
    )


def principal_from_rpc_auth(
    conn: sqlite3.Connection,
    auth: dict[str, Any] | None,
) -> Principal:
    """Convert an RPC ``auth`` payload into a Principal.

    When *auth* is ``None`` or missing a ``token`` key, returns an
    ``AnonymousPrincipal``.
    """
    if not auth or not isinstance(auth, dict):
        return AnonymousPrincipal()
    token = auth.get("token", "")
    if not token:
        return AnonymousPrincipal()
    return authenticate_token(conn, token)


def require_authenticated(principal: Principal) -> None:
    """Raise ``AuthError`` if *principal* is not authenticated."""
    if not principal.is_authenticated:
        raise AuthError("Authentication required")


# Permission bit constants
PERM_READ = 0o4
PERM_WRITE = 0o2
PERM_MANAGE = 0o1

# Mode bit shifts
OWNER_SHIFT = 6
GROUP_SHIFT = 3
OTHER_SHIFT = 0


def _perm_to_str(p: int) -> str:
    parts = []
    if p & PERM_READ:
        parts.append("r")
    if p & PERM_WRITE:
        parts.append("w")
    if p & PERM_MANAGE:
        parts.append("x")
    return "".join(parts) if parts else ""


def _user_group_ids(
    conn: sqlite3.Connection,
    user_id: str,
) -> list[str]:
    """Return all group IDs the user belongs to (via mfdb_group_member)."""
    return [
        row["group_id"]
        for row in _dao(conn).list("mfdb_group_member", filters={"user_id": user_id})
    ]


def _check_acls(
    conn: sqlite3.Connection,
    user_id: str,
    group_ids: list[str],
    object_type: str,
    object_id: str,
    permission: int,
) -> bool | None:
    """Check ACL entries for a principal on an object.

    Precedence (PRD):
    1. User entries evaluated before group entries (user-over-group).
    2. Within each category, deny wins over allow (deny-over-allow).

    Returns ``True`` if allowed, ``False`` if denied, ``None`` if no match.
    """
    rows = _dao(conn).list(
        "mfdb_acl_entry", filters={"object_type": object_type, "object_id": object_id}
    )

    user_deny = False
    user_allow = False
    group_deny = False
    group_allow = False

    for row in rows:
        subj_type = row["subject_type"] if _is_row_dict(row) else row[0]
        subj_id = row["subject_id"] if _is_row_dict(row) else row[1]
        effect = row["effect"] if _is_row_dict(row) else row[2]
        perms = row["permissions"] if _is_row_dict(row) else row[3]

        if subj_type == "user" and subj_id == user_id:
            if effect == "deny" and (perms & permission):
                user_deny = True
            elif effect == "allow" and (perms & permission):
                user_allow = True
        elif subj_type == "group" and subj_id in group_ids:
            if effect == "deny" and (perms & permission):
                group_deny = True
            elif effect == "allow" and (perms & permission):
                group_allow = True

    # User entries first, deny-over-allow within each
    if user_deny:
        return False
    if user_allow:
        return True

    if group_deny:
        return False
    if group_allow:
        return True

    return None


def _get_object_acl(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
) -> dict[str, Any] | None:
    """Fetch the ACL row for an object — follows inheritance chain."""
    seen: set[tuple[str, str]] = set()
    current_type = object_type
    current_id = object_id

    while current_type and current_id:
        key = (current_type, current_id)
        if key in seen:
            break
        seen.add(key)

        rows = _dao(conn).list(
            "mfdb_object_acl",
            filters={"object_type": current_type, "object_id": current_id},
            limit=1,
        )

        if rows:
            row = rows[0]
            return {
                "owner_user_id": row["owner_user_id"],
                "owner_group_id": row["owner_group_id"],
                "mode": row["mode"],
                "inherits_from_type": row["inherits_from_type"],
                "inherits_from_id": row["inherits_from_id"],
            }

        # Follow inheritance
        if current_type == "mfdb_object_acl":
            break
        current_type, current_id = None, None

    return None


def can_access(
    conn: sqlite3.Connection,
    principal: Principal,
    object_type: str,
    object_id: str,
    permission: int,
) -> bool:
    """Check if *principal* has *permission* on the object.

    Returns ``True`` if access is granted, ``False`` otherwise.
    Never raises — callers should check result and raise ``PermissionDenied``.
    """
    # Admin bypass
    if principal.is_admin:
        return True

    user_id = principal.user_id
    is_anonymous = not principal.is_authenticated

    acl = _get_object_acl(conn, object_type, object_id)
    if acl is None:
        # No ACL means the object doesn't exist or isn't protected
        # Match the generic not-found behavior
        return False

    mode = acl["mode"]
    owner_user_id = acl["owner_user_id"]
    owner_group_id = acl["owner_group_id"]

    # Anonymous can only read objects with other-read
    if is_anonymous:
        return bool(permission == PERM_READ and (mode & PERM_READ))

    group_ids = _user_group_ids(conn, user_id)

    if user_id == owner_user_id:
        # Check explicit DENY ACLs first (even for owner)
        acl_result = _check_acls(conn, user_id, group_ids, object_type, object_id, permission)
        if acl_result is False:
            return False
        if acl_result is True:
            return True
        # Owner mode bits
        owner_bits = (mode >> OWNER_SHIFT) & 0o7
        if owner_bits & permission:
            return True

    # Check ACLs for non-owner
    acl_result = _check_acls(conn, user_id, group_ids, object_type, object_id, permission)
    if acl_result is False:
        return False
    if acl_result is True:
        return True

    # Group membership
    if owner_group_id and owner_group_id in group_ids:
        group_bits = (mode >> GROUP_SHIFT) & 0o7
        if group_bits & permission:
            return True

    # Other permissions
    other_bits = mode & 0o7
    if other_bits & permission:
        return True

    return False


def require_access(
    conn: sqlite3.Connection,
    principal: Principal,
    object_type: str,
    object_id: str,
    permission: int,
) -> None:
    """Require *permission* on the object, raising ``PermissionDenied`` on failure.

    Uses generic messages to avoid leaking object existence.
    """
    if not can_access(conn, principal, object_type, object_id, permission):
        raise PermissionDenied()


def create_default_acl_for_object(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    owner_user_id: str,
    owner_group_id: str | None = None,
    mode: int = 0o700,
) -> None:
    """Create a default ACL row for a new object (private owner-only by default).

    Call this in the same transaction as the object write.
    """
    dao = _dao(conn)
    # INSERT OR IGNORE on UNIQUE(object_type, object_id): an ACL already covering
    # the object (even soft-deleted) is left untouched.
    if not dao.list(
        "mfdb_object_acl",
        filters={"object_type": object_type, "object_id": object_id},
        include_deleted=True,
        limit=1,
    ):
        dao.insert(
            "mfdb_object_acl",
            {
                "object_type": object_type,
                "object_id": object_id,
                "owner_user_id": owner_user_id,
                "owner_group_id": owner_group_id,
                "mode": mode,
            },
        )


def inherit_acl_from_parent(
    conn: sqlite3.Connection,
    object_type: str,
    object_id: str,
    parent_type: str,
    parent_id: str,
    owner_user_id: str | None = None,
) -> None:
    """Clone the ACL from a parent object for inheritance.

    Creates an ACL row on *object_type/object_id* that tracks the parent
    relationship via ``inherits_from_type`` / ``inherits_from_id``.
    """
    parent_acl = _get_object_acl(conn, parent_type, parent_id)
    if parent_acl:
        dao = _dao(conn)
        # INSERT OR IGNORE on UNIQUE(object_type, object_id).
        if not dao.list(
            "mfdb_object_acl",
            filters={"object_type": object_type, "object_id": object_id},
            include_deleted=True,
            limit=1,
        ):
            dao.insert(
                "mfdb_object_acl",
                {
                    "object_type": object_type,
                    "object_id": object_id,
                    "owner_user_id": owner_user_id or parent_acl["owner_user_id"],
                    "owner_group_id": parent_acl["owner_group_id"],
                    "mode": parent_acl["mode"],
                    "inherits_from_type": parent_type,
                    "inherits_from_id": parent_id,
                },
            )
    else:
        create_default_acl_for_object(
            conn,
            object_type,
            object_id,
            owner_user_id=owner_user_id or "user_default",
        )


def grant_acl(
    conn: sqlite3.Connection,
    principal: Principal,
    object_type: str,
    object_id: str,
    subject_type: str,
    subject_id: str,
    permissions: int,
    effect: str = "allow",
) -> None:
    """Grant or deny a permission entry on an object.

    Requires manage (``x``) permission on the object.
    """
    require_access(conn, principal, object_type, object_id, PERM_MANAGE)
    user_id = principal.user_id or "system"
    _dao(conn).insert(
        "mfdb_acl_entry",
        {
            "object_type": object_type,
            "object_id": object_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "effect": effect,
            "permissions": permissions,
            "created_by_user_id": user_id,
        },
    )


def revoke_acl(
    conn: sqlite3.Connection,
    principal: Principal,
    entry_id: int,
) -> None:
    """Soft-delete an ACL entry.

    Requires manage (``x``) permission on the referenced object.
    """
    dao = _dao(conn)
    row = dao.get("mfdb_acl_entry", entry_id, include_deleted=True)
    if row:
        require_access(conn, principal, row["object_type"], row["object_id"], PERM_MANAGE)
        dao.soft_delete("mfdb_acl_entry", entry_id, deleted_at=_utc_now_iso())


def chmod(
    conn: sqlite3.Connection,
    principal: Principal,
    object_type: str,
    object_id: str,
    mode: int,
) -> None:
    """Change the mode bits on an object. Requires manage (``x``)."""
    require_access(conn, principal, object_type, object_id, PERM_MANAGE)
    # raw: composite-key update — mfdb_object_acl is keyed by UNIQUE(object_type,
    # object_id) with no single PK, which dao.update cannot target.
    conn.execute(
        "UPDATE mfdb_object_acl SET mode = ?, updated_at = ? WHERE object_type = ? AND object_id = ?",
        (mode, _utc_now_iso(), object_type, object_id),
    )


def chown(
    conn: sqlite3.Connection,
    principal: Principal,
    object_type: str,
    object_id: str,
    owner_user_id: str,
) -> None:
    """Change object owner. Requires manage (``x``)."""
    require_access(conn, principal, object_type, object_id, PERM_MANAGE)
    # raw: composite-key update on UNIQUE(object_type, object_id) — see chmod.
    conn.execute(
        "UPDATE mfdb_object_acl SET owner_user_id = ?, updated_at = ? WHERE object_type = ? AND object_id = ?",
        (owner_user_id, _utc_now_iso(), object_type, object_id),
    )


def chgrp(
    conn: sqlite3.Connection,
    principal: Principal,
    object_type: str,
    object_id: str,
    owner_group_id: str,
) -> None:
    """Change object owning group. Requires manage (``x``)."""
    require_access(conn, principal, object_type, object_id, PERM_MANAGE)
    # raw: composite-key update on UNIQUE(object_type, object_id) — see chmod.
    conn.execute(
        "UPDATE mfdb_object_acl SET owner_group_id = ?, updated_at = ? WHERE object_type = ? AND object_id = ?",
        (owner_group_id, _utc_now_iso(), object_type, object_id),
    )


def filter_readable(
    conn: sqlite3.Connection,
    principal: Principal,
    object_type: str,
    rows: list[dict[str, Any]] | list[sqlite3.Row],
    id_key: str = "id",
) -> list[dict[str, Any]]:
    """Filter a list of rows to only those the principal can read."""
    result: list[dict[str, Any]] = []
    for row in rows:
        obj_id = row[id_key] if _is_row_dict(row) else row[id_key]
        if can_access(conn, principal, object_type, obj_id, PERM_READ):
            if isinstance(row, sqlite3.Row):
                result.append(dict(row))
            else:
                result.append(row)
    return result


# ---- Session helpers ----


def create_session(
    conn: sqlite3.Connection,
    user_id: str,
    client_host: str | None = None,
    client_name: str | None = None,
    client_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a session for *user_id*, returning ``{token, expires_at, user}``.

    The token is returned **only once** in the response. Only the hash is stored.
    """
    token = generate_session_token()
    token_hash = _hash_token(token)
    session_id = secrets.token_hex(16)
    expires_at = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(hours=SESSION_DURATION_HOURS)
    ).isoformat()

    dao = _dao(conn)
    dao.insert(
        "mfdb_session",
        {
            "session_id": session_id,
            "user_id": user_id,
            "token_hash": token_hash,
            "expires_at": expires_at,
            "client_host": client_host,
            "client_name": client_name,
            "client_metadata_json": _json_dumps(client_metadata) if client_metadata else None,
        },
    )

    user_row = dao.get("flr_sample_users", user_id, include_deleted=True) or {}
    return {
        "token": token,
        "expires_at": expires_at,
        "user": {
            "user_id": user_row.get("user_id"),
            "display_name": user_row.get("display_name"),
            "is_admin": bool(user_row.get("is_admin")),
        },
    }


def revoke_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Revoke a session by ID."""
    _dao(conn).update("mfdb_session", session_id, {"revoked_at": _utc_now_iso()})


def revoke_session_by_token(conn: sqlite3.Connection, token: str) -> None:
    """Revoke a session by its raw token."""
    token_hash = _hash_token(token)
    _dao(conn).update(
        "mfdb_session", token_hash, {"revoked_at": _utc_now_iso()}, pk_column="token_hash"
    )


def list_sessions(
    conn: sqlite3.Connection,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """List non-revoked sessions, optionally filtered by user."""
    # raw: deliberate column projection that excludes token_hash — the DAO's
    # list() has no projection and would leak the session token hash.
    if user_id:
        rows = conn.execute(
            """SELECT session_id, user_id, created_at, expires_at, last_used_at,
                      client_host, client_name, client_metadata_json
               FROM mfdb_session
               WHERE user_id = ? AND revoked_at IS NULL""",
            (user_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT session_id, user_id, created_at, expires_at, last_used_at,
                      client_host, client_name, client_metadata_json
               FROM mfdb_session
               WHERE revoked_at IS NULL""",
        ).fetchall()

    return [dict(r) for r in rows]


def record_auth_attempt(
    conn: sqlite3.Connection,
    user_id: str | None,
    success: bool,
    reason: str | None = None,
    client_host: str | None = None,
) -> None:
    """Record an authentication attempt."""
    _dao(conn).insert(
        "mfdb_auth_attempt",
        {
            "user_id": user_id,
            "client_host": client_host,
            "success": 1 if success else 0,
            "reason": reason,
        },
    )


def is_throttled(
    conn: sqlite3.Connection,
    user_id: str | None,
    client_host: str | None = None,
) -> bool:
    """Check if the user or host is throttled due to repeated failures."""
    from datetime import datetime, timedelta, timezone

    cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=THROTTLE_WINDOW_MINUTES)
    ).strftime("%Y-%m-%d %H:%M:%S")

    if user_id:
        row = conn.execute(
            """SELECT COUNT(*) FROM mfdb_auth_attempt
               WHERE user_id = ? AND success = 0 AND attempted_at > ?""",
            (user_id, cutoff),
        ).fetchone()
        count = row[0]
        if count >= MAX_FAILED_ATTEMPTS:
            return True

    if client_host:
        row = conn.execute(
            """SELECT COUNT(*) FROM mfdb_auth_attempt
               WHERE client_host = ? AND success = 0 AND attempted_at > ?""",
            (client_host, cutoff),
        ).fetchone()
        count = row[0]
        if count >= MAX_FAILED_ATTEMPTS:
            return True

    return False


def _json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, default=str, ensure_ascii=False)
