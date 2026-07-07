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

    conn.execute(
        "UPDATE mfdb_session SET last_used_at = ? WHERE token_hash = ?",
        (now, token_hash),
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
    rows = conn.execute(
        """SELECT gm.group_id
           FROM mfdb_group_member gm
           WHERE gm.user_id = ? AND gm.deleted_at IS NULL""",
        (user_id,),
    ).fetchall()
    return [r[0] for r in rows]


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
    rows = conn.execute(
        """SELECT subject_type, subject_id, effect, permissions
           FROM mfdb_acl_entry
           WHERE object_type = ? AND object_id = ? AND deleted_at IS NULL""",
        (object_type, object_id),
    ).fetchall()

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

        row = conn.execute(
            """SELECT owner_user_id, owner_group_id, mode, inherits_from_type, inherits_from_id
               FROM mfdb_object_acl
               WHERE object_type = ? AND object_id = ? AND deleted_at IS NULL""",
            (current_type, current_id),
        ).fetchone()

        if row:
            return {
                "owner_user_id": row[0],
                "owner_group_id": row[1],
                "mode": row[2],
                "inherits_from_type": row[3],
                "inherits_from_id": row[4],
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
    conn.execute(
        """INSERT OR IGNORE INTO mfdb_object_acl
           (object_type, object_id, owner_user_id, owner_group_id, mode)
           VALUES (?, ?, ?, ?, ?)""",
        (object_type, object_id, owner_user_id, owner_group_id, mode),
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
        conn.execute(
            """INSERT OR IGNORE INTO mfdb_object_acl
               (object_type, object_id, owner_user_id, owner_group_id, mode,
                inherits_from_type, inherits_from_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                object_type,
                object_id,
                owner_user_id or parent_acl["owner_user_id"],
                parent_acl["owner_group_id"],
                parent_acl["mode"],
                parent_type,
                parent_id,
            ),
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
    conn.execute(
        """INSERT INTO mfdb_acl_entry
           (object_type, object_id, subject_type, subject_id, effect, permissions, created_by_user_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (object_type, object_id, subject_type, subject_id, effect, permissions, user_id),
    )


def revoke_acl(
    conn: sqlite3.Connection,
    principal: Principal,
    entry_id: int,
) -> None:
    """Soft-delete an ACL entry.

    Requires manage (``x``) permission on the referenced object.
    """
    row = conn.execute(
        "SELECT object_type, object_id FROM mfdb_acl_entry WHERE entry_id = ?",
        (entry_id,),
    ).fetchone()
    if row:
        require_access(conn, principal, row[0], row[1], PERM_MANAGE)
        conn.execute(
            "UPDATE mfdb_acl_entry SET deleted_at = ? WHERE entry_id = ?",
            (_utc_now_iso(), entry_id),
        )


def chmod(
    conn: sqlite3.Connection,
    principal: Principal,
    object_type: str,
    object_id: str,
    mode: int,
) -> None:
    """Change the mode bits on an object. Requires manage (``x``)."""
    require_access(conn, principal, object_type, object_id, PERM_MANAGE)
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

    conn.execute(
        """INSERT INTO mfdb_session
           (session_id, user_id, token_hash, expires_at, client_host, client_name, client_metadata_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            user_id,
            token_hash,
            expires_at,
            client_host,
            client_name,
            _json_dumps(client_metadata) if client_metadata else None,
        ),
    )

    user_row = conn.execute(
        "SELECT user_id, display_name, is_admin FROM flr_sample_users WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    uid = user_row["user_id"] if isinstance(user_row, dict) else user_row[0]
    dname = user_row["display_name"] if isinstance(user_row, dict) else user_row[1]
    iadmin = user_row["is_admin"] if isinstance(user_row, dict) else user_row[2]
    return {
        "token": token,
        "expires_at": expires_at,
        "user": {
            "user_id": uid,
            "display_name": dname,
            "is_admin": bool(iadmin),
        },
    }


def revoke_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Revoke a session by ID."""
    conn.execute(
        "UPDATE mfdb_session SET revoked_at = ? WHERE session_id = ?",
        (_utc_now_iso(), session_id),
    )


def revoke_session_by_token(conn: sqlite3.Connection, token: str) -> None:
    """Revoke a session by its raw token."""
    token_hash = _hash_token(token)
    conn.execute(
        "UPDATE mfdb_session SET revoked_at = ? WHERE token_hash = ?",
        (_utc_now_iso(), token_hash),
    )


def list_sessions(
    conn: sqlite3.Connection,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """List non-revoked sessions, optionally filtered by user."""
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
    conn.execute(
        "INSERT INTO mfdb_auth_attempt (user_id, client_host, success, reason) VALUES (?, ?, ?, ?)",
        (user_id, client_host, 1 if success else 0, reason),
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
