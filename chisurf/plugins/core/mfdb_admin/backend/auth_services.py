"""JSON-RPC handlers for MFDB authentication, groups, and permissions."""

from __future__ import annotations

from typing import Any

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.auth import (
    PERM_MANAGE,
    PERM_READ,
    PERM_WRITE,
    AuthError,
    PermissionDenied,
    authenticate_token,
    can_access,
    chgrp,
    chmod,
    chown,
    create_default_acl_for_object,
    create_session,
    grant_acl,
    is_throttled,
    list_sessions,
    principal_from_rpc_auth,
    record_auth_attempt,
    require_access,
    require_authenticated,
    revoke_acl,
    revoke_session,
)
from chisurf.core.mfdb.database_resolver import resolve_database_path
from chisurf.plugins.core.mfdb_admin.backend.password_services import (
    evaluate_password,
    hash_password,
    verify_password,
)


def _get_db():
    return MFDatabase(resolve_database_path())


def _get_conn(db):
    return db.conn


def register_services(dispatcher_or_context: Any) -> None:
    """Register auth/group/permission RPC handlers."""
    dispatcher = getattr(dispatcher_or_context, "dispatcher", dispatcher_or_context)

    # Auth
    dispatcher.register("mfdb.auth.login", lambda params: login_handler(**params))
    dispatcher.register("mfdb.auth.logout", lambda params: logout_handler(**params))
    dispatcher.register("mfdb.auth.me", lambda params: me_handler(**params))
    dispatcher.register("mfdb.auth.change_password", lambda params: change_password_handler(**params))

    # Sessions
    dispatcher.register("mfdb.auth.sessions.list", lambda params: sessions_list_handler(**params))
    dispatcher.register("mfdb.auth.sessions.revoke", lambda params: sessions_revoke_handler(**params))

    # Groups
    dispatcher.register("mfdb.groups.list", lambda params: groups_list_handler(**params))
    dispatcher.register("mfdb.groups.get", lambda params: groups_get_handler(**params))
    dispatcher.register("mfdb.groups.create", lambda params: groups_create_handler(**params))
    dispatcher.register("mfdb.groups.update", lambda params: groups_update_handler(**params))
    dispatcher.register("mfdb.groups.delete", lambda params: groups_delete_handler(**params))

    # Group members
    dispatcher.register("mfdb.groups.members.list", lambda params: members_list_handler(**params))
    dispatcher.register("mfdb.groups.members.add", lambda params: members_add_handler(**params))
    dispatcher.register("mfdb.groups.members.remove", lambda params: members_remove_handler(**params))

    # Permissions
    dispatcher.register("mfdb.permissions.get", lambda params: permissions_get_handler(**params))
    dispatcher.register("mfdb.permissions.chmod", lambda params: permissions_chmod_handler(**params))
    dispatcher.register("mfdb.permissions.chown", lambda params: permissions_chown_handler(**params))
    dispatcher.register("mfdb.permissions.chgrp", lambda params: permissions_chgrp_handler(**params))
    dispatcher.register("mfdb.permissions.grant", lambda params: permissions_grant_handler(**params))
    dispatcher.register("mfdb.permissions.revoke", lambda params: permissions_revoke_handler(**params))


# ---- Auth handlers ----


def login_handler(
    user_id: str,
    password: str = "",
    client_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate a user and return a session token.

    Parameters
    ----------
    user_id : str
        User identifier.
    password : str
        Password for the user.
    client_metadata : dict, optional
        Optional client info (host, name, etc.).

    Returns
    -------
    dict
        ``{token, expires_at, user}`` on success, or raises ``AuthError``.
    """
    with _get_db() as db:
        conn = _get_conn(db)

        if is_throttled(conn, user_id):
            raise AuthError("Too many failed login attempts. Try again later.")

        row = conn.execute(
            "SELECT display_name, is_admin, password_hash, allow_passwordless_login FROM flr_sample_users WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if not row:
            record_auth_attempt(conn, user_id, False, reason="user_not_found")
            raise AuthError("Invalid credentials")

        display_name, is_admin, password_hash, allow_passwordless_login = row

        if allow_passwordless_login == 1 and not password:
            pass
        elif password_hash:
            if not verify_password(password, password_hash):
                record_auth_attempt(conn, user_id, False, reason="wrong_password")
                raise AuthError("Invalid credentials")
        else:
            if not password:
                # Passwordless user or admin — allow login so admin can reach
                # the "set password" prompt in the GUI.
                pass
            else:
                record_auth_attempt(conn, user_id, False, reason="unexpected_password")
                raise AuthError("Invalid credentials")

        client_host = None
        client_name = None
        if client_metadata:
            client_host = client_metadata.get("host")
            client_name = client_metadata.get("name")

        session = create_session(
            conn,
            user_id=user_id,
            client_host=client_host,
            client_name=client_name,
            client_metadata=client_metadata,
        )

        record_auth_attempt(conn, user_id, True)
        conn.commit()
        return {"ok": True, "authenticated": True, **session}


def logout_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Revoke the current session."""
    if not auth or not isinstance(auth, dict):
        return {"ok": True}
    token = auth.get("token", "")
    if not token:
        return {"ok": True}
    with _get_db() as db:
        revoke_session_by_token(_get_conn(db), token)
    return {"ok": True}


def me_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the authenticated user info, or raise if anonymous."""
    with _get_db() as db:
        principal = principal_from_rpc_auth(_get_conn(db), auth)
        require_authenticated(principal)
        row = _get_conn(db).execute(
            "SELECT user_id, display_name, is_admin FROM flr_sample_users WHERE user_id = ?",
            (principal.user_id,),
        ).fetchone()
        if not row:
            raise AuthError("User not found")
        return {
            "ok": True,
            "user": {
                "user_id": row[0],
                "display_name": row[1],
                "is_admin": bool(row[2]),
            },
        }


def change_password_handler(
    auth: dict[str, Any] | None = None,
    password: str = "",
) -> dict[str, Any]:
    """Change the authenticated user's password.

    Uses the session token for authentication instead of legacy ``requester_id``.
    """
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)
        user_id = principal.user_id

        row = conn.execute(
            "SELECT is_admin FROM flr_sample_users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        is_target_admin = row and row[0] == 1

        if is_target_admin:
            if not password:
                raise AuthError("Admin password cannot be empty")
            strength = evaluate_password(password)
            if strength["score"] < 4:
                raise AuthError(
                    f"Admin password is too weak. Requirements: {', '.join(strength['feedback'])}"
                )

        password_hash = hash_password(password) if password else None
        with conn:
            conn.execute(
                "UPDATE flr_sample_users SET password_hash = ? WHERE user_id = ?",
                (password_hash, user_id),
            )
        return {"ok": True}


# ---- Session handlers ----


def sessions_list_handler(
    auth: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """List active sessions. Admin can see all; users see their own."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)

        if user_id and not principal.is_admin:
            user_id = principal.user_id

        sessions = list_sessions(conn, user_id=user_id)

        safe_sessions = []
        for s in sessions:
            safe = dict(s)
            safe.pop("token_hash", None)
            safe_sessions.append(safe)

        return {"ok": True, "sessions": safe_sessions}


def sessions_revoke_handler(
    auth: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Revoke a session by ID. Admin can revoke any; users revoke their own."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)

        if not principal.is_admin:
            row = conn.execute(
                "SELECT user_id FROM mfdb_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row or row[0] != principal.user_id:
                raise PermissionDenied()

        revoke_session(conn, session_id)
        return {"ok": True}


# ---- Group handlers ----


def groups_list_handler(auth: dict[str, Any] | None = None) -> dict[str, Any]:
    """List all groups."""
    with _get_db() as db:
        principal = principal_from_rpc_auth(_get_conn(db), auth)
        require_authenticated(principal)

        rows = _get_conn(db).execute(
            "SELECT * FROM mfdb_group WHERE deleted_at IS NULL ORDER BY display_name"
        ).fetchall()
        groups = [dict(r) for r in rows]
        return {"ok": True, "groups": groups}


def groups_get_handler(
    auth: dict[str, Any] | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    """Get a single group by ID."""
    with _get_db() as db:
        principal = principal_from_rpc_auth(_get_conn(db), auth)
        require_authenticated(principal)

        row = _get_conn(db).execute(
            "SELECT * FROM mfdb_group WHERE group_id = ? AND deleted_at IS NULL",
            (group_id,),
        ).fetchone()
        if not row:
            return {"ok": True, "group": None}
        return {"ok": True, "group": dict(row)}


def groups_create_handler(
    auth: dict[str, Any] | None = None,
    group: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new group. Admin only."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)
        if not principal.is_admin:
            raise PermissionDenied("Only admins can create groups")

        group_id = str(group.get("group_id", "")).strip()
        display_name = str(group.get("display_name", "")).strip()
        if not group_id:
            raise ValueError("group_id is required")
        if not display_name:
            group_id = display_name

        description = group.get("description")

        conn.execute(
            """INSERT INTO mfdb_group (group_id, display_name, description, created_by_user_id)
               VALUES (?, ?, ?, ?)""",
            (group_id, display_name, description, principal.user_id),
        )
        return {"ok": True, "group": {"group_id": group_id, "display_name": display_name}}


def groups_update_handler(
    auth: dict[str, Any] | None = None,
    group: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update a group. Admin or group manager only."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)

        group_id = str(group.get("group_id", "")).strip()
        if not group_id:
            raise ValueError("group_id is required")

        if not principal.is_admin:
            row = conn.execute(
                "SELECT 1 FROM mfdb_group_member WHERE group_id = ? AND user_id = ? AND role IN ('owner', 'manager') AND deleted_at IS NULL",
                (group_id, principal.user_id),
            ).fetchone()
            if not row:
                raise PermissionDenied()

        display_name = group.get("display_name")
        description = group.get("description")

        updates = []
        params = []
        if display_name is not None:
            updates.append("display_name = ?")
            params.append(display_name)
        if description is not None:
            updates.append("description = ?")
            params.append(description)

        if updates:
            params.append(group_id)
            conn.execute(
                f"UPDATE mfdb_group SET {', '.join(updates)} WHERE group_id = ?",
                params,
            )

        return {"ok": True}


def groups_delete_handler(
    auth: dict[str, Any] | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    """Soft-delete a group. Admin only. Built-in groups cannot be deleted."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)
        if not principal.is_admin:
            raise PermissionDenied("Only admins can delete groups")

        if group_id in ("admins", "users", "public"):
            raise ValueError("Built-in groups cannot be deleted")

        conn.execute(
            "UPDATE mfdb_group SET deleted_at = ? WHERE group_id = ?",
            (__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), group_id),
        )
        return {"ok": True}


# ---- Group member handlers ----


def members_list_handler(
    auth: dict[str, Any] | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    """List members of a group."""
    with _get_db() as db:
        principal = principal_from_rpc_auth(_get_conn(db), auth)
        require_authenticated(principal)

        rows = _get_conn(db).execute(
            """SELECT gm.*, u.display_name AS user_display_name
               FROM mfdb_group_member gm
               JOIN flr_sample_users u ON u.user_id = gm.user_id
               WHERE gm.group_id = ? AND gm.deleted_at IS NULL""",
            (group_id,),
        ).fetchall()
        return {"ok": True, "members": [dict(r) for r in rows]}


def members_add_handler(
    auth: dict[str, Any] | None = None,
    group_id: str | None = None,
    user_id: str | None = None,
    role: str = "member",
) -> dict[str, Any]:
    """Add a member to a group. Admin or group manager/owner only."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)

        if not principal.is_admin:
            row = conn.execute(
                "SELECT 1 FROM mfdb_group_member WHERE group_id = ? AND user_id = ? AND role IN ('owner', 'manager') AND deleted_at IS NULL",
                (group_id, principal.user_id),
            ).fetchone()
            if not row:
                raise PermissionDenied()

        conn.execute(
            """INSERT OR IGNORE INTO mfdb_group_member (group_id, user_id, role, created_by_user_id)
               VALUES (?, ?, ?, ?)""",
            (group_id, user_id, role, principal.user_id),
        )
        return {"ok": True}


def members_remove_handler(
    auth: dict[str, Any] | None = None,
    group_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Remove a member from a group. Admin or group manager/owner only."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)

        if not principal.is_admin:
            row = conn.execute(
                "SELECT 1 FROM mfdb_group_member WHERE group_id = ? AND user_id = ? AND role IN ('owner', 'manager') AND deleted_at IS NULL",
                (group_id, principal.user_id),
            ).fetchone()
            if not row:
                raise PermissionDenied()

        conn.execute(
            "UPDATE mfdb_group_member SET deleted_at = ? WHERE group_id = ? AND user_id = ?",
            (__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), group_id, user_id),
        )
        return {"ok": True}


# ---- Permission handlers ----


def _resolve_object_acl(conn, object_type, object_id):
    """Return the ACL for an object, or None."""
    row = conn.execute(
        "SELECT * FROM mfdb_object_acl WHERE object_type = ? AND object_id = ? AND deleted_at IS NULL",
        (object_type, object_id),
    ).fetchone()
    if not row:
        return None
    acl = dict(row)
    entries = conn.execute(
        "SELECT * FROM mfdb_acl_entry WHERE object_type = ? AND object_id = ? AND deleted_at IS NULL",
        (object_type, object_id),
    ).fetchall()
    acl["entries"] = [dict(e) for e in entries]
    return acl


def permissions_get_handler(
    auth: dict[str, Any] | None = None,
    object_type: str | None = None,
    object_id: str | None = None,
) -> dict[str, Any]:
    """Get the ACL for an object. Requires read access on the object."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)

        acl = _resolve_object_acl(conn, object_type, object_id)
        if acl is None:
            return {"ok": True, "acl": None}

        require_access(conn, principal, object_type, object_id, PERM_READ)
        return {"ok": True, "acl": acl}


def permissions_chmod_handler(
    auth: dict[str, Any] | None = None,
    object_type: str | None = None,
    object_id: str | None = None,
    mode: int | None = None,
) -> dict[str, Any]:
    """Change mode bits on an object. Requires manage (``x``)."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)
        chmod(conn, principal, object_type, object_id, mode)
        return {"ok": True}


def permissions_chown_handler(
    auth: dict[str, Any] | None = None,
    object_type: str | None = None,
    object_id: str | None = None,
    owner_user_id: str | None = None,
) -> dict[str, Any]:
    """Change object owner. Requires manage (``x``)."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)
        chown(conn, principal, object_type, object_id, owner_user_id)
        return {"ok": True}


def permissions_chgrp_handler(
    auth: dict[str, Any] | None = None,
    object_type: str | None = None,
    object_id: str | None = None,
    owner_group_id: str | None = None,
) -> dict[str, Any]:
    """Change object owning group. Requires manage (``x``)."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)
        chgrp(conn, principal, object_type, object_id, owner_group_id)
        return {"ok": True}


def permissions_grant_handler(
    auth: dict[str, Any] | None = None,
    object_type: str | None = None,
    object_id: str | None = None,
    subject_type: str | None = None,
    subject_id: str | None = None,
    permissions: int | None = None,
    effect: str = "allow",
) -> dict[str, Any]:
    """Grant or deny a permission on an object. Requires manage (``x``)."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)
        grant_acl(conn, principal, object_type, object_id, subject_type, subject_id, permissions, effect)
        return {"ok": True}


def permissions_revoke_handler(
    auth: dict[str, Any] | None = None,
    entry_id: int | None = None,
) -> dict[str, Any]:
    """Revoke an ACL entry. Requires manage (``x``) on the referenced object."""
    with _get_db() as db:
        conn = _get_conn(db)
        principal = principal_from_rpc_auth(conn, auth)
        require_authenticated(principal)
        revoke_acl(conn, principal, entry_id)
        return {"ok": True}


def revoke_session_by_token(conn, token):
    """Revoke a session by its raw token."""
    from chisurf.core.mfdb.auth import revoke_session_by_token as _revoke
    _revoke(conn, token)


def extract_principal_and_conn(auth):
    """Helper to get principal and connection from an auth dict."""
    db = _get_db()
    conn = _get_conn(db)
    principal = principal_from_rpc_auth(conn, auth)
    return principal, conn, db


def require_handler_auth(
    auth: dict[str, Any] | None = None,
) -> tuple[Any, sqlite3.Connection, Principal]:
    """Open the DB, extract principal, and require authentication.

    Returns ``(db, conn, principal)``. Caller must close ``db``.
    """
    from chisurf.core.mfdb.repository import MFDatabase
    db = MFDatabase(resolve_database_path())
    conn = db.conn
    principal = principal_from_rpc_auth(conn, auth)
    require_authenticated(principal)
    return db, conn, principal
