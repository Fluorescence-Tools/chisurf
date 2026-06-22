"""Canonical identity / session context for MFDB (PRD-17).

One place resolves "who is the active user" and carries it, so that writes
(which stamp ownership) and reads (which scope "mine") always agree. Before this,
``cs_settings["mfdb"]["default_user_id"]`` was read independently in ~10 sites and
RPC handlers resolved the auth principal separately, which made the dataset
browser's "Mine" diverge from what registration stamped.

Use :func:`resolve_active_user_id` everywhere a user id is needed, and
:func:`resolve_session` at an entry point (GUI launch / RPC dispatch) to build the
:class:`SessionContext` that should be threaded onward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Fallback when no user is configured or authenticated.
DEFAULT_USER_ID = "user_default"


def configured_default_user_id() -> str:
    """Return the configured local default user.

    This is the identity that in-process registration stamps ownership with.
    """
    try:
        import chisurf.core.settings

        uid = chisurf.core.settings.cs_settings.get("mfdb", {}).get("default_user_id")
        if uid:
            return uid
    except Exception:
        pass
    return DEFAULT_USER_ID


def resolve_active_user_id(auth: dict[str, Any] | None = None, *, conn: Any = None) -> str:
    """Resolve the acting user id — the single resolver for reads and writes.

    Uses the authenticated principal when an ``auth`` payload and a database
    connection are supplied and the principal is not anonymous; otherwise falls
    back to the configured ``mfdb.default_user_id`` (then :data:`DEFAULT_USER_ID`).

    Parameters
    ----------
    auth : dict, optional
        RPC auth payload. When omitted (in-process GUI), the configured default
        user is used.
    conn : sqlite3.Connection, optional
        Connection used to resolve the principal from ``auth``.
    """
    if auth is not None and conn is not None:
        try:
            from chisurf.core.mfdb.auth import AnonymousPrincipal, principal_from_rpc_auth

            principal = principal_from_rpc_auth(conn, auth)
            if not isinstance(principal, AnonymousPrincipal):
                return principal.user_id
        except Exception:
            pass
    return configured_default_user_id()


@dataclass
class SessionContext:
    """The resolved identity + database for one entry point.

    Built once (GUI launch or RPC dispatch) and threaded onward instead of having
    each module re-resolve identity. ``user_id`` is the canonical acting user.
    """

    user_id: str
    db: Any = None
    is_admin: bool = False
    groups: tuple[str, ...] = ()
    auth: dict[str, Any] | None = None


def resolve_session(auth: dict[str, Any] | None = None, db: Any = None) -> SessionContext:
    """Build a :class:`SessionContext` once, at an entry point."""
    conn = getattr(db, "conn", None)
    user_id = resolve_active_user_id(auth, conn=conn)
    return SessionContext(user_id=user_id, db=db, auth=auth)
