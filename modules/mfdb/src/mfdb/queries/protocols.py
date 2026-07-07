"""Protocol entity access for :class:`~mfdb.repository.MFDatabase`.

Provides the ``mfdb_protocol`` surface (named, versioned, append-only procedures
and their declared parameter schema) as a mixin. Extracted verbatim from the
former repository god-class; behaviour is unchanged.
"""

from __future__ import annotations

import uuid
from typing import Any

from mfdb.schema._sqlutil import _utc_now


class ProtocolMixin:
    """Named, versioned measurement/processing/analysis procedures (PRD-14)."""

    #: Allowed protocol categories (declared as the .dic enumeration on
    #: mfdb_protocol.category).
    PROTOCOL_CATEGORIES = ("measurement", "processing", "analysis")

    def create_protocol(
        self,
        name: str,
        category: str,
        *,
        operation_type: str | None = None,
        setup_id: str | None = None,
        description: str = "",
        is_public: bool = False,
        created_by_user_id: str | None = None,
    ) -> tuple[str, int]:
        """Create a protocol (or a new version of an existing name); append-only.

        Returns ``(protocol_id, version)``. Editing a protocol means calling this
        again with the same ``name`` — it never mutates an existing row; a new row
        with ``version = max(version for name) + 1`` is recorded, so operations keep
        the exact version they ran. Owner defaults to the active user.
        """
        if category not in self.PROTOCOL_CATEGORIES:
            raise ValueError(
                f"unknown protocol category {category!r}; "
                f"expected one of {self.PROTOCOL_CATEGORIES}"
            )
        if not name:
            raise ValueError("protocol name is required")
        if created_by_user_id is None:
            from mfdb.security.session import configured_default_user_id
            created_by_user_id = configured_default_user_id()
        now = _utc_now()
        with self._transaction():
            prev = self.conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM mfdb_protocol "
                "WHERE name = ? AND deleted_at IS NULL",
                (name,),
            ).fetchone()[0]
            version = int(prev) + 1
            protocol_id = str(uuid.uuid4())
            self.conn.execute(
                "INSERT INTO mfdb_protocol (protocol_id, name, version, category, "
                "description, operation_type, setup_id, created_by_user_id, is_public, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    protocol_id, name, version, category, description or None,
                    operation_type, setup_id, created_by_user_id,
                    1 if is_public else 0, now, now, None,
                ),
            )
            self.add_audit_log(
                action="create",
                target_type="protocol",
                target_id=protocol_id,
                operator_user_id=created_by_user_id,
                details={"name": name, "version": version, "category": category},
            )
        return protocol_id, version

    def get_protocol(
        self, name: str, version: int | str = "latest"
    ) -> dict[str, Any] | None:
        """Return a protocol by ``name`` and ``version`` (default the latest)."""
        if version == "latest":
            row = self.conn.execute(
                "SELECT * FROM mfdb_protocol WHERE name = ? AND deleted_at IS NULL "
                "ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM mfdb_protocol WHERE name = ? AND version = ? "
                "AND deleted_at IS NULL",
                (name, int(version)),
            ).fetchone()
        return dict(row) if row else None

    def get_protocol_by_id(self, protocol_id: str) -> dict[str, Any] | None:
        """Return a specific protocol version row by its ``protocol_id``."""
        row = self.conn.execute(
            "SELECT * FROM mfdb_protocol WHERE protocol_id = ? AND deleted_at IS NULL",
            (protocol_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_protocol_versions(self, name: str) -> list[dict[str, Any]]:
        """Return all versions of a protocol ``name``, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM mfdb_protocol WHERE name = ? AND deleted_at IS NULL "
            "ORDER BY version",
            (name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_protocols(
        self, scope: str = "all", owner_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List the latest version of each protocol, scoped own/public/all.

        ``scope``: ``'own'`` (owned by ``owner_id``), ``'public'`` (``is_public=1``),
        or ``'all'`` (public + own). ``owner_id`` defaults to the active user.
        """
        if owner_id is None and scope in ("own", "all"):
            from mfdb.security.session import configured_default_user_id
            owner_id = configured_default_user_id()
        latest = (
            "version = (SELECT MAX(p2.version) FROM mfdb_protocol p2 "
            "WHERE p2.name = mfdb_protocol.name AND p2.deleted_at IS NULL)"
        )
        where = ["deleted_at IS NULL", latest]
        params: list[Any] = []
        if scope == "own":
            where.append("created_by_user_id = ?")
            params.append(owner_id)
        elif scope == "public":
            where.append("is_public = 1")
        else:  # all
            where.append("(is_public = 1 OR created_by_user_id = ?)")
            params.append(owner_id)
        rows = self.conn.execute(
            f"SELECT * FROM mfdb_protocol WHERE {' AND '.join(where)} ORDER BY name",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_protocol_parameter_schema(
        self, protocol: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the declared parameter schema for a protocol (no forked stack).

        The schema is the protocol's ``operation_type`` schema from PRD-11
        (``mfdb_operation_parameter_def``) — ``{name: OperationParameterDef}`` — so a
        protocol pins a named procedure to an operation kind without duplicating the
        parameter declarations.
        """
        from mfdb.provenance.operation_parameters import get_operation_parameter_defs

        operation_type = (protocol or {}).get("operation_type") or ""
        if not operation_type:
            return {}
        return get_operation_parameter_defs(self.conn, operation_type)
