"""Branch / version-control access for :class:`~mfdb.repository.MFDatabase`.

Provides the ``mfdb_branch`` surface (create/fork branches, jump a user to an
operation, branch heads, and per-user active branch) as a mixin. Extracted
verbatim from the former repository god-class; behaviour is unchanged.
"""

from __future__ import annotations

import uuid
from typing import Any

from mfdb.schema._sqlutil import _exists, _row_to_dict, _utc_now


class BranchMixin:
    """Version-control-style branches over the operation/provenance graph."""

    def create_branch(
        self,
        branch_uuid: str | None = None,
        name: str | None = None,
        parent_branch_uuid: str | None = None,
        head_operation_id: str | None = None,
        created_by_user_id: str | None = None,
        description: str | None = None,
    ) -> str:
        if not branch_uuid:
            branch_uuid = str(uuid.uuid4())
        if not name:
            raise ValueError("Branch name cannot be empty")
        if parent_branch_uuid is not None and self.get_branch(parent_branch_uuid) is None:
            raise ValueError(f"Parent branch {parent_branch_uuid!r} does not exist")
        if head_operation_id is not None:
            if not _exists(self.conn, "mfdb_operation", "operation_id", head_operation_id):
                raise ValueError(f"Operation {head_operation_id!r} does not exist")

        now = _utc_now()
        with self._transaction():
            existing = self.conn.execute(
                "SELECT branch_uuid FROM mfdb_branch WHERE name = ? AND deleted_at IS NULL",
                (name,)
            ).fetchone()
            if existing:
                raise ValueError(f"Branch name {name!r} already exists")

            self.conn.execute(
                """INSERT INTO mfdb_branch (
                    branch_uuid, name, description, parent_branch_uuid,
                    head_operation_id, created_by_user_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    branch_uuid,
                    name,
                    description,
                    parent_branch_uuid,
                    head_operation_id,
                    created_by_user_id,
                    now,
                    now,
                ),
            )
            self.add_audit_log(
                action=f"Branch created: {name} ({branch_uuid})",
                target_type="branch",
                target_id=branch_uuid,
                details={"name": name, "parent_branch_uuid": parent_branch_uuid, "head_operation_id": head_operation_id},
            )
        return branch_uuid

    def fork_branch(
        self,
        source_branch_uuid: str,
        name: str,
        branch_uuid: str | None = None,
        head_operation_id: str | None = None,
        created_by_user_id: str | None = None,
        description: str | None = None,
    ) -> str:
        """Create a parallel branch from an existing branch head or older operation.

        Parameters
        ----------
        source_branch_uuid : str
            Existing branch used as the parent branch.
        name : str
            Name for the new branch.
        branch_uuid : str, optional
            Explicit branch UUID. A UUID is generated when omitted.
        head_operation_id : str, optional
            Operation that becomes the new branch head. When omitted, the
            source branch head is used.
        created_by_user_id : str, optional
            User creating the branch.
        description : str, optional
            Branch description.

        Returns
        -------
        str
            UUID of the created branch.
        """
        source = self.get_branch(source_branch_uuid)
        if source is None:
            raise ValueError(f"Source branch {source_branch_uuid!r} does not exist")
        fork_head = head_operation_id
        if fork_head is None:
            fork_head = source.get("head_operation_id")
        return self.create_branch(
            branch_uuid=branch_uuid,
            name=name,
            parent_branch_uuid=source["branch_uuid"],
            head_operation_id=fork_head,
            created_by_user_id=created_by_user_id,
            description=description,
        )

    def jump_user_to_operation(
        self,
        user_id: str,
        operation_id: str,
        branch_name: str | None = None,
        branch_uuid: str | None = None,
        parent_branch_uuid: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Move a user to a new branch rooted at a historical operation.

        Parameters
        ----------
        user_id : str
            User whose active branch should change.
        operation_id : str
            Existing operation to use as the new branch head.
        branch_name : str, optional
            Name for the created branch. A readable name is generated when
            omitted.
        branch_uuid : str, optional
            Explicit branch UUID. A UUID is generated when omitted.
        parent_branch_uuid : str, optional
            Parent branch for provenance. Defaults to the user's current active
            branch, or main when the user has no active branch.
        description : str, optional
            Branch description.

        Returns
        -------
        dict
            Created branch dictionary.
        """
        if not _exists(self.conn, "flr_sample_users", "user_id", user_id):
            raise ValueError(f"User {user_id!r} does not exist")
        if not _exists(self.conn, "mfdb_operation", "operation_id", operation_id):
            raise ValueError(f"Operation {operation_id!r} does not exist")

        if parent_branch_uuid is None:
            active = self.get_user_active_branch(user_id)
            parent_branch_uuid = (
                active["branch_uuid"]
                if active is not None
                else "00000000-0000-0000-0000-000000000000"
            )
        parent = self.get_branch(parent_branch_uuid)
        if parent is None:
            raise ValueError(f"Parent branch {parent_branch_uuid!r} does not exist")

        if not branch_name:
            short_operation = str(operation_id).replace(" ", "_")[:24]
            branch_name = f"{user_id}-at-{short_operation}"
        if description is None:
            description = f"Time-travel branch for {user_id} at operation {operation_id}"

        with self._transaction():
            created_uuid = self.create_branch(
                branch_uuid=branch_uuid,
                name=branch_name,
                parent_branch_uuid=parent["branch_uuid"],
                head_operation_id=operation_id,
                created_by_user_id=user_id,
                description=description,
            )
            self.set_user_active_branch(user_id, created_uuid)
            branch = self.get_branch(created_uuid)
            self.add_audit_log(
                action=f"User {user_id} jumped to operation {operation_id}",
                target_type="user",
                target_id=user_id,
                details={
                    "branch_uuid": created_uuid,
                    "parent_branch_uuid": parent["branch_uuid"],
                    "head_operation_id": operation_id,
                },
            )
        return branch

    def get_branch(self, branch_uuid_or_name: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM mfdb_branch WHERE (branch_uuid = ? OR name = ?) AND deleted_at IS NULL",
            (branch_uuid_or_name, branch_uuid_or_name)
        ).fetchone()
        return _row_to_dict(row)

    def list_branches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM mfdb_branch WHERE deleted_at IS NULL ORDER BY name"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def update_branch_head(self, branch_uuid: str, head_operation_id: str | None) -> None:
        if head_operation_id is not None:
            if not _exists(self.conn, "mfdb_operation", "operation_id", head_operation_id):
                raise ValueError(f"Operation {head_operation_id!r} does not exist")

        now = _utc_now()
        with self._transaction():
            self.conn.execute(
                "UPDATE mfdb_branch SET head_operation_id = ?, updated_at = ? WHERE branch_uuid = ?",
                (head_operation_id, now, branch_uuid)
            )
            self.add_audit_log(
                action=f"Branch {branch_uuid} head updated to {head_operation_id}",
                target_type="branch",
                target_id=branch_uuid,
                details={"head_operation_id": head_operation_id},
            )

    def delete_branch(self, branch_uuid: str) -> None:
        if branch_uuid == "00000000-0000-0000-0000-000000000000":
            raise ValueError("Cannot delete the main branch")

        with self._transaction():
            active_count = self.conn.execute(
                "SELECT COUNT(*) FROM flr_sample_users WHERE active_branch_uuid = ?",
                (branch_uuid,)
            ).fetchone()[0]
            if active_count > 0:
                raise ValueError("Cannot delete branch because it is currently the active branch for one or more users")

            now = _utc_now()
            self.conn.execute(
                "UPDATE mfdb_branch SET deleted_at = ?, updated_at = ? WHERE branch_uuid = ?",
                (now, now, branch_uuid)
            )
            self.add_audit_log(
                action=f"Branch deleted: {branch_uuid}",
                target_type="branch",
                target_id=branch_uuid,
            )

    def set_user_active_branch(self, user_id: str, branch_uuid: str) -> None:
        with self._transaction():
            if not _exists(self.conn, "flr_sample_users", "user_id", user_id):
                raise ValueError(f"User {user_id!r} does not exist")
            if not _exists(self.conn, "mfdb_branch", "branch_uuid", branch_uuid):
                raise ValueError(f"Branch {branch_uuid!r} does not exist")

            self.conn.execute(
                "UPDATE flr_sample_users SET active_branch_uuid = ? WHERE user_id = ?",
                (branch_uuid, user_id)
            )
            self.add_audit_log(
                action=f"User {user_id} active branch set to {branch_uuid}",
                target_type="user",
                target_id=user_id,
                details={"active_branch_uuid": branch_uuid},
            )

    def get_user_active_branch(self, user_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """SELECT b.* FROM mfdb_branch b
               JOIN flr_sample_users u ON u.active_branch_uuid = b.branch_uuid
               WHERE u.user_id = ? AND b.deleted_at IS NULL""",
            (user_id,)
        ).fetchone()
        if row:
            return _row_to_dict(row)
        return self.get_branch("00000000-0000-0000-0000-000000000000")
