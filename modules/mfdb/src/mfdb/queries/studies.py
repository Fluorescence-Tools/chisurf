"""Study / project entity access for :class:`~mfdb.repository.MFDatabase`.

Provides the ``mfdb_study`` surface (studies, members, configurable fields, and
the project-id backfill) as a mixin. Extracted verbatim from the former
repository god-class; behaviour is unchanged.
"""

from __future__ import annotations

import uuid
from typing import Any

from mfdb._sqlutil import _utc_now


class StudyMixin:
    """Study/project grouping, membership, and configurable fields."""

    STUDY_MEMBER_TYPES = ("sample", "artifact")

    def create_study(
        self,
        name: str,
        description: str = "",
        *,
        is_public: bool = False,
        study_id: str | None = None,
        created_by_user_id: str | None = None,
    ) -> str:
        """Create a study/project and return its ``study_id`` (owner = active user)."""
        if not name:
            raise ValueError("study name is required")
        if created_by_user_id is None:
            from mfdb.session import configured_default_user_id
            created_by_user_id = configured_default_user_id()
        sid = study_id or str(uuid.uuid4())
        now = _utc_now()
        with self._transaction():
            self.conn.execute(
                "INSERT INTO mfdb_study (study_id, name, description, "
                "created_by_user_id, is_public, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sid, name, description or None, created_by_user_id,
                 1 if is_public else 0, now, now, None),
            )
            self.add_audit_log(
                action="create", target_type="study", target_id=sid,
                operator_user_id=created_by_user_id,
                details={"name": name, "is_public": bool(is_public)},
            )
        return sid

    def get_study(self, study_id: str) -> dict[str, Any] | None:
        """Return a study row by id, or ``None``."""
        row = self.conn.execute(
            "SELECT * FROM mfdb_study WHERE study_id = ? AND deleted_at IS NULL",
            (study_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_studies(
        self, scope: str = "all", owner_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List studies scoped ``mine``/``own`` | ``public`` | ``all`` (own+public)."""
        if owner_id is None and scope in ("mine", "own", "all"):
            from mfdb.session import configured_default_user_id
            owner_id = configured_default_user_id()
        where = ["deleted_at IS NULL"]
        params: list[Any] = []
        if scope in ("mine", "own"):
            where.append("created_by_user_id = ?")
            params.append(owner_id)
        elif scope == "public":
            where.append("is_public = 1")
        else:  # all
            where.append("(is_public = 1 OR created_by_user_id = ?)")
            params.append(owner_id)
        rows = self.conn.execute(
            f"SELECT * FROM mfdb_study WHERE {' AND '.join(where)} ORDER BY name",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def add_study_member(
        self, study_id: str, member_type: str, member_id: str, role: str = "member"
    ) -> None:
        """Add a sample/artifact to a study (idempotent on the unique triple)."""
        if member_type not in self.STUDY_MEMBER_TYPES:
            raise ValueError(
                f"unknown member_type {member_type!r}; expected {self.STUDY_MEMBER_TYPES}"
            )
        now = _utc_now()
        with self._transaction():
            exists = self.conn.execute(
                "SELECT 1 FROM mfdb_study_member WHERE study_id = ? AND member_type = ? "
                "AND member_id = ? AND deleted_at IS NULL",
                (study_id, member_type, member_id),
            ).fetchone()
            if exists:
                return
            next_id = (
                self.conn.execute(
                    "SELECT COALESCE(MAX(member_row_id), 0) FROM mfdb_study_member"
                ).fetchone()[0]
                + 1
            )
            self.conn.execute(
                "INSERT INTO mfdb_study_member (member_row_id, study_id, member_type, "
                "member_id, role, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (next_id, study_id, member_type, member_id, role or None, now, now, None),
            )

    def list_study_members(
        self, study_id: str, member_type: str | None = None
    ) -> list[dict[str, Any]]:
        """List a study's members, optionally filtered by ``member_type``."""
        sql = (
            "SELECT member_type, member_id, role FROM mfdb_study_member "
            "WHERE study_id = ? AND deleted_at IS NULL"
        )
        params: list[Any] = [study_id]
        if member_type is not None:
            sql += " AND member_type = ?"
            params.append(member_type)
        sql += " ORDER BY member_type, member_id"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def list_studies_for_member(
        self, member_type: str, member_id: str
    ) -> list[str]:
        """Return the ids of studies a sample/artifact belongs to (many-to-many)."""
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT study_id FROM mfdb_study_member "
                "WHERE member_type = ? AND member_id = ? AND deleted_at IS NULL",
                (member_type, member_id),
            ).fetchall()
        ]

    def set_study_field(self, study_id: str, key: str, value: str) -> None:
        """Set a configurable per-study metadata field (upsert on (study_id, key))."""
        now = _utc_now()
        with self._transaction():
            existing = self.conn.execute(
                "SELECT kv_id FROM mfdb_study_key_value WHERE study_id = ? AND key = ? "
                "AND deleted_at IS NULL",
                (study_id, key),
            ).fetchone()
            if existing:
                self.conn.execute(
                    "UPDATE mfdb_study_key_value SET value = ?, updated_at = ? "
                    "WHERE kv_id = ?",
                    (value, now, existing[0]),
                )
                return
            next_id = (
                self.conn.execute(
                    "SELECT COALESCE(MAX(kv_id), 0) FROM mfdb_study_key_value"
                ).fetchone()[0]
                + 1
            )
            self.conn.execute(
                "INSERT INTO mfdb_study_key_value (kv_id, study_id, key, value, "
                "details, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (next_id, study_id, key, value, None, now, now, None),
            )

    def get_study_fields(self, study_id: str) -> dict[str, str]:
        """Return a study's configurable fields as a ``{key: value}`` mapping."""
        rows = self.conn.execute(
            "SELECT key, value FROM mfdb_study_key_value WHERE study_id = ? "
            "AND deleted_at IS NULL ORDER BY key",
            (study_id,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def backfill_studies_from_project_ids(self) -> dict[str, int]:
        """Create one study per distinct ``flr_sample.project_id`` and join its samples.

        A one-shot, idempotent backfill (callable on demand — not a version migration,
        per PRD-19's disposable-DB policy): for each distinct non-empty ``project_id``
        on ``flr_sample`` that has no study of the same name yet, create a study named
        after the project_id and add each carrying sample as a member. Returns
        ``{"studies_created", "members_added"}``.
        """
        studies_created = 0
        members_added = 0
        rows = self.conn.execute(
            "SELECT project_id, sample_id FROM flr_sample "
            "WHERE project_id IS NOT NULL AND project_id != '' AND deleted_at IS NULL "
            "ORDER BY project_id"
        ).fetchall()
        by_project: dict[str, list[str]] = {}
        for project_id, sample_id in rows:
            by_project.setdefault(project_id, []).append(sample_id)
        for project_id, sample_ids in by_project.items():
            existing = self.conn.execute(
                "SELECT study_id FROM mfdb_study WHERE name = ? AND deleted_at IS NULL "
                "ORDER BY created_at LIMIT 1",
                (project_id,),
            ).fetchone()
            if existing:
                study_id = existing[0]
            else:
                study_id = self.create_study(
                    project_id, description=f"Backfilled from project_id {project_id!r}"
                )
                studies_created += 1
            for sample_id in sample_ids:
                before = self.conn.execute(
                    "SELECT COUNT(*) FROM mfdb_study_member WHERE study_id = ? "
                    "AND member_type = 'sample' AND member_id = ? AND deleted_at IS NULL",
                    (study_id, sample_id),
                ).fetchone()[0]
                self.add_study_member(study_id, "sample", sample_id)
                if not before:
                    members_added += 1
        return {"studies_created": studies_created, "members_added": members_added}
