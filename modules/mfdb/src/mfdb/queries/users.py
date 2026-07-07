"""User and device access for :class:`~mfdb.repository.MFDatabase`.

Provides the ``flr_sample_users`` / ``flr_sample_devices`` surface (users,
device registry, and artifact ownership) as a mixin. Extracted verbatim from the
former repository god-class; behaviour is unchanged.
"""

from __future__ import annotations

import uuid

from mfdb.schema._sqlutil import _utc_now


class UserDeviceMixin:
    """People, devices, and artifact ownership."""

    def get_users(self):
        return self.conn.execute("SELECT * FROM flr_sample_users WHERE deleted_at IS NULL ORDER BY user_id").fetchall()

    def ensure_user(self, user_id: str, display_name: str | None = None) -> None:
        """Create a minimal ``flr_sample_users`` row if the user is absent.

        Ownership stamping (``created_by_user_id``) carries a foreign key to
        ``flr_sample_users``; a configured ``default_user_id`` that was never
        seeded (e.g. a personal user id) would otherwise fail the FK and roll
        back the whole registration. This makes the active user exist on demand.
        """
        if not user_id:
            return
        import uuid as _uuid
        self.conn.execute(
            "INSERT OR IGNORE INTO flr_sample_users (user_id, user_uuid, display_name) "
            "VALUES (?, ?, ?)",
            (user_id, str(_uuid.uuid4()), display_name or user_id),
        )

    def add_artifact_owner(self, artifact_id: str, user_id: str, role: str = "owner") -> None:
        """Add a co-owner to an artifact (idempotent).

        Datasets are many-to-many owned; registering or reusing content by the
        same user adds them to the owner set without duplicating the artifact.
        """
        if not artifact_id or not user_id:
            return
        self.ensure_user(user_id)
        self.conn.execute(
            "INSERT OR IGNORE INTO mfdb_artifact_owner (artifact_id, user_id, role) "
            "VALUES (?, ?, ?)",
            (artifact_id, user_id, role),
        )

    def list_artifact_owners(self, artifact_id: str) -> list[str]:
        """Return the user IDs that own an artifact."""
        rows = self.conn.execute(
            "SELECT user_id FROM mfdb_artifact_owner "
            "WHERE artifact_id = ? AND deleted_at IS NULL ORDER BY created_at",
            (artifact_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def add_user(self, user_id, display_name, email=None, affiliation=None, department=None, role=None, address=None, website=None, phone=None, details=None, user_uuid=None, is_admin=0, password_hash=None, allow_passwordless_login=None):
        if not user_uuid:
            # Check if user already has a uuid
            row = self.conn.execute("SELECT user_uuid FROM flr_sample_users WHERE user_id = ?", (user_id,)).fetchone()
            if row and row[0]:
                user_uuid = row[0]
            else:
                user_uuid = str(uuid.uuid4())

        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_sample_users "
                "(user_id, user_uuid, display_name, email, affiliation, department, role, address, website, phone, is_admin, allow_passwordless_login, password_hash, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, user_uuid, display_name, email, affiliation, department, role, address, website, phone, is_admin, allow_passwordless_login, password_hash, details,
                 now, now, None)
            )

    def delete_user(self, user_id):
        with self.conn:
            self.conn.execute("UPDATE flr_sample_users SET deleted_at = ? WHERE user_id = ?", (_utc_now(), user_id))

    def get_devices(self):
        return self.conn.execute("SELECT * FROM flr_sample_devices WHERE deleted_at IS NULL ORDER BY device_id").fetchall()

    def add_device(self, device_id, name, device_type=None, model=None, serial_number=None, location=None, owner=None, details=None):
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_sample_devices "
                "(device_id, name, device_type, model, serial_number, location, owner, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (device_id, name, device_type, model, serial_number, location, owner, details,
                 now, now, None)
            )

    def delete_device(self, device_id):
        with self.conn:
            self.conn.execute("UPDATE flr_sample_devices SET deleted_at = ? WHERE device_id = ?", (_utc_now(), device_id))
