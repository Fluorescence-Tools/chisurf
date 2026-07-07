"""User and device access for :class:`~mfdb.repository.MFDatabase`.

Provides the ``flr_sample_users`` / ``flr_sample_devices`` surface (users,
device registry, and artifact ownership) as a mixin. Extracted verbatim from the
former repository god-class; behaviour is unchanged.
"""

from __future__ import annotations

import uuid


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
        # INSERT OR IGNORE: create a minimal row only when the user is absent
        # (an existing row — even soft-deleted — is left untouched).
        if not self.dao.get("flr_sample_users", user_id, include_deleted=True):
            self.dao.insert(
                "flr_sample_users",
                {
                    "user_id": user_id,
                    "user_uuid": str(_uuid.uuid4()),
                    "display_name": display_name or user_id,
                },
            )

    def add_artifact_owner(self, artifact_id: str, user_id: str, role: str = "owner") -> None:
        """Add a co-owner to an artifact (idempotent).

        Datasets are many-to-many owned; registering or reusing content by the
        same user adds them to the owner set without duplicating the artifact.
        """
        if not artifact_id or not user_id:
            return
        self.ensure_user(user_id)
        # INSERT OR IGNORE on UNIQUE(artifact_id, user_id): an existing ownership
        # row (even soft-deleted) already occupying the pair is left untouched.
        if not self.dao.list(
            "mfdb_artifact_owner",
            filters={"artifact_id": artifact_id, "user_id": user_id},
            include_deleted=True,
            limit=1,
        ):
            self.dao.insert(
                "mfdb_artifact_owner",
                {"artifact_id": artifact_id, "user_id": user_id, "role": role},
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
            self.dao.upsert("flr_sample_users", {
                "user_id": user_id, "user_uuid": user_uuid, "display_name": display_name,
                "email": email, "affiliation": affiliation, "department": department, "role": role,
                "address": address, "website": website, "phone": phone, "is_admin": is_admin,
                "allow_passwordless_login": allow_passwordless_login, "password_hash": password_hash,
                "details": details, "deleted_at": None,
            })

    def delete_user(self, user_id):
        with self.conn:
            self.dao.soft_delete("flr_sample_users", user_id)

    def get_devices(self):
        return self.conn.execute("SELECT * FROM flr_sample_devices WHERE deleted_at IS NULL ORDER BY device_id").fetchall()

    def add_device(self, device_id, name, device_type=None, model=None, serial_number=None, location=None, owner=None, details=None):
        with self.conn:
            self.dao.upsert("flr_sample_devices", {
                "device_id": device_id, "name": name, "device_type": device_type, "model": model,
                "serial_number": serial_number, "location": location, "owner": owner,
                "details": details, "deleted_at": None,
            })

    def delete_device(self, device_id):
        with self.conn:
            self.dao.soft_delete("flr_sample_devices", device_id)
