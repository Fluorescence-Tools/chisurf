"""Object-store queries.

Content-addressed blob storage access (put/get/list/delete) and artifact-file
materialization — extracted from the MFDatabase god-class (PRD-26). Routes
through the object store; methods resolve cross-concern calls via the MFDatabase
MRO.
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any

from mfdb.schema._sqlutil import _json_dumps, _utc_now


class ObjectStoreMixin:
    def _get_object_store(self):
        """Return the shared ObjectStore instance, creating it if needed."""
        if not hasattr(self, "_object_store") or self._object_store is None:
            from mfdb.store.database_resolver import object_store_root
            from mfdb.store.object_store import ObjectStore
            self._object_store = ObjectStore(object_store_root())
        return self._object_store

    def put_object(
        self,
        path: str | os.PathLike | None = None,
        data: bytes | None = None,
        filename: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_by_user_uuid: str | None = None,
    ) -> dict[str, Any]:
        """Store a file or bytes in the object store and register in mfdb_object.

        Parameters
        ----------
        path : str or PathLike, optional
            Path to the file to store. Mutually exclusive with ``data``.
        data : bytes, optional
            Binary content to store. Mutually exclusive with ``path``.
        filename : str, optional
            Original filename to record in metadata.
        mime_type : str, optional
            MIME type of the content.
        metadata : dict, optional
            Additional metadata to store as JSON.
        created_by_user_uuid : str, optional
            UUID of the user who created the object.

        Returns
        -------
        dict
            Object reference with keys: ``object_uuid``, ``content_md5``,
            ``size_bytes``, ``original_filename``, ``deduplicated``, ``storage_path``.
        """
        store = self._get_object_store()
        if path is not None and data is not None:
            raise ValueError("Cannot specify both path and data")
        if path is not None:
            ref = store.put_from_path(Path(path), original_filename=filename)
        elif data is not None:
            ref = store.put_bytes(data, filename=filename or "unnamed")
        else:
            raise ValueError("Must specify either path or data")

        now = _utc_now()
        with self._transaction():
            existing = self.conn.execute(
                "SELECT object_uuid, refcount FROM mfdb_object WHERE content_md5 = ?",
                (ref.md5,),
            ).fetchone()
            if existing:
                self.conn.execute(
                    "UPDATE mfdb_object SET refcount = refcount + 1 WHERE content_md5 = ?",
                    (ref.md5,),
                )
                object_uuid = existing["object_uuid"]
                refcount = existing["refcount"] + 1
                deduplicated = True
            else:
                object_uuid = ref.uuid
                refcount = 1
                deduplicated = False
                self.conn.execute(
                    """INSERT INTO mfdb_object (
                        object_uuid, content_md5, original_filename, size_bytes,
                        mime_type, storage_path, refcount, metadata_json,
                        created_at, created_by_user_uuid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        object_uuid,
                        ref.md5,
                        ref.original_filename,
                        ref.size,
                        mime_type,
                        ref.storage_path,
                        refcount,
                        _json_dumps(metadata),
                        now,
                        created_by_user_uuid,
                    ),
                )
            self.add_audit_log(
                action="create" if not deduplicated else "reference",
                target_type="object",
                target_id=object_uuid,
                details={"content_md5": ref.md5, "deduplicated": deduplicated},
            )
        return {
            "object_uuid": object_uuid,
            "content_md5": ref.md5,
            "size_bytes": ref.size,
            "original_filename": ref.original_filename,
            "deduplicated": deduplicated,
            "storage_path": ref.storage_path,
            "refcount": refcount,
        }

    def get_object(self, object_uuid: str) -> bytes:
        """Retrieve blob content by object UUID.

        Parameters
        ----------
        object_uuid : str
            The object UUID.

        Returns
        -------
        bytes
            The stored content.
        """
        row = self.conn.execute(
            "SELECT content_md5 FROM mfdb_object WHERE object_uuid = ?",
            (object_uuid,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Object not found: {object_uuid}")
        store = self._get_object_store()
        return store.get(row["content_md5"])

    def get_object_info(self, object_uuid: str) -> dict[str, Any] | None:
        """Retrieve object metadata by UUID.

        Parameters
        ----------
        object_uuid : str
            The object UUID.

        Returns
        -------
        dict or None
            Object metadata, or None if not found.
        """
        # PRD-26 Task 2: parameterised, schema-driven get-by-PK (was a hand SELECT).
        return self.dao.get("mfdb_object", object_uuid, include_deleted=True)

    def get_object_path(self, object_uuid: str) -> Path:
        """Return the filesystem path for an object.

        Parameters
        ----------
        object_uuid : str
            The object UUID.

        Returns
        -------
        Path
            Path to the stored blob.
        """
        row = self.conn.execute(
            "SELECT content_md5 FROM mfdb_object WHERE object_uuid = ?",
            (object_uuid,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Object not found: {object_uuid}")
        store = self._get_object_store()
        return store.get_path(row["content_md5"])

    def materialize_artifact_file(self, artifact_id: str, *, into: str | None = None) -> str:
        """Copy an artifact's stored blob to a temp file and return its path.

        The object store is content-addressed, so its blob carries no file
        extension; readers such as ``tttrlib`` infer the container from the suffix.
        This copies the blob into a fresh temp file that carries the artifact's
        recorded ``data_format`` suffix so a re-read works — the materialization
        primitive behind replay/recompute (PRD-21 Task 2).

        Parameters
        ----------
        artifact_id : str
            Artifact whose stored object should be materialized.
        into : str, optional
            Directory for the temp file (defaults to the system temp dir).

        Returns
        -------
        str
            Path to the materialized copy.
        """
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            raise KeyError(f"artifact {artifact_id!r} not found")
        object_uuid = artifact.get("object_uuid")
        if not object_uuid:
            raise ValueError(
                f"artifact {artifact_id!r} has no stored object to materialize"
            )
        blob = str(self.get_object_path(object_uuid))
        data_format = (artifact.get("data_format") or "").lstrip(".")
        suffix = f".{data_format}" if data_format else ""
        fd, tmp = tempfile.mkstemp(prefix="mfdb_materialize_", suffix=suffix, dir=into)
        os.close(fd)
        shutil.copyfile(blob, tmp)
        return tmp

    def delete_object(self, object_uuid: str) -> dict[str, Any]:
        """Delete an object or decrement its refcount.

        Parameters
        ----------
        object_uuid : str
            The object UUID.

        Returns
        -------
        dict
            Result with keys: ``deleted`` (bool), ``refcount`` (int).
        """
        row = self.conn.execute(
            "SELECT content_md5, refcount FROM mfdb_object WHERE object_uuid = ?",
            (object_uuid,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Object not found: {object_uuid}")

        md5 = row["content_md5"]
        refcount = row["refcount"]

        with self._transaction():
            if refcount <= 1:
                self.conn.execute(
                    "DELETE FROM mfdb_object WHERE object_uuid = ?",
                    (object_uuid,),
                )
                store = self._get_object_store()
                store.delete(md5)
                self.add_audit_log(
                    action="delete",
                    target_type="object",
                    target_id=object_uuid,
                    details={"content_md5": md5, "blob_deleted": True},
                )
                return {"deleted": True, "refcount": 0}
            else:
                self.conn.execute(
                    "UPDATE mfdb_object SET refcount = refcount - 1 WHERE object_uuid = ?",
                    (object_uuid,),
                )
                self.add_audit_log(
                    action="dereference",
                    target_type="object",
                    target_id=object_uuid,
                    details={"content_md5": md5, "new_refcount": refcount - 1},
                )
                return {"deleted": False, "refcount": refcount - 1}

    def list_objects(
        self,
        filename: str | None = None,
        user_uuid: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List objects with optional filtering.

        Parameters
        ----------
        filename : str, optional
            Filter by original filename (substring match).
        user_uuid : str, optional
            Filter by creator user UUID.
        limit : int
            Maximum number of results.
        offset : int
            Offset for pagination.

        Returns
        -------
        list of dict
            List of object records.
        """
        query = "SELECT * FROM mfdb_object WHERE 1=1"
        params: list[Any] = []
        if filename is not None:
            query += " AND original_filename LIKE ?"
            params.append(f"%{filename}%")
        if user_uuid is not None:
            query += " AND created_by_user_uuid = ?"
            params.append(user_uuid)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]
