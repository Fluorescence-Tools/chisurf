"""Artifact, operation, edge, and provenance-graph queries.

The provenance core: artifacts, operations, operation↔artifact links, processing
runs, provenance edges, and DAG traversal (ancestors/descendants, upstream/
downstream, impact, export) — extracted from the MFDatabase god-class (PRD-26).
Methods run on the shared ``self.conn``/``self.dao``/``self.lineage`` and resolve
cross-concern calls via the MFDatabase MRO.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid

from typing import Any

from mfdb.models import (
    DIRECTIONS,
    OPERATION_TYPES,
    RELATIONSHIP_TYPES,
    STATUS_VALUES,
    STORAGE_MODES,
    VALIDATION_STATUS_VALUES,
    validate_vocabulary,
)
from mfdb.provenance.graph import map_legacy_node_type
from mfdb.schema._sqlutil import _exists, _json_dumps, _json_hash, _utc_now, _validate_checksum


class ArtifactOpsMixin:
    def refresh_materialized_views(self):
        for view_name in schema.MATERIALIZED_VIEWS:
            self.conn.execute("DELETE FROM " + view_name)
            self.conn.execute("INSERT INTO " + view_name + " SELECT * FROM " + view_name + "__source")

    def add_operation(
        self,
        operation_id: str,
        operation_type: str,
        name: str | None = None,
        description: str | None = None,
        status: str = "pending",
        workflow_id: str | None = None,
        parent_operation_id: str | None = None,
        details: dict[str, Any] | str | None = None,
    ) -> str:
        """Register an operation using the canonical operation table.

        Parameters
        ----------
        operation_id : str
            Unique operation identifier.
        operation_type : str
            Operation vocabulary value.
        name : str, optional
            Display name stored in operation metadata.
        description : str, optional
            Description stored in operation metadata.
        status : str, default='pending'
            Lifecycle status.
        workflow_id : str, optional
            Parent workflow identifier stored in operation metadata.
        parent_operation_id : str, optional
            Parent operation identifier stored in operation metadata.
        details : dict or str, optional
            Additional metadata stored in operation metadata.

        Returns
        -------
        str
            The operation identifier.
        """
        metadata: dict[str, Any] | None = None
        if any(value is not None for value in (name, description, workflow_id, parent_operation_id, details)):
            metadata = {
                "name": name,
                "description": description,
                "workflow_id": workflow_id,
                "parent_operation_id": parent_operation_id,
                "details": details,
            }
        return self.record_operation(operation_id, operation_type, status=status, metadata=metadata)

    def update_operation(self, operation_id, **kwargs):
        allowed = {
            "operation_type", "experiment_id", "setup_id", "status", "operator_user_id",
            "software_package", "software_module", "software_version",
            "runtime_environment_json", "started_at", "ended_at", "error_message",
            "traceback_summary", "settings_json", "metadata_json",
        }
        if not kwargs:
            return
        kwargs["updated_at"] = _utc_now()
        cols, vals = [], []
        for key, value in kwargs.items():
            if key not in allowed and key != "updated_at":
                raise ValueError(f"Unsupported operation column: {key}")
            cols.append(f"{key} = ?")
            vals.append(value)
        vals.append(operation_id)
        with self._transaction():
            self.conn.execute(f"UPDATE mfdb_operation SET {', '.join(cols)} WHERE operation_id = ?", vals)

    def update_operation_settings(self, operation_id, settings: dict):
        now = _utc_now()
        blob = _json_dumps(settings)
        with self._transaction():
            self.conn.execute("UPDATE mfdb_operation SET settings_json = ?, updated_at = ? WHERE operation_id = ?", (blob, now, operation_id))

    def update_operation_status(self, operation_id, status):
        now = _utc_now()
        with self._transaction():
            self.conn.execute("UPDATE mfdb_operation SET status = ?, updated_at = ? WHERE operation_id = ?", (status, now, operation_id))

    def get_operations(self, operation_type=None, status=None, workflow_id=None):
        query = "SELECT * FROM mfdb_operation WHERE 1=1 AND deleted_at IS NULL"
        params = []
        if operation_type:
            query += " AND operation_type = ?"
            params.append(operation_type)
        if status:
            query += " AND status = ?"
            params.append(status)
        if workflow_id:
            query += " AND json_extract(metadata_json, '$.workflow_id') = ?"
            params.append(workflow_id)
        query += " ORDER BY created_at DESC"
        return self.conn.execute(query, params).fetchall()

    def add_artifact(
        self,
        artifact_id: str,
        artifact_kind: str,
        name: str | None = None,
        description: str | None = None,
        storage_mode: str = "local_file",
        file_path: str | None = None,
        file_format: str | None = None,
        content_type: str | None = None,
        file_size_bytes: int | None = None,
        md5: str | None = None,
        data_format: str | None = None,
        external_id: str | None = None,
        details: dict[str, Any] | str | None = None,
    ) -> str:
        """Register an artifact using the canonical artifact table.

        Parameters
        ----------
        artifact_id : str
            Unique artifact identifier.
        artifact_kind : str
            Artifact vocabulary value.
        name : str, optional
            Display name stored in artifact metadata.
        description : str, optional
            Description stored in artifact metadata.
        storage_mode : str, default='local_file'
            Storage mode vocabulary value.
        file_path : str, optional
            File path passed through as legacy metadata.
        file_format : str, optional
            Legacy data format.
        content_type : str, optional
            MIME type passed through as legacy metadata.
        file_size_bytes : int, optional
            Size passed through as legacy metadata.
        md5 : str, optional
            Checksum passed through as legacy metadata.
        data_format : str, optional
            Canonical data format.
        external_id : str, optional
            External identifier.
        details : dict or str, optional
            Additional metadata.

        Returns
        -------
        str
            The artifact identifier.
        """
        metadata = {
            "name": name,
            "description": description,
            "file_format": file_format,
            "content_type": content_type,
            "file_size_bytes": file_size_bytes,
            "md5": md5,
            "external_id": external_id,
            "details": details,
        }
        checksum_value = md5
        checksum_algorithm = "md5" if md5 else None
        return self.register_artifact(
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            storage_mode=storage_mode,
            file_path=file_path,
            data_format=data_format or file_format,
            checksum=checksum_value,
            checksum_algorithm=checksum_algorithm,
            size_bytes=file_size_bytes,
            mime_type=content_type,
            metadata=metadata,
        )

    def update_artifact(self, artifact_id, **kwargs):
        allowed = {
            "artifact_kind", "data_format", "experiment_id", "storage_mode", "file_path",
            "url", "folder_path", "mime_type", "size_bytes", "checksum",
            "checksum_algorithm", "row_count", "validation_status", "validation_message",
            "data_json", "data_blob", "metadata_json",
        }
        if not kwargs:
            return
        kwargs["updated_at"] = _utc_now()
        cols, vals = [], []
        for key, value in kwargs.items():
            if key not in allowed and key != "updated_at":
                raise ValueError(f"Unsupported artifact column: {key}")
            cols.append(f"{key} = ?")
            vals.append(value)
        vals.append(artifact_id)
        with self._transaction():
            self.conn.execute(f"UPDATE mfdb_artifact SET {', '.join(cols)} WHERE artifact_id = ?", vals)

    def set_artifact_validation(
        self,
        artifact_id: str,
        validation_status: str,
        validation_message: str | None = None,
    ) -> dict[str, Any]:
        """Set validation status for a live artifact.

        Parameters
        ----------
        artifact_id : str
            Artifact identifier.
        validation_status : str
            New validation status vocabulary value.
        validation_message : str, optional
            Human-readable validation note.

        Returns
        -------
        dict
            Updated artifact row.
        """
        validate_vocabulary(validation_status, VALIDATION_STATUS_VALUES, "validation_status")
        artifact = self.get_artifact(artifact_id)
        if artifact is None or artifact.get("deleted_at"):
            raise KeyError(f"Artifact not found: {artifact_id}")
        self.update_artifact(
            artifact_id,
            validation_status=validation_status,
            validation_message=validation_message,
        )
        updated = self.get_artifact(artifact_id)
        if updated is None:
            raise KeyError(f"Artifact not found after update: {artifact_id}")
        return updated

    def delete_artifact(self, artifact_id: str) -> dict[str, Any]:
        """Soft-delete an artifact and its direct provenance links.

        Parameters
        ----------
        artifact_id : str
            Artifact identifier.

        Returns
        -------
        dict
            Deletion result with the deleted artifact id.
        """
        artifact = self.get_artifact(artifact_id)
        if artifact is None or artifact.get("deleted_at"):
            raise KeyError(f"Artifact not found: {artifact_id}")
        now = _utc_now()
        with self._transaction():
            self.conn.execute(
                "UPDATE mfdb_artifact SET deleted_at = ?, updated_at = ? WHERE artifact_id = ?",
                (now, now, artifact_id),
            )
            self.conn.execute(
                "UPDATE mfdb_operation_artifact SET deleted_at = ? WHERE artifact_id = ?",
                (now, artifact_id),
            )
            self.conn.execute(
                "UPDATE mfdb_artifact_owner SET deleted_at = ? WHERE artifact_id = ?",
                (now, artifact_id),
            )
            self.conn.execute(
                """UPDATE mfdb_edge SET deleted_at = ?
                   WHERE ((source_node_type IN ('artifact', 'raw_data', 'processed_data')
                           AND source_node_id = ?)
                      OR (target_node_type IN ('artifact', 'raw_data', 'processed_data')
                           AND target_node_id = ?))""",
                (now, artifact_id, artifact_id),
            )
        return {"ok": True, "artifact_id": artifact_id, "deleted_at": now}

    def get_artifacts(self, artifact_kind=None):
        query = "SELECT * FROM mfdb_artifact WHERE 1=1 AND deleted_at IS NULL"
        params = []
        if artifact_kind:
            query += " AND artifact_kind = ?"
            params.append(artifact_kind)
        query += " ORDER BY artifact_id"
        return self.conn.execute(query, params).fetchall()

    def add_operation_artifact(self, operation_id, artifact_id, role="output", direction="output"):
        with self._transaction():
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO mfdb_operation_artifact "
                "(operation_id, artifact_id, role, direction, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (operation_id, artifact_id, role, direction, now, now, None)
            )

    def get_operation_artifacts(self, operation_id, direction=None):
        query = (
            "SELECT mfdb_artifact.*, mfdb_operation_artifact.role, "
            "mfdb_operation_artifact.direction "
            "FROM mfdb_operation_artifact "
            "JOIN mfdb_artifact ON mfdb_artifact.artifact_id = mfdb_operation_artifact.artifact_id "
            "WHERE mfdb_operation_artifact.operation_id = ?"
            " AND mfdb_operation_artifact.deleted_at IS NULL"
            " AND mfdb_artifact.deleted_at IS NULL"
        )
        params = [operation_id]
        if direction:
            query += " AND mfdb_operation_artifact.direction = ?"
            params.append(direction)
        query += " ORDER BY mfdb_operation_artifact.role"
        return self.conn.execute(query, params).fetchall()

    def remove_operation_artifact(self, operation_id, artifact_id):
        with self._transaction():
            now = _utc_now()
            self.conn.execute(
                "UPDATE mfdb_operation_artifact SET deleted_at = ? "
                "WHERE operation_id = ? AND artifact_id = ?",
                (now, operation_id, artifact_id)
            )

    def add_provenance_edge(
        self,
        edge_id: str | None = None,
        source_artifact_id: str | None = None,
        target_artifact_id: str | None = None,
        relationship_type: str | None = None,
        direction: str = "downstream",
        details: str | None = None,
        **kwargs,
    ):
        src_id = kwargs.get("source_node_id", source_artifact_id)
        tgt_id = kwargs.get("target_node_id", target_artifact_id)
        rel_type = relationship_type or kwargs.get("relationship_type", None)
        if rel_type not in {"input_to", "produced"}:
            validate_vocabulary(rel_type or "derived_from", RELATIONSHIP_TYPES, "relationship_type")

        if rel_type == "input_to":
            self.record_operation_link(
                operation_id=tgt_id,
                artifact_id=src_id,
                direction="input",
                role=kwargs.get("role"),
                checksum_snapshot=kwargs.get("checksum_snapshot"),
            )
            return
        elif rel_type == "produced":
            self.record_operation_link(
                operation_id=src_id,
                artifact_id=tgt_id,
                direction="output",
                role=kwargs.get("role"),
                checksum_snapshot=kwargs.get("checksum_snapshot"),
            )
            return

        with self._transaction():
            src_id = kwargs.pop("source_node_id", source_artifact_id)
            tgt_id = kwargs.pop("target_node_id", target_artifact_id)
            src_type = kwargs.pop("source_node_type", None)
            tgt_type = kwargs.pop("target_node_type", None)
            rel_type = relationship_type or kwargs.pop("relationship_type", None)
            metadata = {}
            op_id = None
            if kwargs.get("processing_id"):
                op_id = kwargs.pop("processing_id")
                metadata["processing_id"] = op_id
            if kwargs.get("settings_hash"):
                metadata["settings_hash"] = kwargs.pop("settings_hash")
            if kwargs.get("checksum_snapshot"):
                metadata["checksum_snapshot"] = kwargs.pop("checksum_snapshot")
            if kwargs.get("software_version"):
                metadata["software_version"] = kwargs.pop("software_version")
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO mfdb_edge "
                "(source_node_type, source_node_id, target_node_type, "
                "target_node_id, relationship_type, operation_id, metadata_json, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (src_type or "artifact", src_id, tgt_type or "artifact", tgt_id,
                 rel_type or "derived_from", op_id,
                 _json_dumps(metadata) if metadata else None,
                 now, now, None)
            )

    def get_downstream_artifacts(self, artifact_id):
        return self.conn.execute(
            "SELECT * FROM mfdb_edge WHERE source_node_id = ? AND deleted_at IS NULL", (artifact_id,)
        ).fetchall()

    def get_upstream_artifacts(self, artifact_id):
        return self.conn.execute(
            "SELECT * FROM mfdb_edge WHERE target_node_id = ? AND deleted_at IS NULL", (artifact_id,)
        ).fetchall()

    def get_downstream_dependencies(
        self, node_type: str, node_id: str
    ) -> list[dict[str, Any]]:
        if not node_type or not node_id:
            return []
        from mfdb.provenance.graph import traverse_canonical_graph
        edges = traverse_canonical_graph(self.conn, node_type, node_id, direction="downstream")
        results = []
        for edge in edges:
            meta = edge.get("metadata") or {}
            d = {
                "edge_id": edge["edge_id"],
                "source_node_type": edge["source_node_type"],
                "source_node_id": edge["source_node_id"],
                "target_node_type": edge["target_node_type"],
                "target_node_id": edge["target_node_id"],
                "relationship_type": edge["relationship_type"],
                "operation_id": meta.get("processing_id") or meta.get("operation_id") or edge.get("operation_id"),
                "settings_hash": meta.get("settings_hash"),
                "timestamp": meta.get("timestamp"),
                "software_version": meta.get("software_version"),
                "checksum_snapshot_json": _json_dumps(meta.get("checksum_snapshot")),
                "metadata_json": _json_dumps(meta),
            }
            results.append(d)
        return results

    def get_upstream_dependencies(
        self, node_type: str, node_id: str
    ) -> list[dict[str, Any]]:
        if not node_type or not node_id:
            return []
        from mfdb.provenance.graph import traverse_canonical_graph
        edges = traverse_canonical_graph(self.conn, node_type, node_id, direction="upstream")
        results = []
        for edge in edges:
            meta = edge.get("metadata") or {}
            d = {
                "edge_id": edge["edge_id"],
                "source_node_type": edge["source_node_type"],
                "source_node_id": edge["source_node_id"],
                "target_node_type": edge["target_node_type"],
                "target_node_id": edge["target_node_id"],
                "relationship_type": edge["relationship_type"],
                "operation_id": meta.get("processing_id") or meta.get("operation_id") or edge.get("operation_id"),
                "settings_hash": meta.get("settings_hash"),
                "timestamp": meta.get("timestamp"),
                "software_version": meta.get("software_version"),
                "checksum_snapshot_json": _json_dumps(meta.get("checksum_snapshot")),
                "metadata_json": _json_dumps(meta),
            }
            results.append(d)
        return results

    def export_provenance_graph(
        self,
        seed_node_type: str,
        seed_node_id: str,
    ) -> dict[str, Any]:
        """Export a JSON-serializable provenance graph of all related nodes and edges.

        Parameters
        ----------
        seed_node_type : str
            The seed node type (e.g. 'analysis_run' or 'processed_data').
        seed_node_id : str
            The seed node identifier.

        Returns
        -------
        dict
            Dict with "nodes" and "edges" keys.
        """
        from mfdb.provenance.graph import normalize_node_type, traverse_canonical_graph

        upstream_edges = traverse_canonical_graph(self.conn, seed_node_type, seed_node_id, direction="upstream")
        downstream_edges = traverse_canonical_graph(self.conn, seed_node_type, seed_node_id, direction="downstream")

        seen_edges = set()
        edges = []
        for edge in upstream_edges + downstream_edges:
            eid = edge.get("edge_id")
            if eid not in seen_edges:
                seen_edges.add(eid)
                edges.append(edge)

        seed_norm_type = normalize_node_type(seed_node_type)
        if seed_norm_type == "operation":
            seed_key = ("operation", seed_node_id)
        elif seed_norm_type == "parameter":
            seed_key = ("parameter", seed_node_id)
        else:
            seed_key = ("artifact", seed_node_id)
        node_keys = {seed_key}
        for edge in edges:
            node_keys.add((edge["source_node_type"], edge["source_node_id"]))
            node_keys.add((edge["target_node_type"], edge["target_node_id"]))

        nodes = []
        for n_type, n_id in sorted(node_keys):
            node_dict = {
                "node_id": n_id,
                "node_type": n_type,
            }
            norm_type = normalize_node_type(n_type)
            if norm_type == "artifact":
                row = self.conn.execute("SELECT * FROM mfdb_artifact WHERE artifact_id = ?", (n_id,)).fetchone()
                if row:
                    node_dict.update(dict(row))
            elif norm_type == "operation":
                row = self.conn.execute("SELECT * FROM mfdb_operation WHERE operation_id = ?", (n_id,)).fetchone()
                if row:
                    node_dict.update(dict(row))
            nodes.append(node_dict)

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def get_artifact_ancestors(self, artifact_id: str) -> list[str]:
        """Artifact IDs ``artifact_id`` was (transitively) derived from."""
        return self.lineage.ancestors(artifact_id)

    def get_artifact_descendants(self, artifact_id: str) -> list[str]:
        """Artifact IDs (transitively) derived from ``artifact_id``."""
        return self.lineage.descendants(artifact_id)

    def get_artifact_impact(self, node_id: str) -> list[str]:
        """Artifacts impacted by a change to ``node_id`` (PRD-05's impact query).

        The data-side of "when X changes, which results used it". Besides the
        transitive operation-graph descendants this also follows ``mfdb_edge`` usage
        links (e.g. a fit ``calibrated_by`` a calibration, a run that
        ``measured_sample`` a sample) so calibration/setup/reagent nodes resolve to
        the downstream artifacts they affect. See :meth:`Lineage.what_used`.
        """
        return self.lineage.what_used(node_id)

    def get_artifact_provenance_graph(
        self, artifact_id: str, *, depth: int = 100
    ) -> dict[str, list[dict[str, Any]]]:
        """Operation-graph provenance (nodes + edges) around an artifact."""
        return self.lineage.provenance_graph(artifact_id, depth=depth)

    def get_artifact_compute_spec(self, artifact_id: str):
        """Return the replayable compute spec for an artifact (PRD-21 Task 2).

        The producing operation captured as a unit (operation_type + parameters +
        source artifact ids), or ``None`` for a root/imported artifact.
        """
        from mfdb.provenance.compute_spec import get_compute_spec

        return get_compute_spec(self, artifact_id)

    def find_raw_artifact_by_md5(self, content_md5: str) -> str:
        """Return a raw-measurement artifact for object-store content MD5.

        Parameters
        ----------
        content_md5 : str
            MD5 hex digest of the raw file content.

        Returns
        -------
        str
            Existing raw-measurement artifact ID, or an empty string.

        """
        row = self.conn.execute(
            """SELECT artifact.artifact_id
               FROM mfdb_artifact artifact
               JOIN mfdb_object object_ref
                 ON object_ref.object_uuid = artifact.object_uuid
               WHERE object_ref.content_md5 = ?
                 AND artifact.artifact_kind = 'raw_measurement'
                 AND artifact.deleted_at IS NULL
               ORDER BY artifact.created_at DESC
               LIMIT 1""",
            (content_md5,),
        ).fetchone()
        return str(row["artifact_id"]) if row else ""

    def update_processing_run_status(
        self,
        run_id: str | None = None,
        status: str = "pending",
        photon_count: int | None = None,
        burst_count: int | None = None,
        selected_photon_count: int | None = None,
        error_message: str | None = None,
        traceback_summary: str | None = None,
        **kwargs,
    ):
        run_id = run_id or kwargs.pop("processing_id", None)
        if not run_id:
            raise ValueError("processing_id is required")
        now = _utc_now()
        updates = ["status = ?", "updated_at = ?"]
        params = [status, now]
        for key, value in (
            ("photon_count", photon_count),
            ("burst_count", burst_count),
            ("selected_photon_count", selected_photon_count),
            ("error_message", error_message),
            ("traceback_summary", traceback_summary),
        ):
            if value is not None:
                updates.append(f"{key} = ?")
                params.append(value)
        params.append(run_id)
        with self.conn:
            self.conn.execute(
                """UPDATE mfdb_operation
                   SET status = ?,
                       error_message = COALESCE(?, error_message),
                       traceback_summary = COALESCE(?, traceback_summary),
                       updated_at = ?
                   WHERE operation_id = ?""",
                (status, error_message, traceback_summary, now, run_id),
            )
        self.add_audit_log(
            action="update",
            target_type="processing_run",
            target_id=run_id,
            details={"status": status, "error_message": error_message},
        )

    def register_artifact(
        self,
        artifact_id: str,
        artifact_type: str | None = None,
        storage_mode: str = "local_file",
        experiment_id: str | None = None,
        file_path: str | None = None,
        url: str | None = None,
        folder_path: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        checksum: str | None = None,
        checksum_algorithm: str = "sha256",
        row_count: int | None = None,
        validation_status: str = "unvalidated",
        validation_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        data_json: str | None = None,
        data_blob: bytes | None = None,
        artifact_kind: str | None = None,
        data_format: str | None = None,
        object_uuid: str | None = None,
        created_by_user_id: str | None = None,
        is_public: bool | int | None = None,
    ) -> str:
        """Register or update an artifact in the canonical MFDB tables.

        Parameters
        ----------
        artifact_id : str
            Unique artifact identifier.
        artifact_type : str, optional
            Backward-compatible artifact kind name.
        storage_mode : str, default='local_file'
            Artifact storage vocabulary value.
        experiment_id : str, optional
            Associated experiment identifier.
        file_path : str, optional
            Local file path.
        url : str, optional
            Remote URL.
        folder_path : str, optional
            Local folder path.
        mime_type : str, optional
            MIME type.
        size_bytes : int, optional
            File size in bytes.
        checksum : str, optional
            Artifact checksum.
        checksum_algorithm : str, default='sha256'
            Checksum algorithm.
        row_count : int, optional
            Row count for tabular artifacts.
        validation_status : str, default='unvalidated'
            Validation status.
        validation_message : str, optional
            Validation message.
        metadata : dict, optional
            JSON-serializable metadata.
        data_json : str, optional
            Inline JSON payload.
        data_blob : bytes, optional
            Inline binary payload.
        artifact_kind : str, optional
            Canonical artifact kind.
        data_format : str, optional
            Data format vocabulary value.

        Returns
        -------
        str
            The artifact identifier.
        """
        kind = artifact_kind or artifact_type or "raw_data"
        self.validate_extensible_vocab("artifact_kind", kind)
        if data_format is not None:
            self.validate_extensible_vocab("data_format", data_format)
        validate_vocabulary(storage_mode, STORAGE_MODES, "storage_mode")
        validate_vocabulary(validation_status, VALIDATION_STATUS_VALUES, "validation_status")
        _validate_checksum(checksum, checksum_algorithm)
        now = _utc_now()
        with self._transaction():
            self.conn.execute(
                """INSERT INTO mfdb_artifact (
                    artifact_id, artifact_kind, data_format, experiment_id, storage_mode,
                    file_path, url, folder_path, mime_type, size_bytes, checksum,
                    checksum_algorithm, row_count, validation_status, validation_message,
                    metadata_json, data_json, data_blob, object_uuid,
                    created_by_user_id, is_public,
                    created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    artifact_kind=excluded.artifact_kind,
                    data_format=excluded.data_format,
                    experiment_id=excluded.experiment_id,
                    storage_mode=excluded.storage_mode,
                    file_path=excluded.file_path,
                    url=excluded.url,
                    folder_path=excluded.folder_path,
                    mime_type=excluded.mime_type,
                    size_bytes=excluded.size_bytes,
                    checksum=excluded.checksum,
                    checksum_algorithm=excluded.checksum_algorithm,
                    row_count=excluded.row_count,
                    validation_status=excluded.validation_status,
                    validation_message=excluded.validation_message,
                    metadata_json=excluded.metadata_json,
                    data_json=excluded.data_json,
                    data_blob=excluded.data_blob,
                    object_uuid=excluded.object_uuid,
                    created_by_user_id=excluded.created_by_user_id,
                    is_public=excluded.is_public,
                    updated_at=excluded.updated_at,
                    deleted_at=excluded.deleted_at""",
                (
                    artifact_id,
                    kind,
                    data_format,
                    experiment_id,
                    storage_mode,
                    file_path,
                    url,
                    folder_path,
                    mime_type,
                    size_bytes,
                    checksum,
                    checksum_algorithm,
                    row_count,
                    validation_status,
                    validation_message,
                    _json_dumps(metadata),
                    data_json,
                    data_blob,
                    object_uuid,
                    created_by_user_id,
                    1 if is_public else 0,
                    now,
                    now,
                    None,
                ),
            )
            self.add_audit_log(
                action="create",
                target_type=kind,
                target_id=artifact_id,
                details={"artifact_kind": kind, "storage_mode": storage_mode},
            )
        return artifact_id

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        # PRD-26 Task 1: parameterised, schema-driven get. ``include_deleted`` is
        # required to preserve the historical behaviour of returning artifacts
        # regardless of soft-delete (callers that need live-only filter explicitly).
        return self.dao.get("mfdb_artifact", artifact_id, include_deleted=True)

    def list_artifacts(
        self,
        artifact_type: str | None = None,
        experiment_id: str | None = None,
        artifact_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM mfdb_artifact WHERE 1=1 AND deleted_at IS NULL"
        params: list[Any] = []
        kind = artifact_kind or artifact_type
        if kind is not None:
            query += " AND artifact_kind = ?"
            params.append(kind)
        if experiment_id is not None:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        query += " ORDER BY created_at, artifact_id"
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    def record_operation(
        self,
        operation_id: str,
        operation_type: str,
        experiment_id: str | None = None,
        setup_id: str | None = None,
        settings: dict[str, Any] | None = None,
        operator_user_id: str | None = None,
        software_package: str | None = "chisurf",
        software_module: str | None = None,
        software_version: str | None = None,
        runtime_environment: dict[str, Any] | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        status: str = "pending",
        error_message: str | None = None,
        traceback_summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        setup_version: int | None = None,
        acl_owner_user_id: str | None = None,
        protocol_id: str | None = None,
        protocol_version: int | None = None,
    ) -> str:
        if operator_user_id is None:
            from mfdb.security.session import configured_default_user_id
            operator_user_id = configured_default_user_id()
        self.validate_extensible_vocab("operation_type", operation_type)
        validate_vocabulary(status, STATUS_VALUES, "status")
        settings_hash = _json_hash(settings)
        now = _utc_now()
        with self._transaction():
            operation_existed = _exists(self.conn, "mfdb_operation", "operation_id", operation_id)
            self.conn.execute(
                """INSERT INTO mfdb_operation (
                    operation_id, operation_type, experiment_id, setup_id,
                    settings_json, settings_hash, operator_user_id,
                    software_package, software_module, software_version,
                    runtime_environment_json, started_at, ended_at, status,
                    error_message, traceback_summary, metadata_json,
                    protocol_id, protocol_version,
                    created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(operation_id) DO UPDATE SET
                    operation_type=excluded.operation_type,
                    experiment_id=excluded.experiment_id,
                    setup_id=excluded.setup_id,
                    settings_json=excluded.settings_json,
                    settings_hash=excluded.settings_hash,
                    operator_user_id=excluded.operator_user_id,
                    software_package=excluded.software_package,
                    software_module=excluded.software_module,
                    software_version=excluded.software_version,
                    runtime_environment_json=excluded.runtime_environment_json,
                    started_at=excluded.started_at,
                    ended_at=excluded.ended_at,
                    status=excluded.status,
                    error_message=excluded.error_message,
                    traceback_summary=excluded.traceback_summary,
                    metadata_json=excluded.metadata_json,
                    protocol_id=excluded.protocol_id,
                    protocol_version=excluded.protocol_version,
                    updated_at=excluded.updated_at,
                    deleted_at=excluded.deleted_at""",
                (
                    operation_id,
                    operation_type,
                    experiment_id,
                    setup_id,
                    _json_dumps(settings),
                    settings_hash,
                    operator_user_id,
                    software_package,
                    software_module,
                    software_version,
                    _json_dumps(runtime_environment or self.default_runtime_environment()),
                    started_at,
                    ended_at,
                    status,
                    error_message,
                    traceback_summary,
                    _json_dumps(metadata),
                    protocol_id,
                    protocol_version,
                    now,
                    now,
                    None,
                ),
            )
            if not operation_existed:
                active_branch_row = self.conn.execute(
                    "SELECT active_branch_uuid FROM flr_sample_users WHERE user_id = ?",
                    (operator_user_id,)
                ).fetchone()
                active_branch_uuid = active_branch_row[0] if active_branch_row else None
                if not active_branch_uuid:
                    active_branch_uuid = "00000000-0000-0000-0000-000000000000"
                    if _exists(self.conn, "flr_sample_users", "user_id", operator_user_id):
                        self.conn.execute(
                            "UPDATE flr_sample_users SET active_branch_uuid = ? WHERE user_id = ?",
                            (active_branch_uuid, operator_user_id)
                        )
                if not _exists(self.conn, "mfdb_branch", "branch_uuid", active_branch_uuid):
                    self.conn.execute(
                        "INSERT OR IGNORE INTO mfdb_branch (branch_uuid, name, description) VALUES (?, 'main', 'Default main branch')",
                        (active_branch_uuid,)
                    )
                self.conn.execute(
                    "UPDATE mfdb_branch SET head_operation_id = ?, updated_at = ? WHERE branch_uuid = ?",
                    (operation_id, now, active_branch_uuid)
                )
            self.add_audit_log(
                action=f"Operation recorded: {operation_id}",
                target_type="operation",
                target_id=operation_id,
                details={"operation_type": operation_type, "status": status},
            )
            if acl_owner_user_id is not None:
                from mfdb.security.auth import create_default_acl_for_object
                create_default_acl_for_object(
                    self.conn, "mfdb_operation", operation_id,
                    owner_user_id=acl_owner_user_id,
                )
        return operation_id

    def get_operation(self, operation_id: str) -> dict[str, Any] | None:
        # PRD-26 Task 2: parameterised, schema-driven get-by-PK (was a hand SELECT).
        return self.dao.get("mfdb_operation", operation_id, include_deleted=True)

    def list_operations(
        self,
        operation_type: str | None = None,
        experiment_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM mfdb_operation WHERE 1=1 AND deleted_at IS NULL"
        params: list[Any] = []
        if operation_type is not None:
            query += " AND operation_type = ?"
            params.append(operation_type)
        if experiment_id is not None:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at, operation_id"
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    def transition_operation_status(
        self,
        operation_id: str,
        status: str,
        error_message: str | None = None,
        traceback_summary: str | None = None,
        operator_user_id: str | None = None,
    ) -> str:
        """Transition an operation status inside one audited transaction.

        Parameters
        ----------
        operation_id : str
            Existing operation identifier.
        status : str
            Target lifecycle status.
        error_message : str, optional
            Failure message for failed/cancelled transitions.
        traceback_summary : str, optional
            Traceback summary for failed transitions.
        operator_user_id : str, optional
            User performing the transition.

        Returns
        -------
        str
            The operation identifier.
        """
        validate_vocabulary(status, STATUS_VALUES, "status")
        now = _utc_now()
        with self._transaction():
            row = self.get_operation(operation_id)
            if row is None:
                raise ValueError(f"Unknown operation_id {operation_id!r}")
            self.conn.execute(
                """UPDATE mfdb_operation
                   SET status = ?, error_message = COALESCE(?, error_message),
                       traceback_summary = COALESCE(?, traceback_summary),
                       updated_at = ?
                   WHERE operation_id = ?""",
                (status, error_message, traceback_summary, now, operation_id),
            )
            self.add_audit_log(
                action="transition_status",
                target_type="operation",
                target_id=operation_id,
                operator_user_id=operator_user_id,
                details={
                    "previous_status": row["status"],
                    "status": status,
                    "error_message": error_message,
                },
            )
        return operation_id

    def record_operation_link(
        self,
        operation_id: str,
        artifact_id: str,
        direction: str,
        role: str | None = None,
        ordinal: int = 0,
        checksum_snapshot: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        validate_vocabulary(direction, DIRECTIONS, "direction")
        role = role or "generic"
        if isinstance(checksum_snapshot, (dict, list)):
            checksum_snapshot = _json_dumps(checksum_snapshot)
        with self._transaction():
            self.conn.execute(
                """INSERT INTO mfdb_operation_artifact (
                    operation_id, artifact_id, direction, role, ordinal,
                    checksum_snapshot, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(operation_id, artifact_id, direction, role) DO UPDATE SET
                    role=excluded.role,
                    ordinal=excluded.ordinal,
                    checksum_snapshot=excluded.checksum_snapshot,
                    metadata_json=excluded.metadata_json""",
                (
                    operation_id,
                    artifact_id,
                    direction,
                    role,
                    ordinal,
                    checksum_snapshot,
                    _json_dumps(metadata),
                ),
            )
            self.add_audit_log(
                action=f"Link added: {operation_id} -> {artifact_id}",
                target_type="operation_artifact",
                target_id=f"{operation_id}/{artifact_id}",
                details={"direction": direction, "role": role},
            )

    def _normalize_artifact_payload(
        self,
        artifact: dict[str, Any],
        default_experiment_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(artifact, dict):
            raise ValueError("artifact payload must be a mapping")
        artifact_id = artifact.get("artifact_id", artifact.get("id"))
        if not artifact_id:
            raise ValueError("artifact_id is required")
        kind = artifact.get("artifact_kind", artifact.get("artifact_type", "raw_data"))
        storage_mode = artifact.get("storage_mode", "local_file")
        validation_status = artifact.get("validation_status", "unvalidated")
        data_format = artifact.get("data_format")
        self.validate_extensible_vocab("artifact_kind", kind)
        if data_format is not None:
            self.validate_extensible_vocab("data_format", data_format)
        validate_vocabulary(storage_mode, STORAGE_MODES, "storage_mode")
        validate_vocabulary(validation_status, VALIDATION_STATUS_VALUES, "validation_status")
        return {
            "artifact_id": artifact_id,
            "artifact_kind": kind,
            "storage_mode": storage_mode,
            "experiment_id": artifact.get("experiment_id", default_experiment_id),
            "file_path": artifact.get("file_path"),
            "url": artifact.get("url"),
            "folder_path": artifact.get("folder_path"),
            "mime_type": artifact.get("mime_type"),
            "size_bytes": artifact.get("size_bytes"),
            "checksum": artifact.get("checksum"),
            "checksum_algorithm": artifact.get("checksum_algorithm", "sha256"),
            "row_count": artifact.get("row_count"),
            "validation_status": validation_status,
            "validation_message": artifact.get("validation_message"),
            "metadata": artifact.get("metadata"),
            "data_json": artifact.get("data_json"),
            "data_blob": artifact.get("data_blob"),
            "data_format": data_format,
            "role": artifact.get("role"),
            "ordinal": artifact.get("ordinal", 0),
            "link_metadata": artifact.get("link_metadata"),
        }

    def record_operation_with_artifacts(
        self,
        operation_id: str,
        operation_type: str,
        status: str = "pending",
        experiment_id: str | None = None,
        setup_id: str | None = None,
        settings: dict[str, Any] | None = None,
        operator_user_id: str | None = None,
        software_package: str | None = None,
        software_module: str | None = None,
        software_version: str | None = None,
        runtime_environment: dict[str, Any] | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        input_artifacts: list[dict[str, Any]] | None = None,
        output_artifacts: list[dict[str, Any]] | None = None,
        parameters: list[dict[str, Any]] | None = None,
        error_message: str | None = None,
        traceback_summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an operation, artifacts, links, and parameters atomically.

        Parameters
        ----------
        operation_id : str
            Unique operation identifier.
        operation_type : str
            Operation vocabulary value.
        status : str, default='pending'
            Initial operation status.
        experiment_id : str, optional
            Associated experiment identifier.
        setup_id : str, optional
            Associated setup identifier.
        settings : dict, optional
            Operation settings.
        operator_user_id : str, optional
            Operator user identifier.
        software_package : str, optional
            Software package name.
        software_module : str, optional
            Software module name.
        software_version : str, optional
            Software version.
        runtime_environment : dict, optional
            Runtime environment metadata.
        started_at : str, optional
            Start timestamp.
        ended_at : str, optional
            End timestamp.
        input_artifacts : list of dict, optional
            Artifact payloads to register and link as inputs.
        output_artifacts : list of dict, optional
            Artifact payloads to register and link as outputs.
        parameters : list of dict, optional
            Parameter payloads linked to the operation.
        error_message : str, optional
            Failure message.
        traceback_summary : str, optional
            Traceback summary.
        metadata : dict, optional
            Operation metadata.

        Returns
        -------
        dict
            Counts of registered artifacts, links, and parameters.
        """
        validate_vocabulary(operation_type, OPERATION_TYPES, "operation_type")
        validate_vocabulary(status, STATUS_VALUES, "status")
        input_payloads = [self._normalize_artifact_payload(art, experiment_id) for art in (input_artifacts or [])]
        output_payloads = [self._normalize_artifact_payload(art, experiment_id) for art in (output_artifacts or [])]
        parameter_payloads = [self._normalize_parameter_payload(param) for param in (parameters or [])]

        counts = {
            "operation_inserted": 0,
            "input_artifact_inserted": 0,
            "output_artifact_inserted": 0,
            "input_link_inserted": 0,
            "output_link_inserted": 0,
            "parameter_inserted": 0,
        }

        with self._transaction():
            operation_existed = _exists(self.conn, "mfdb_operation", "operation_id", operation_id)
            self.record_operation(
                operation_id=operation_id,
                operation_type=operation_type,
                status=status,
                experiment_id=experiment_id,
                setup_id=setup_id,
                settings=settings,
                operator_user_id=operator_user_id,
                software_package=software_package,
                software_module=software_module,
                software_version=software_version,
                runtime_environment=runtime_environment,
                started_at=started_at,
                ended_at=ended_at,
                error_message=error_message,
                traceback_summary=traceback_summary,
                metadata=metadata,
            )
            counts["operation_inserted"] = 0 if operation_existed else 1

            for art in input_payloads:
                artifact_kwargs = art.copy()
                role = artifact_kwargs.pop("role", None)
                ordinal = artifact_kwargs.pop("ordinal", 0)
                link_metadata = artifact_kwargs.pop("link_metadata", None)
                artifact_id = artifact_kwargs["artifact_id"]
                artifact_existed = _exists(self.conn, "mfdb_artifact", "artifact_id", artifact_id)
                self.register_artifact(**artifact_kwargs)
                counts["input_artifact_inserted"] += 0 if artifact_existed else 1
                link_existed = self.conn.execute(
                    """SELECT 1 FROM mfdb_operation_artifact
                       WHERE operation_id = ? AND artifact_id = ? AND direction = 'input'
                         AND role = ?""",
                    (operation_id, artifact_id, role or "generic"),
                ).fetchone() is not None
                self.record_operation_link(
                    operation_id=operation_id,
                    artifact_id=artifact_id,
                    direction="input",
                    role=role,
                    ordinal=ordinal,
                    metadata=link_metadata,
                )
                counts["input_link_inserted"] += 0 if link_existed else 1

            for art in output_payloads:
                artifact_kwargs = art.copy()
                role = artifact_kwargs.pop("role", None)
                ordinal = artifact_kwargs.pop("ordinal", 0)
                link_metadata = artifact_kwargs.pop("link_metadata", None)
                artifact_id = artifact_kwargs["artifact_id"]
                artifact_existed = _exists(self.conn, "mfdb_artifact", "artifact_id", artifact_id)
                self.register_artifact(**artifact_kwargs)
                counts["output_artifact_inserted"] += 0 if artifact_existed else 1
                link_existed = self.conn.execute(
                    """SELECT 1 FROM mfdb_operation_artifact
                       WHERE operation_id = ? AND artifact_id = ? AND direction = 'output'
                         AND role = ?""",
                    (operation_id, artifact_id, role or "generic"),
                ).fetchone() is not None
                self.record_operation_link(
                    operation_id=operation_id,
                    artifact_id=artifact_id,
                    direction="output",
                    role=role,
                    ordinal=ordinal,
                    metadata=link_metadata,
                )
                counts["output_link_inserted"] += 0 if link_existed else 1

            for param in parameter_payloads:
                parameter_uuid = param["parameter_uuid"]
                parameter_existed = _exists(self.conn, "mfdb_parameter", "parameter_uuid", parameter_uuid)
                self.record_parameter(operation_id=operation_id, **param)
                counts["parameter_inserted"] += 0 if parameter_existed else 1

        return {
            "operation_id": operation_id,
            "operation_inserted": counts["operation_inserted"],
            "input_artifact_inserted": counts["input_artifact_inserted"],
            "output_artifact_inserted": counts["output_artifact_inserted"],
            "parameter_inserted": counts["parameter_inserted"],
            "input_link_inserted": counts["input_link_inserted"],
            "output_link_inserted": counts["output_link_inserted"],
            "input_count": counts["input_artifact_inserted"],
            "output_count": counts["output_artifact_inserted"],
            "parameter_count": counts["parameter_inserted"],
            "input_link_count": counts["input_link_inserted"],
            "output_link_count": counts["output_link_inserted"],
        }

    def add_processing_run(
        self,
        run_id: str | None = None,
        processing_type: str = "burst_selection",
        **kwargs,
    ):
        import uuid
        processing_id = kwargs.pop("processing_id", None)
        oid = run_id or processing_id or f"proc_{uuid.uuid4().hex[:12]}"
        input_raw_data_ids = kwargs.pop("input_raw_data_ids", None) or []
        experiment_id = kwargs.pop("experiment_id", None)
        operator_user_id = kwargs.pop("operator_user_id", None)
        kwargs.pop("selected_setup_name", None)
        kwargs.pop("detector_definitions", None)
        kwargs.pop("pie_window_definitions", None)
        kwargs.pop("file_count", None)
        kwargs.pop("result_metadata", None)
        kwargs.pop("photon_count", None)
        kwargs.pop("burst_count", None)
        kwargs.pop("selected_photon_count", None)
        settings = kwargs.get("settings")
        settings_hash = _json_hash(settings) if settings else None
        with self._transaction():
            missing_inputs = [
                raw_id for raw_id in input_raw_data_ids
                if not _exists(self.conn, "mfdb_artifact", "artifact_id", raw_id)
            ]
            if missing_inputs:
                raise sqlite3.IntegrityError(
                    "Missing input raw data artifact(s): " + ", ".join(missing_inputs)
                )
            self.record_operation(
                operation_id=oid,
                operation_type=processing_type,
                experiment_id=experiment_id,
                operator_user_id=operator_user_id,
                **kwargs,
            )
            for raw_id in input_raw_data_ids:
                self.add_provenance_edge(
                    source_node_type="raw_data",
                    source_node_id=raw_id,
                    target_node_type="processing_run",
                    target_node_id=oid,
                    relationship_type="input_to",
                    processing_id=oid,
                    settings_hash=settings_hash,
                )
            self.add_audit_log(
                action="create",
                target_type="processing_run",
                target_id=oid,
                operator_user_id=operator_user_id,
                details={"experiment_id": experiment_id, "processing_type": processing_type},
            )
        return oid

    def get_provenance_edges(self, **kwargs) -> list[dict[str, Any]]:
        # 1. Fetch edges from mfdb_edge
        query = "SELECT * FROM mfdb_edge WHERE 1=1 AND deleted_at IS NULL"
        params = []
        for key in ("source_node_id", "target_node_id", "relationship_type"):
            val = kwargs.get(key)
            if val is not None:
                query += f" AND {key} = ?"
                params.append(val)
        pid = kwargs.get("processing_id") or kwargs.get("operation_id")
        if pid is not None:
            query += " AND (operation_id = ? OR (metadata_json IS NOT NULL AND json_extract(metadata_json, '$.processing_id') = ?))"
            params.extend([pid, pid])
        query += " ORDER BY edge_id"
        rows = self.conn.execute(query, params).fetchall()
        res = []

        src_type_filter = kwargs.get("source_node_type")
        tgt_type_filter = kwargs.get("target_node_type")

        for r in rows:
            d = dict(r)
            if "source_node_id" in d and "source_artifact_id" not in d:
                d["source_artifact_id"] = d["source_node_id"]
            if "target_node_id" in d and "target_artifact_id" not in d:
                d["target_artifact_id"] = d["target_node_id"]

            if src_type_filter:
                f_mapped = map_legacy_node_type(src_type_filter)
                r_mapped = map_legacy_node_type(d["source_node_type"])
                if f_mapped != r_mapped and src_type_filter != "artifact" and d["source_node_type"] != "artifact":
                    continue
            if tgt_type_filter:
                f_mapped = map_legacy_node_type(tgt_type_filter)
                r_mapped = map_legacy_node_type(d["target_node_type"])
                if f_mapped != r_mapped and tgt_type_filter != "artifact" and d["target_node_type"] != "artifact":
                    continue
            res.append(d)

        # Track seen edges for deduplication
        seen_edges = set()
        for d in res:
            seen_edges.add((
                map_legacy_node_type(d["source_node_type"]),
                d["source_node_id"],
                map_legacy_node_type(d["target_node_type"]),
                d["target_node_id"],
                d["relationship_type"]
            ))

        # 2. Fetch edges from mfdb_operation_artifact
        oa_query = "SELECT * FROM mfdb_operation_artifact WHERE 1=1 AND deleted_at IS NULL"
        oa_params = []

        def is_op_type(t):
            if not t:
                return False
            return map_legacy_node_type(t) in ("processing_run", "analysis_run")

        def is_art_type(t):
            if not t:
                return False
            return map_legacy_node_type(t) in ("processed_data", "raw_data")

        oa_op_id = kwargs.get("source_node_id") if is_op_type(kwargs.get("source_node_type")) else None
        if not oa_op_id:
            oa_op_id = kwargs.get("target_node_id") if is_op_type(kwargs.get("target_node_type")) else None
        if not oa_op_id:
            oa_op_id = pid

        oa_art_id = kwargs.get("source_node_id") if is_art_type(kwargs.get("source_node_type")) else None
        if not oa_art_id:
            oa_art_id = kwargs.get("target_node_id") if is_art_type(kwargs.get("target_node_type")) else None

        if oa_op_id:
            oa_query += " AND operation_id = ?"
            oa_params.append(oa_op_id)
        if oa_art_id:
            oa_query += " AND artifact_id = ?"
            oa_params.append(oa_art_id)
        rel = kwargs.get("relationship_type")
        if rel == "input_to":
            oa_query += " AND direction = 'input'"
        elif rel == "produced":
            oa_query += " AND direction = 'output'"
        oa_rows = self.conn.execute(oa_query, oa_params).fetchall()
        for r in oa_rows:
            art_kind = "artifact"
            art_row = self.conn.execute("SELECT artifact_kind FROM mfdb_artifact WHERE artifact_id = ?", (r["artifact_id"],)).fetchone()
            if art_row:
                art_kind = art_row["artifact_kind"]
            op_type = "processing_run"
            op_row = self.conn.execute("SELECT operation_type FROM mfdb_operation WHERE operation_id = ?", (r["operation_id"],)).fetchone()
            if op_row:
                op_type = op_row["operation_type"]
                if op_type in ("local_fit", "global_fit", "analysis"):
                    op_type = "analysis_run"
            if r["direction"] == "input":
                src_type = art_kind
                src_id = r["artifact_id"]
                tgt_type = op_type
                tgt_id = r["operation_id"]
                rel_type = "input_to"
            else:
                src_type = op_type
                src_id = r["operation_id"]
                tgt_type = art_kind
                tgt_id = r["artifact_id"]
                rel_type = "produced"

            if src_type_filter:
                f_mapped = map_legacy_node_type(src_type_filter)
                r_mapped = map_legacy_node_type(src_type)
                if f_mapped != r_mapped and src_type_filter != "artifact" and src_type != "artifact":
                    continue
            if tgt_type_filter:
                f_mapped = map_legacy_node_type(tgt_type_filter)
                r_mapped = map_legacy_node_type(tgt_type)
                if f_mapped != r_mapped and tgt_type_filter != "artifact" and tgt_type != "artifact":
                    continue
            if kwargs.get("relationship_type") and kwargs.get("relationship_type") != rel_type:
                continue

            edge_key = (
                map_legacy_node_type(src_type),
                src_id,
                map_legacy_node_type(tgt_type),
                tgt_id,
                rel_type
            )
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            d = {
                "edge_id": f"op_art_{r['operation_id']}_{r['artifact_id']}_{r['direction']}",
                "source_node_type": src_type,
                "source_node_id": src_id,
                "source_artifact_id": src_id,
                "target_node_type": tgt_type,
                "target_node_id": tgt_id,
                "target_artifact_id": tgt_id,
                "relationship_type": rel_type,
                "operation_id": r["operation_id"],
                "metadata_json": r["metadata_json"],
            }
            res.append(d)
        return res

    def get_processing_run_full(self, run_id: str) -> dict[str, Any] | None:
        row = self.get_operation(run_id)
        if row is None:
            return None
        row = dict(row)
        if "operation_type" in row and "processing_type" not in row:
            row["processing_type"] = row["operation_type"]
        if "operation_id" in row and "processing_id" not in row:
            row["processing_id"] = row["operation_id"]
        if "settings_json" in row and "settings" not in row:
            try:
                row["settings"] = json.loads(row["settings_json"]) if isinstance(row["settings_json"], str) else row["settings_json"]
            except (json.JSONDecodeError, TypeError):
                row["settings"] = row.get("settings_json")

        # Fetch inputs: raw_data ids from edges and operation_artifacts
        op_arts = self.conn.execute(
            "SELECT artifact_id FROM mfdb_operation_artifact WHERE operation_id = ? AND direction = 'input' AND deleted_at IS NULL",
            (run_id,)
        ).fetchall()
        raw_ids = [r["artifact_id"] for r in op_arts]

        raw_data_list = []
        for rid in raw_ids:
            art = self.get_artifact(rid)
            if art:
                art = dict(art)
                art["raw_data_id"] = art.get("artifact_id")
                if art.get("metadata_json"):
                    try:
                        meta = json.loads(art["metadata_json"]) if isinstance(art["metadata_json"], str) else art["metadata_json"]
                        if isinstance(meta, dict):
                            for k, v in meta.items():
                                if k not in art:
                                    art[k] = v
                    except (json.JSONDecodeError, TypeError):
                        pass
                raw_data_list.append(art)
        row["input_raw_data"] = raw_data_list

        # Fetch outputs: processed_data products
        products = self.get_processed_data_products(processing_id=run_id)
        row["processed_data"] = [self._decode_processed_data_row(p) for p in products]

        # Fetch edges
        row["provenance_edges"] = self.get_provenance_edges(processing_id=run_id)

        return row

    def get_processing_runs(self, **kwargs) -> list[dict[str, Any]]:
        return self.list_operations(**kwargs)

    def get_processing_run(self, run_id: str) -> dict[str, Any] | None:
        return self.get_operation(run_id)

    def _decode_provenance_edge_row(self, row):
        data = dict(row)
        if isinstance(data.get("checksum_snapshot_json"), str):
            try:
                data["checksum_snapshot"] = json.loads(data.pop("checksum_snapshot_json"))
            except json.JSONDecodeError:
                data["checksum_snapshot"] = data.pop("checksum_snapshot_json", None)
        else:
            data["checksum_snapshot"] = data.pop("checksum_snapshot_json", None)
        if isinstance(data.get("metadata_json"), str):
            try:
                data["metadata"] = json.loads(data.pop("metadata_json"))
            except json.JSONDecodeError:
                data["metadata"] = data.pop("metadata_json", None)
        else:
            data["metadata"] = data.pop("metadata_json", None)
        return data

    def add_edge(
        self,
        source_node_type: str,
        source_node_id: str,
        target_node_type: str,
        target_node_id: str,
        relationship_type: str,
        **kwargs,
    ) -> None:
        if relationship_type in ("input_to", "produced"):
            raise ValueError("Operation input/output links must use record_operation_link")
        validate_vocabulary(relationship_type, RELATIONSHIP_TYPES, "relationship_type")
        with self._transaction():
            now = _utc_now()
            self.conn.execute(
                """INSERT INTO mfdb_edge (
                    source_node_type, source_node_id, target_node_type,
                    target_node_id, relationship_type, operation_id, metadata_json,
                    created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    source_node_type, source_node_id,
                    target_node_type, target_node_id,
                    relationship_type, kwargs.get("operation_id"), _json_dumps(kwargs.get("metadata")),
                    now, now, None,
                ),
            )
            self.add_audit_log(
                action="create",
                target_type="edge",
                target_id=f"{source_node_type}:{source_node_id}->{target_node_type}:{target_node_id}",
                details={"relationship_type": relationship_type, "metadata": kwargs.get("metadata")},
            )

    def graph_upstream(self, node_type: str, node_id: str, max_depth: int = 100) -> list[dict[str, Any]]:
        from mfdb.provenance.graph import traverse_canonical_graph
        return traverse_canonical_graph(
            self.conn,
            node_type,
            node_id,
            direction="upstream",
            max_depth=max_depth,
            canonical=True,
        )

    def graph_downstream(self, node_type: str, node_id: str, max_depth: int = 100) -> list[dict[str, Any]]:
        from mfdb.provenance.graph import traverse_canonical_graph
        return traverse_canonical_graph(
            self.conn,
            node_type,
            node_id,
            direction="downstream",
            max_depth=max_depth,
            canonical=True,
        )

    link_operation_artifact = record_operation_link
