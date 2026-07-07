"""Analysis-run queries.

Analysis runs, products, data, parameters, and metadata — extracted from the
MFDatabase god-class (PRD-26). Methods run on the shared ``self.conn``/``self.dao``
and resolve cross-concern calls via the MFDatabase MRO.
"""

from __future__ import annotations

import json
import sqlite3
import uuid

import numpy as np


from typing import Any

from mfdb.schema._sqlutil import _json_dumps, _json_loads, _utc_now


class AnalysisMixin:
    def add_analysis_run(
        self,
        analysis_id: str | None = None,
        operation_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        status: str = "pending",
        settings_json: Any = None,
        details: Any = None,
        **kwargs,
    ) -> str:
        import uuid
        aid = analysis_id or kwargs.pop("analysis_run_id", None) or operation_id or f"anal_{uuid.uuid4().hex[:12]}"
        analysis_type = kwargs.pop("analysis_type", operation_id or "local_fit")
        experiment_id = kwargs.pop("experiment_id", None)
        model_name = kwargs.pop("model_name", name)
        model_type = kwargs.pop("model_type", None)
        model_version = kwargs.pop("model_version", None)
        fit_structure = kwargs.pop("fit_structure", None)
        parameter_links = kwargs.pop("parameter_links", None)
        software_package = kwargs.pop("software_package", "chisurf")
        software_module = kwargs.pop("software_module", None)
        software_version = kwargs.pop("software_version", None)
        optimizer_settings = kwargs.pop("optimizer_settings", settings_json)
        covariance_matrix = kwargs.pop("covariance_matrix", None)
        goodness_of_fit = kwargs.pop("goodness_of_fit", kwargs.pop("goodness_of_fit_json", None))
        notes = kwargs.pop("notes", description)
        metadata = kwargs.pop("metadata", details)

        meta_dict = metadata if isinstance(metadata, dict) else {}
        meta_dict.update({
            "model_name": model_name,
            "model_type": model_type,
            "model_version": model_version,
            "fit_structure": fit_structure,
            "parameter_links": parameter_links,
            "goodness_of_fit": goodness_of_fit,
            "covariance_matrix": covariance_matrix,
            "notes": notes,
        })

        self.record_operation(
            operation_id=aid,
            operation_type=analysis_type,
            experiment_id=experiment_id,
            settings=optimizer_settings if isinstance(optimizer_settings, dict) else None,
            software_package=software_package,
            software_module=software_module,
            software_version=software_version,
            status=status,
            metadata=meta_dict,
        )

        self.add_audit_log(
            action="create",
            target_type="analysis_run",
            target_id=aid,
            details={"analysis_type": analysis_type, "model_name": model_name, "model_type": model_type},
        )
        return aid


    def update_analysis_run(self, analysis_run_id, **kwargs):
        allowed = {
            "operation_type", "experiment_id", "setup_id", "settings", "status",
            "operator_user_id", "software_package", "software_module",
            "software_version", "runtime_environment", "started_at", "ended_at",
            "error_message", "traceback_summary", "metadata",
        }
        if not kwargs:
            return
        kwargs["updated_at"] = _utc_now()
        cols, vals = [], []
        for key, value in kwargs.items():
            if key not in allowed and key != "updated_at":
                raise ValueError(f"Unsupported analysis run column: {key}")
            cols.append(f"{key} = ?")
            vals.append(value)
        vals.append(analysis_run_id)
        with self._transaction():
            self.conn.execute(f"UPDATE mfdb_operation SET {', '.join(cols)} WHERE operation_id = ?", vals)

    def get_analysis_runs(self, operation_id=None, status=None):
        query = "SELECT * FROM mfdb_operation WHERE 1=1 AND deleted_at IS NULL"
        params = []
        if operation_id:
            query += " AND operation_id = ?"
            params.append(operation_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        return [dict(row) for row in self.conn.execute(query, params).fetchall()]
    def update_analysis_record(self, analysis_id: str, **kwargs):
        existing = self.conn.execute(
            "SELECT * FROM flr_fret_analysis WHERE analysis_id = ?",
            (analysis_id,)
        ).fetchone()
        allowed = {
            "experiment_id", "sample_id", "type", "method",
            "sample_probe_id_1", "sample_probe_id_2", "forster_radius_id",
            "dataset_list_id", "external_file_id", "software_id", "details"
        }
        if existing:
            cols, vals = [], []
            for key, value in kwargs.items():
                if key in allowed:
                    cols.append(f"{key} = ?")
                    vals.append(value)
            if cols:
                vals.append(analysis_id)
                with self._transaction():
                    self.conn.execute(
                        f"UPDATE flr_fret_analysis SET {', '.join(cols)}, updated_at = ? WHERE analysis_id = ?",
                        vals[:-1] + [_utc_now(), analysis_id]
                    )
        else:
            cols = ["analysis_id"]
            vals = [analysis_id]
            for key, value in kwargs.items():
                if key in allowed:
                    cols.append(key)
                    vals.append(value)
            now = _utc_now()
            cols += ["created_at", "updated_at", "deleted_at"]
            vals += [now, now, None]
            placeholders = ", ".join(["?"] * len(cols))
            with self.conn:
                self.conn.execute(
                    f"INSERT INTO flr_fret_analysis ({', '.join(cols)}) VALUES ({placeholders})",
                    vals
                )

    def add_analysis_metadata(self, analysis_id: str, key: str, value: Any, details: str | None = None):
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO analysis_metadata (analysis_id, key, value, details, created_at, updated_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (analysis_id, key, str(value), details, now, now, None),
            )

    def get_analysis_metadata(self, analysis_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value FROM analysis_metadata WHERE analysis_id = ? AND deleted_at IS NULL ORDER BY key",
            (analysis_id,),
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_analysis_metadata(self, analysis_id: str, metadata: dict[str, Any]):
        old = self.get_analysis_metadata(analysis_id)
        with self.conn:
            for key in set(old) - set(metadata):
                now = _utc_now()
                self.conn.execute(
                    "UPDATE analysis_metadata SET deleted_at = ? WHERE analysis_id = ? AND key = ?",
                    (now, analysis_id, key),
                )
            for key, value in metadata.items():
                now = _utc_now()
                self.conn.execute(
                    "INSERT OR REPLACE INTO analysis_metadata (analysis_id, key, value, details, created_at, updated_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (analysis_id, key, str(value), None, now, now, None),
                )

    def delete_analysis_metadata(self, analysis_id: str, key: str):
        with self.conn:
            self.conn.execute(
                "UPDATE analysis_metadata SET deleted_at = ? WHERE analysis_id = ? AND key = ?",
                (_utc_now(), analysis_id, key),
            )

    def add_analysis_data(self, analysis_id: str, data_type: str, x_values: np.ndarray, y_values: np.ndarray, data_name: str | None = None, x_unit: str | None = None, y_unit: str | None = None, details: str | None = None) -> int:
        x_values = np.asarray(x_values, dtype=np.float64)
        y_values = np.asarray(y_values, dtype=np.float64)
        x_blob = x_values.tobytes()
        y_blob = y_values.tobytes()
        with self.conn:
            where = "analysis_id = ? AND data_type = ?"
            params = [analysis_id, data_type]
            if data_name is None:
                where += " AND data_name IS NULL"
            else:
                where += " AND data_name = ?"
                params.append(data_name)
            self.conn.execute(f"UPDATE analysis_data SET deleted_at = ? WHERE {where}", [_utc_now()] + params)
            now = _utc_now()
            self.conn.execute(
                """INSERT OR REPLACE INTO analysis_data
                   (analysis_id, data_type, data_name, x_values, y_values, x_unit, y_unit, details,
                    created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (analysis_id, data_type, data_name, x_blob, y_blob, x_unit, y_unit, details,
                 now, now, None),
            )
        row = self.conn.execute(
            "SELECT id FROM analysis_data WHERE analysis_id = ? AND data_type = ? AND data_name IS ? ORDER BY id DESC LIMIT 1",
            (analysis_id, data_type, data_name),
        ).fetchone()
        return int(row["id"]) if row else 0

    def get_analysis_data(self, analysis_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM analysis_data WHERE analysis_id = ? AND deleted_at IS NULL ORDER BY data_type, data_name, id",
            (analysis_id,),
        ).fetchall()

    def _decode_analysis_run_row(self, row: dict[str, Any] | sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        settings_raw = data.pop("settings_json", None) or data.pop("settings", None)
        metadata_raw = data.pop("metadata_json", None)
        if settings_raw:
            data["settings"] = _json_loads(settings_raw)
        if metadata_raw:
            data["metadata"] = _json_loads(metadata_raw)
        return data

    def get_analysis_run(self, analysis_id: str) -> sqlite3.Row | dict[str, Any] | None:
        row = self.conn.execute(
            """SELECT operation_id AS analysis_id, operation_type AS analysis_type,
                      experiment_id, software_package, software_module, software_version,
                      settings_json AS optimizer_settings_json, status AS convergence_status,
                      metadata_json, created_at, updated_at, deleted_at
               FROM mfdb_operation
               WHERE operation_id = ?""",
            (analysis_id,),
        ).fetchone()
        if row:
            d = dict(row)
            m = _json_loads(d.pop("metadata_json", None)) or {}
            d["model_name"] = m.get("model_name")
            d["model_type"] = m.get("model_type")
            d["model_version"] = m.get("model_version")
            d["notes"] = m.get("notes")
            d["fit_structure_json"] = _json_dumps(m.get("fit_structure"))
            d["parameter_links_json"] = _json_dumps(m.get("parameter_links"))
            d["optimizer_settings_json"] = d["optimizer_settings_json"] or _json_dumps(m.get("optimizer_settings"))
            d["covariance_matrix_json"] = _json_dumps(m.get("covariance_matrix"))
            d["goodness_of_fit_json"] = _json_dumps(m.get("goodness_of_fit"))
            d["metadata_json"] = _json_dumps(m)
            return d
        return None

    def get_analysis_run_full(self, analysis_id: str) -> dict[str, Any] | None:
        run_row = self.get_analysis_run(analysis_id)
        if run_row is None:
            return None
        run = self._decode_analysis_run_row(run_row)

        # Get parameters
        param_rows = self.conn.execute(
            "SELECT * FROM mfdb_parameter WHERE operation_id = ? AND deleted_at IS NULL ORDER BY parameter_id",
            (analysis_id,),
        ).fetchall()
        run["parameters"] = []
        for row in param_rows:
            d = dict(row)
            d["analysis_id"] = d.pop("operation_id", None)
            run["parameters"].append(self._decode_analysis_parameter_row(d))

        # Get input processed data via operation-artifact links
        run["input_processed_data"] = [
            self._decode_processed_data_row(self.get_artifact(row["artifact_id"]))
            for row in self.conn.execute(
                """SELECT artifact_id
                   FROM mfdb_operation_artifact
                   WHERE operation_id = ? AND direction = 'input' AND deleted_at IS NULL
                   ORDER BY ordinal, artifact_id""",
                (analysis_id,),
            ).fetchall()
        ]

        # Get output processed data via operation-artifact links
        run["processed_data"] = [
            self._decode_processed_data_row(self.get_artifact(row["artifact_id"]))
            for row in self.conn.execute(
                """SELECT artifact_id
                   FROM mfdb_operation_artifact
                   WHERE operation_id = ? AND direction = 'output' AND deleted_at IS NULL
                   ORDER BY ordinal, artifact_id""",
                (analysis_id,),
            ).fetchall()
        ]

        # Get sub-fits grouped in this analysis
        grouped_rows = self.conn.execute(
            """SELECT op.operation_id AS analysis_id, op.operation_type AS analysis_type,
                      op.experiment_id, op.software_package, op.software_module, op.software_version,
                      op.settings_json AS optimizer_settings_json, op.status AS convergence_status,
                      op.metadata_json, op.created_at, op.updated_at
               FROM mfdb_edge AS pe
               JOIN mfdb_operation AS op ON op.operation_id = pe.target_node_id
               WHERE pe.source_node_type = 'analysis_run' AND pe.source_node_id = ?
                 AND pe.target_node_type = 'analysis_run' AND pe.relationship_type = 'grouped_in'
                 AND pe.deleted_at IS NULL AND op.deleted_at IS NULL
               ORDER BY pe.edge_id""",
            (analysis_id,),
        ).fetchall()
        run["grouped_fits"] = []
        for row in grouped_rows:
            d = dict(row)
            m = _json_loads(d.pop("metadata_json", None)) or {}
            d["model_name"] = m.get("model_name")
            d["model_type"] = m.get("model_type")
            d["model_version"] = m.get("model_version")
            d["notes"] = m.get("notes")
            d["fit_structure_json"] = _json_dumps(m.get("fit_structure"))
            d["parameter_links_json"] = _json_dumps(m.get("parameter_links"))
            d["optimizer_settings_json"] = d["optimizer_settings_json"] or _json_dumps(m.get("optimizer_settings"))
            d["covariance_matrix_json"] = _json_dumps(m.get("covariance_matrix"))
            d["goodness_of_fit_json"] = _json_dumps(m.get("goodness_of_fit"))
            d["metadata_json"] = _json_dumps(m)
            run["grouped_fits"].append(self._decode_analysis_run_row(d))

        # Get provenance edges referencing this run
        run["provenance_edges"] = [
            self._decode_provenance_edge_row(row)
            for row in self.get_provenance_edges(processing_id=analysis_id)
        ]

        return run

    def list_analysis_runs(
        self,
        experiment_id: str | None = None,
        analysis_type: str | None = None,
    ) -> list[sqlite3.Row | dict[str, Any]]:
        query = """
            SELECT operation_id AS analysis_id, operation_type AS analysis_type, experiment_id,
                   software_package, software_module, software_version,
                   settings_json AS optimizer_settings_json, status AS convergence_status,
                   metadata_json, created_at, updated_at
            FROM mfdb_operation
            WHERE 1=1 AND deleted_at IS NULL
        """
        params: list[Any] = []
        if experiment_id is not None:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if analysis_type is not None:
            query += " AND operation_type = ?"
            params.append(analysis_type)
        else:
            query += " AND operation_type IN ('local_fit', 'global_fit', 'analysis', 'fitting', 'project_archive', 'project', 'decay_fit')"
        query += " ORDER BY created_at DESC"
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            m = _json_loads(d.pop("metadata_json", None)) or {}
            d["model_name"] = m.get("model_name")
            d["model_type"] = m.get("model_type")
            d["model_version"] = m.get("model_version")
            d["notes"] = m.get("notes")
            d["fit_structure_json"] = _json_dumps(m.get("fit_structure"))
            d["parameter_links_json"] = _json_dumps(m.get("parameter_links"))
            d["optimizer_settings_json"] = d["optimizer_settings_json"] or _json_dumps(m.get("optimizer_settings"))
            d["covariance_matrix_json"] = _json_dumps(m.get("covariance_matrix"))
            d["goodness_of_fit_json"] = _json_dumps(m.get("goodness_of_fit"))
            d["metadata_json"] = _json_dumps(m)
            results.append(d)
        return results

    def delete_analysis_run(self, analysis_id: str) -> None:
        parameter_ids = [
            row["parameter_uuid"]
            for row in self.conn.execute(
                "SELECT parameter_uuid FROM mfdb_parameter WHERE operation_id = ?",
                (analysis_id,),
            ).fetchall()
        ]

        with self.conn:
            now = _utc_now()
            for parameter_id in parameter_ids:
                self.conn.execute(
                    """UPDATE mfdb_edge SET deleted_at = ?
                       WHERE (source_node_type IN ('analysis_parameter', 'parameter') AND source_node_id = ?)
                          OR (target_node_type IN ('analysis_parameter', 'parameter') AND target_node_id = ?)""",
                    (now, parameter_id, parameter_id),
                )
                self.conn.execute(
                    "UPDATE mfdb_parameter SET deleted_at = ? WHERE parameter_uuid = ?",
                    (now, parameter_id),
                )
            self.conn.execute(
                """UPDATE mfdb_edge SET deleted_at = ?
                   WHERE (source_node_type = 'analysis_run' AND source_node_id = ?)
                      OR (target_node_type = 'analysis_run' AND target_node_id = ?)
                      OR operation_id = ?
                      OR (metadata_json IS NOT NULL AND json_extract(metadata_json, '$.processing_id') = ?)""",
                (now, analysis_id, analysis_id, analysis_id, analysis_id),
            )
            self.conn.execute("UPDATE mfdb_operation SET deleted_at = ? WHERE operation_id = ?", (now, analysis_id))
        self.add_audit_log(
            action="delete",
            target_type="analysis_run",
            target_id=analysis_id,
        )

    def add_analysis_parameter(
        self,
        analysis_id: str,
        name: str,
        value: float | None = None,
        standard_error: float | None = None,
        confidence_interval_low: float | None = None,
        confidence_interval_high: float | None = None,
        initial_value: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        bounds_on: bool = False,
        units: str | None = None,
        parameter_type: str = "free",
        expression: str | None = None,
        prior: dict[str, Any] | None = None,
        mapping: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parameter_uuid: str | None = None,
        **kwargs,
    ) -> str:
        if not analysis_id:
            raise ValueError("analysis_id is required")
        if not name:
            raise ValueError("name is required")
        uuid_str = parameter_uuid or f"param_{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        self.record_parameter(
            parameter_uuid=uuid_str,
            operation_id=analysis_id,
            name=name,
            value=value,
            standard_error=standard_error,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bounds_on=bounds_on,
            units=units,
            parameter_type=parameter_type,
            metadata=metadata,
        )
        return uuid_str

    def get_analysis_parameter(self, parameter_uuid: str) -> sqlite3.Row | dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM mfdb_parameter WHERE parameter_uuid = ?",
            (parameter_uuid,),
        ).fetchone()
        if row:
            d = dict(row)
            d["analysis_id"] = d.pop("operation_id", None)
            return d
        return None

    def add_analysis_product(
        self,
        analysis_id: str,
        product_type: str,
        storage_mode: str,
        processed_data_id: str | None = None,
        file_path: str | None = None,
        url: str | None = None,
        folder_path: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        checksum: str | None = None,
        checksum_algorithm: str = "sha256",
        row_count: int | None = None,
        product_summary: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        data_json: str | None = None,
        data_blob: bytes | None = None,
        validation_status: str = "unvalidated",
        validation_message: str | None = None,
        **kwargs,
    ) -> str:
        if not analysis_id:
            raise ValueError("analysis_id is required")
        if not product_type:
            raise ValueError("product_type is required")
        if not storage_mode:
            raise ValueError("storage_mode is required")
        prod_id = processed_data_id or f"prod_{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        self.add_processed_data_product(
            processing_id=analysis_id,
            product_type=product_type,
            storage_mode=storage_mode,
            processed_data_id=prod_id,
            file_path=file_path,
            url=url,
            folder_path=folder_path,
            mime_type=mime_type,
            size_bytes=size_bytes,
            checksum=checksum,
            checksum_algorithm=checksum_algorithm,
            row_count=row_count,
            metadata=metadata,
            data_json=data_json,
            data_blob=data_blob,
            validation_status=validation_status,
            validation_message=validation_message,
        )
        self.add_provenance_edge(
            source_node_type="analysis_run",
            source_node_id=analysis_id,
            target_node_type="processed_data",
            target_node_id=prod_id,
            relationship_type="produced",
            processing_id=analysis_id,
        )
        return prod_id

    def _decode_analysis_run_row(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        data["fit_structure"] = _json_loads(data.pop("fit_structure_json", None))
        data["parameter_links"] = _json_loads(data.pop("parameter_links_json", None))
        data["optimizer_settings"] = _json_loads(data.pop("optimizer_settings_json", None))
        data["covariance_matrix"] = _json_loads(data.pop("covariance_matrix_json", None))
        data["goodness_of_fit"] = _json_loads(data.pop("goodness_of_fit_json", None))
        data["metadata"] = _json_loads(data.pop("metadata_json", None))
        return data

    def _decode_analysis_parameter_row(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        data["bounds_on"] = bool(data["bounds_on"])
        data["prior"] = _json_loads(data.pop("prior_json", None))
        data["mapping"] = _json_loads(data.pop("mapping_json", None))
        data["metadata"] = _json_loads(data.pop("metadata_json", None))
        return data

    def link_analysis_parameters(self, source_param_uuid: str, target_param_uuid: str) -> None:
        src = self.get_analysis_parameter(source_param_uuid)
        self.add_provenance_edge(
            source_node_type="analysis_parameter",
            source_node_id=target_param_uuid,
            target_node_type="analysis_parameter",
            target_node_id=source_param_uuid,
            relationship_type="linked_to",
            processing_id=src["analysis_id"] if src else None,
        )
