"""Experiment access for :class:`~mfdb.repository.MFDatabase`.

Provides the ``flr_experiment`` surface (experiment types, experiments, and
experiment data rows) as a mixin. Extracted verbatim from the former repository
god-class; behaviour is unchanged.
"""

from __future__ import annotations

from mfdb._sqlutil import _utc_now


class ExperimentMixin:
    """Experiment types, experiments, and their data rows."""

    def add_experiment_type(self, name, category=None, description=None, details=None):
        if not name:
            raise ValueError("experiment type name is required")
        row = self.conn.execute("SELECT type_id FROM flr_experiment_type WHERE name = ?", (name,)).fetchone()
        if row:
            type_id = row["type_id"]
            with self.conn:
                self.conn.execute(
                    "UPDATE flr_experiment_type SET category = ?, description = ?, details = ?, updated_at = ? WHERE type_id = ?",
                    (category, description, details, _utc_now(), type_id)
                )
            return type_id
        else:
            with self.conn:
                now = _utc_now()
                cursor = self.conn.execute(
                    "INSERT INTO flr_experiment_type (name, category, description, details, created_at, updated_at, deleted_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (name, category, description, details, now, now, None)
                )
                return cursor.lastrowid

    def get_experiment_types(self):
        return self.conn.execute("SELECT * FROM flr_experiment_type WHERE deleted_at IS NULL ORDER BY category, name").fetchall()

    def delete_experiment_type(self, type_id):
        with self.conn:
            self.conn.execute("UPDATE flr_experiment_type SET deleted_at = ? WHERE type_id = ?", (_utc_now(), type_id))

    def add_experiment(self, experiment_id, type_id=None, sample_id=None, project_id=None, measured_by_user_id=None, measured_by_device_id=None, started_at=None, ended_at=None, status=None, details=None, setup_definition_id=None):
        if not experiment_id:
            raise ValueError("experiment_id is required")
        if measured_by_user_id is None:
            from mfdb.session import configured_default_user_id
            measured_by_user_id = configured_default_user_id()
        with self._transaction():
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_experiment "
                "(experiment_id, type_id, sample_id, project_id, measured_by_user_id, "
                "measured_by_device_id, started_at, ended_at, status, details, setup_definition_id, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (experiment_id, type_id, sample_id, project_id, measured_by_user_id,
                 measured_by_device_id, started_at, ended_at, status, details, setup_definition_id,
                 now, now, None)
            )

    def get_experiment(self, experiment_id):
        return self.conn.execute(
            "SELECT e.*, et.name AS experiment_type, et.category AS experiment_category, "
            "s.description AS sample_description, u.display_name AS measured_by_user, "
            "d.name AS measured_by_device, sd.name AS setup_name "
            "FROM flr_experiment AS e "
            "LEFT JOIN flr_experiment_type AS et ON et.type_id = e.type_id "
            "LEFT JOIN flr_sample AS s ON s.sample_id = e.sample_id "
            "LEFT JOIN flr_sample_users AS u ON u.user_id = e.measured_by_user_id "
            "LEFT JOIN flr_sample_devices AS d ON d.device_id = e.measured_by_device_id "
            "LEFT JOIN mfdb_setup AS sd ON sd.setup_id = e.setup_definition_id "
            "WHERE e.experiment_id = ? AND e.deleted_at IS NULL",
            (experiment_id,)
        ).fetchone()

    def get_experiments(self, sample_id=None, project_id=None, type_id=None):
        query = (
            "SELECT e.*, et.name AS experiment_type, et.category AS experiment_category, "
            "s.description AS sample_description, u.display_name AS measured_by_user, "
            "d.name AS measured_by_device "
            "FROM flr_experiment AS e "
            "LEFT JOIN flr_experiment_type AS et ON et.type_id = e.type_id "
            "LEFT JOIN flr_sample AS s ON s.sample_id = e.sample_id "
            "LEFT JOIN flr_sample_users AS u ON u.user_id = e.measured_by_user_id "
            "LEFT JOIN flr_sample_devices AS d ON d.device_id = e.measured_by_device_id "
            "WHERE 1=1 AND e.deleted_at IS NULL"
        )
        params = []
        if sample_id is not None:
            query += " AND e.sample_id = ?"
            params.append(sample_id)
        if project_id is not None:
            query += " AND e.project_id = ?"
            params.append(project_id)
        if type_id is not None:
            query += " AND e.type_id = ?"
            params.append(type_id)
        query += " ORDER BY e.started_at, e.experiment_id"
        return self.conn.execute(query, params).fetchall()

    def add_experiment_data(self, experiment_id, data_type, storage_mode, file_path=None, url=None, folder_path=None, mime_type=None, size_bytes=None, checksum=None, data_json=None, data_blob=None, reading_options_json=None, details=None):
        if not data_type:
            raise ValueError("data_type is required")
        if not storage_mode:
            raise ValueError("storage_mode is required")
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT INTO flr_experiment_data "
                "(experiment_id, data_type, storage_mode, file_path, url, folder_path, "
                "mime_type, size_bytes, checksum, data_json, data_blob, reading_options_json, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (experiment_id, data_type, storage_mode, file_path, url, folder_path,
                 mime_type, size_bytes, checksum, data_json, data_blob, reading_options_json, details,
                 now, now, None)
            )
            return int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def get_experiment_data(self, experiment_id):
        return self.conn.execute(
            "SELECT * FROM flr_experiment_data WHERE experiment_id = ? AND deleted_at IS NULL ORDER BY data_type, data_id",
            (experiment_id,)
        ).fetchall()

    def update_experiment_data(self, data_id, experiment_id, data_type, storage_mode, file_path=None, url=None, folder_path=None, mime_type=None, size_bytes=None, checksum=None, data_json=None, data_blob=None, reading_options_json=None, details=None):
        with self.conn:
            self.conn.execute(
                "UPDATE flr_experiment_data SET experiment_id=?, data_type=?, storage_mode=?, "
                "file_path=?, url=?, folder_path=?, mime_type=?, size_bytes=?, checksum=?, "
                "data_json=?, data_blob=?, reading_options_json=?, details=?, updated_at=? WHERE data_id=?",
                (experiment_id, data_type, storage_mode, file_path, url, folder_path,
                 mime_type, size_bytes, checksum, data_json, data_blob, reading_options_json,
                 details, _utc_now(), data_id)
            )

    def delete_experiment_data(self, data_id):
        with self.conn:
            self.conn.execute("UPDATE flr_experiment_data SET deleted_at = ? WHERE data_id = ?", (_utc_now(), data_id))

    def delete_experiment(self, experiment_id):
        with self.conn:
            self.conn.execute("UPDATE flr_experiment SET deleted_at = ? WHERE experiment_id = ?", (_utc_now(), experiment_id))
