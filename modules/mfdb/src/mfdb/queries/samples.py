"""Sample and entity queries.

Samples, entities, sample conditions/assemblies, probe positions, FRET Förster
radii, and sample↔artifact/object linkage — extracted from the MFDatabase
god-class (PRD-26). Methods run on the shared ``self.conn``/``self.dao`` and
resolve cross-concern calls via the MFDatabase MRO.
"""

from __future__ import annotations

import json
import sqlite3

from typing import Any

from mfdb.schema._sqlutil import _json_dumps, _utc_now


class SampleMixin:
    def get_entities(self):
        return self.conn.execute("SELECT * FROM entities WHERE deleted_at IS NULL ORDER BY entity_id").fetchall()

    def add_entity(self, entity_id, name, sequence=None, entity_type=None, organism=None, entity_source=None, details=None):
        with self.conn:
            # Dictionary-driven upsert (PRD-26) — no hand-written INSERT OR REPLACE.
            self.dao.upsert(
                "entities",
                {
                    "entity_id": entity_id,
                    "type": entity_type or "polymer",
                    "description": details or name,
                    "formula_weight": None,
                    "src_method": None,
                    "number_of_molecules": 1,
                    "common_name": name,
                    "deleted_at": None,
                },
            )
            # Note: sequence is stored in entity_poly_seq table, not in entities
            if sequence is not None:
                self.set_sequence(entity_id, sequence)

    def get_entity_by_name(self, name):
        return self.conn.execute("SELECT * FROM entities WHERE common_name = ?", (name,)).fetchone()

    def set_sequence(self, entity_id, sequence):
        """Set the sequence for an entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier.
        sequence : list of str
            List of residue names (mon_id values).

        """
        if not isinstance(sequence, (list, tuple)):
            sequence = list(sequence)
        with self.conn:
            now = _utc_now()
            # Delete existing sequence for this entity
            self.conn.execute(
                "DELETE FROM entity_poly_seq WHERE entity_id = ?", (entity_id,)
            )
            # Insert new sequence
            for idx, mon_id in enumerate(sequence, start=1):
                self.conn.execute(
                    "INSERT INTO entity_poly_seq (entity_id, num, mon_id, created_at, updated_at, deleted_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (entity_id, idx, str(mon_id), now, now, None)
                )

    def list_samples(self):
        return self.conn.execute(
            "SELECT s.sample_id, s.sample_uuid, s.description, s.details, "
            "s.num_of_probes, s.solvent_phase, s.sample_condition_id, "
            "s.entity_assembly_id, s.project_id, s.measured_by_user_id, "
            "s.measured_by_device_id, s.measured_at, "
            "COUNT(sp.sample_probe_id) AS mapped_probe_count, "
            "u.display_name AS measured_by_user, "
            "d.name AS measured_by_device "
            "FROM flr_sample AS s "
            "LEFT JOIN flr_sample_probe AS sp ON sp.sample_id = s.sample_id "
            "LEFT JOIN flr_sample_users AS u ON u.user_id = s.measured_by_user_id "
            "LEFT JOIN flr_sample_devices AS d ON d.device_id = s.measured_by_device_id "
            "WHERE s.deleted_at IS NULL "
            "GROUP BY s.sample_id "
            "ORDER BY s.sample_id"
        ).fetchall()

    def search_samples(self, query: str, limit: int = 50):
        """Return samples whose ID or display fields contain ``query``.

        Parameters
        ----------
        query : str
            Search substring.
        limit : int
            Maximum number of rows to return.

        Returns
        -------
        list of sqlite3.Row
            Matching sample rows.
        """
        pattern = f"%{query}%" if query else "%"
        return self.conn.execute(
            "SELECT s.sample_id, s.sample_uuid, s.description, s.details, "
            "s.num_of_probes, s.solvent_phase, s.sample_condition_id, "
            "s.entity_assembly_id, s.project_id, s.measured_by_user_id, "
            "s.measured_by_device_id, s.measured_at, "
            "COUNT(sp.sample_probe_id) AS mapped_probe_count, "
            "u.display_name AS measured_by_user, "
            "d.name AS measured_by_device "
            "FROM flr_sample AS s "
            "LEFT JOIN flr_sample_probe AS sp ON sp.sample_id = s.sample_id "
            "LEFT JOIN flr_sample_users AS u ON u.user_id = s.measured_by_user_id "
            "LEFT JOIN flr_sample_devices AS d ON d.device_id = s.measured_by_device_id "
            "WHERE s.deleted_at IS NULL "
            "AND (s.sample_id LIKE ? OR s.description LIKE ? OR s.details LIKE ?) "
            "GROUP BY s.sample_id "
            "ORDER BY s.sample_id "
            "LIMIT ?",
            (pattern, pattern, pattern, limit),
        ).fetchall()

    def sample_exists(self, sample_id: str) -> bool:
        """Return whether an active canonical sample exists.

        Parameters
        ----------
        sample_id : str
            Sample identifier to check.

        Returns
        -------
        bool
            ``True`` when the sample exists in ``flr_sample`` and is active.

        """
        if not sample_id:
            return False
        row = self.conn.execute(
            """SELECT 1 FROM flr_sample
               WHERE sample_id = ? AND deleted_at IS NULL""",
            (sample_id,),
        ).fetchone()
        return row is not None

    def link_artifact_to_sample(self, artifact_id: str, sample_id: str) -> None:
        """Create an idempotent measured-sample edge from artifact to sample.

        Parameters
        ----------
        artifact_id : str
            Artifact identifier to link.
        sample_id : str
            Canonical sample identifier.

        Returns
        -------
        None
            The edge is created when missing and left unchanged when present.

        """
        if not artifact_id or not sample_id:
            return
        existing = self.conn.execute(
            """SELECT edge_id FROM mfdb_edge
               WHERE source_node_type = 'artifact'
                 AND source_node_id = ?
                 AND target_node_type = 'sample'
                 AND target_node_id = ?
                 AND relationship_type = 'measured_sample'
                 AND deleted_at IS NULL""",
            (artifact_id, sample_id),
        ).fetchone()
        if existing:
            return
        self.add_edge(
            source_node_type="artifact",
            source_node_id=artifact_id,
            target_node_type="sample",
            target_node_id=sample_id,
            relationship_type="measured_sample",
        )

    def set_object_sample_id(self, object_uuid: str, sample_id: str | None) -> None:
        """Store or remove a sample_id on an object-store file reference.

        Parameters
        ----------
        object_uuid : str
            Object-store UUID of the file.
        sample_id : str or None
            Sample ID to link to the file, or None to remove the link.
        """
        row = self.conn.execute(
            "SELECT metadata_json FROM mfdb_object WHERE object_uuid = ?",
            (object_uuid,),
        ).fetchone()
        if not row:
            return
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except Exception:
            metadata = {}
        if sample_id:
            metadata["sample_id"] = sample_id
        else:
            metadata.pop("sample_id", None)
        with self.conn:
            self.conn.execute(
                "UPDATE mfdb_object SET metadata_json = ? WHERE object_uuid = ?",
                (_json_dumps(metadata), object_uuid),
            )

    def get_sample(self, sample_id):
        # PRD-26 Task 2: schema-driven get-by-PK (was a hand SELECT of a column
        # subset returning a sqlite3.Row). Now returns a dict (or None) covering all
        # flr_sample columns — a superset of the former subset; callers index by key
        # (some already rely on dict `.get(...)`). include_deleted=True preserves the
        # former "return regardless of soft-delete" semantics (no deleted_at filter).
        return self.dao.get("flr_sample", sample_id, include_deleted=True)

    def add_sample(self, sample_id, uuid=None, description="", details="", num_of_probes=None, solvent_phase=None, sample_condition_id=None, entity_assembly_id=None, project_id=None, measured_by_user_id=None, measured_by_device_id=None, measured_at=None):
        if measured_by_user_id is None:
            from mfdb.security.session import configured_default_user_id
            measured_by_user_id = configured_default_user_id()
        import uuid as _uuid
        if uuid is None:
            existing = self.get_sample(sample_id)
            uuid = existing["sample_uuid"] if existing else str(_uuid.uuid4())
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_sample "
                "(sample_id, sample_uuid, description, details, num_of_probes, solvent_phase, "
                "sample_condition_id, entity_assembly_id, project_id, measured_by_user_id, "
                "measured_by_device_id, measured_at, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sample_id, uuid, description, details, num_of_probes, solvent_phase,
                 sample_condition_id, entity_assembly_id, project_id, measured_by_user_id,
                 measured_by_device_id, measured_at,
                 now, now, None)
            )

    def update_sample(self, sample_id, **kwargs):
        if not kwargs:
            return
        allowed = {"sample_uuid", "description", "details", "num_of_probes", "solvent_phase", "sample_condition_id", "entity_assembly_id"}
        cols, vals = [], []
        for key, value in kwargs.items():
            if key not in allowed:
                raise ValueError(f"Unsupported sample column: {key}")
            cols.append(f"{key} = ?")
            vals.append(value)
        vals.append(sample_id)
        with self.conn:
            self.conn.execute(f"UPDATE flr_sample SET {', '.join(cols)}, updated_at = ? WHERE sample_id = ?", vals[:-1] + [_utc_now(), sample_id])

    def delete_sample(self, sample_id):
        with self.conn:
            now = _utc_now()
            self.conn.execute("UPDATE flr_sample_probe SET deleted_at = ? WHERE sample_id = ?", (now, sample_id))
            self.conn.execute("UPDATE flr_sample SET deleted_at = ? WHERE sample_id = ?", (now, sample_id))

    def set_sample_key_value(self, sample_id: str, key: str, value: str, details: str | None = None):
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_sample_key_value (sample_id, key, value, details, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sample_id, key, value, details, now, now, None)
            )

    def clear_sample_key_values(self, sample_id: str) -> None:
        self.conn.execute(
            "DELETE FROM flr_sample_key_value WHERE sample_id = ?",
            (sample_id,),
        )

    def add_sample_condition(self, condition_id, ph=None, temperature=None, ionic_strength=None,
                              buffer_composition=None, details=None):
        """Add a sample condition record.

        Parameters
        ----------
        condition_id : str
            Unique condition identifier.
        ph : float, optional
            pH value.
        temperature : float, optional
            Temperature in Kelvin.
        ionic_strength : float, optional
            Ionic strength in M.
        buffer_composition : str, optional
            Buffer composition description.
        details : str, optional
            Additional details.

        """
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_sample_condition "
                "(condition_id, ph, temperature, ionic_strength, buffer_composition, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (condition_id, ph, temperature, ionic_strength, buffer_composition, details,
                 now, now, None)
            )

    def add_entity_assembly(self, assembly_id, description=None, details=None):
        """Add an entity assembly record.

        Parameters
        ----------
        assembly_id : str
            Unique assembly identifier.
        description : str, optional
            Assembly description.
        details : str, optional
            Additional details.

        """
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_entity_assembly "
                "(assembly_id, description, details, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (assembly_id, description, details, now, now, None)
            )

    def add_poly_probe_position(self, probe_id, entity_id, residue_number, asym_id="A",
                                  residue_name=None, description=None,
                                  atom_id=None, mutation_flag="no", modification_flag="no",
                                  auth_name=None):
        """Add a probe position on a polymer entity.

        Parameters
        ----------
        probe_id : int
            Probe identifier.
        entity_id : str
            Entity identifier.
        residue_number : int
            Residue sequence number.
        asym_id : str, optional
            Chain/asymmetry identifier. Default is "A".
        residue_name : str, optional
            Residue name (mon_id).
        description : str, optional
            Position description.
        atom_id : str, optional
            Attachment atom identifier (e.g. "CB", "C5").
        mutation_flag : str, optional
            Mutation flag: "yes" if residue was mutated for labeling. Default "no".
        modification_flag : str, optional
            Modification flag: "yes" if residue is chemically modified. Default "no".
        auth_name : str, optional
            Author-provided position name (e.g. "S131C").

        """
        with self.conn:
            now = _utc_now()
            cursor = self.conn.execute(
                "INSERT INTO flr_poly_probe_position "
                "(probe_id, entity_id, residue_number, asym_id, residue_name, description, "
                "atom_id, mutation_flag, modification_flag, auth_name, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (probe_id, entity_id, residue_number, asym_id, residue_name, description,
                 atom_id, mutation_flag, modification_flag, auth_name,
                 now, now, None)
            )
            return cursor.lastrowid

    def add_fret_forster_radius(self, forster_radius_id, sample_id, probe_id_1, probe_id_2,
                                  forster_radius, kappa_squared=None, refractive_index=None,
                                  details=None):
        """Add a Förster radius calculation for a FRET pair.

        Parameters
        ----------
        forster_radius_id : str
            Unique Förster radius identifier (stored in forster_radius_id column).
        sample_id : str
            Sample identifier (stored in sample_id column).
        probe_id_1 : int
            First probe identifier (donor, stored as donor_probe_id).
        probe_id_2 : int
            Second probe identifier (acceptor, stored as acceptor_probe_id).
        forster_radius : float
            Förster radius in nanometers.
        kappa_squared : float, optional
            Orientation factor κ². Default is 2/3.
        refractive_index : float, optional
            Refractive index of the medium. Default is 1.4 (stored as index_of_refraction).
        details : str, optional
            Additional details.

        """
        if kappa_squared is None:
            kappa_squared = 2.0 / 3.0
        if refractive_index is None:
            refractive_index = 1.4
        with self.conn:
            now = _utc_now()
            cursor = self.conn.execute(
                "INSERT INTO flr_fret_forster_radius "
                "(forster_radius_id, sample_id, donor_probe_id, acceptor_probe_id, forster_radius, "
                "reduced_forster_radius, kappa_squared, index_of_refraction, overlap_integral, "
                "details, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (forster_radius_id, sample_id, probe_id_1, probe_id_2, forster_radius,
                 None,  # reduced_forster_radius
                 kappa_squared, refractive_index, None,  # overlap_integral
                 details, now, now, None)
            )
            return cursor.lastrowid

    def get_sample_key_values(self, sample_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT key, value, details FROM flr_sample_key_value WHERE sample_id = ? AND deleted_at IS NULL",
            (sample_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_sample_full(self, sample_id: str) -> dict[str, Any] | None:
        row = self.get_sample(sample_id)
        if row is None:
            return None
        res = dict(row)
        res["key_values"] = self.get_sample_key_values(sample_id)
        return res

    def add_sample_probe(self, sample_id, probe_id, fluorophore_type="unspecified", description=None, poly_probe_position_id=None):
        with self.conn:
            now = _utc_now()
            cursor = self.conn.execute(
                "INSERT OR REPLACE INTO flr_sample_probe (sample_id, probe_id, fluorophore_type, description, poly_probe_position_id, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (sample_id, probe_id, fluorophore_type, description, poly_probe_position_id, now, now, None)
            )
            return cursor.lastrowid

    def get_sample_probe_mappings(self, sample_id=None, probe_id=None):
        filters = ["sp.deleted_at IS NULL", "p.deleted_at IS NULL"]
        params: list[Any] = []
        if sample_id is not None:
            filters.append("sp.sample_id = ?")
            params.append(sample_id)
        if probe_id is not None:
            filters.append("sp.probe_id = ?")
            params.append(probe_id)
        rows = self.conn.execute(
            "SELECT sp.*, p.chromophore_name FROM flr_sample_probe AS sp "
            "JOIN probes AS p ON p.probe_id = sp.probe_id "
            f"WHERE {' AND '.join(filters)}",
            tuple(params),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_sample_probes(self, sample_id):
        with self.conn:
            self.conn.execute("UPDATE flr_sample_probe SET deleted_at = ? WHERE sample_id = ?", (_utc_now(), sample_id))

    def register_sample(self, sample_id, **kwargs):
        return self.add_sample(sample_id, **kwargs)
    def lookup_sample_by_md5(self, content_md5: str) -> str | None:
        """Return the sample_id already linked to a file content MD5.

        Parameters
        ----------
        content_md5 : str
            MD5 hex digest of the file content.

        Returns
        -------
        str or None
            The sample_id associated with the file, or None if no sample was
            linked yet.
        """
        row = self.conn.execute(
            "SELECT metadata_json FROM mfdb_object WHERE content_md5 = ?",
            (content_md5,),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["metadata_json"] or "{}").get("sample_id")
        except Exception:
            return None
