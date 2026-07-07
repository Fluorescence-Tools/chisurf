"""Probe, spectra, and optical-property queries.

Fluorophore/probe reference data, emission/absorption spectra, Förster-radius
lookups, chemical descriptors, and probe-quality curation — extracted from the
MFDatabase god-class (PRD-26). Methods run on the shared ``self.conn``/``self.dao``
and resolve cross-concern calls via the MFDatabase MRO.
"""

from __future__ import annotations

from typing import Any

import sqlite3

import numpy as np

from mfdb.schema._sqlutil import _utc_now
from mfdb.store.database_resolver import _default_reference_spectra_path


class ProbeMixin:
    def get_probe_types(self):
        return self.conn.execute("SELECT * FROM probe_types ORDER BY type_id").fetchall()

    def add_probe_type(self, name, description=None, details=None):
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                """INSERT INTO probe_types
                   (type_name, display_name, created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(type_name) DO UPDATE SET
                       display_name = excluded.display_name,
                       updated_at = excluded.updated_at,
                       deleted_at = NULL""",
                (name, description or name, now, now, None)
            )
            row = self.conn.execute(
                "SELECT type_id FROM probe_types WHERE type_name = ?",
                (name,),
            ).fetchone()
            return int(row["type_id"])

    def get_probe_categories(self):
        return self.conn.execute("SELECT DISTINCT category FROM probes WHERE category IS NOT NULL AND deleted_at IS NULL ORDER BY category").fetchall()

    def get_probe(self, probe_id):
        return self.conn.execute("SELECT * FROM probes WHERE probe_id = ?", (probe_id,)).fetchone()

    def get_probe_by_uuid(self, uuid_str):
        # Fallback to chromophore_name if uuid column doesn't exist
        return self.conn.execute("SELECT * FROM probes WHERE chromophore_name = ?", (uuid_str,)).fetchone()

    def find_or_add_probe(self, name: str, category: str = "other", description: str | None = None,
                          reactive_probe_flag: str = "no", reactive_probe_name: str | None = None,
                          probe_origin: str = "extrinsic", probe_link_type: str = "covalent",
                          chromophore_center_atom: str | None = None) -> int:
        """Find an existing probe by name or create a new one.

        Parameters
        ----------
        name : str
            Probe/fluorophore name.
        category : str, optional
            Probe category. Default is "other".
        description : str, optional
            Probe description.
        reactive_probe_flag : str, optional
            "yes" if reactive form differs from chromophore. Default "no".
        reactive_probe_name : str, optional
            Name of reactive form, e.g. "Cy3B-maleimide".
        probe_origin : str, optional
            "extrinsic" or "intrinsic" (e.g. Trp). Default "extrinsic".
        probe_link_type : str, optional
            How probe attaches to biomolecule. Default "covalent".
        chromophore_center_atom : str, optional
            Atom name for AV simulation center.

        Returns
        -------
        int
            Probe identifier.

        """
        # Try to find existing probe
        existing = self.conn.execute(
            "SELECT probe_id FROM probes WHERE chromophore_name = ? AND deleted_at IS NULL",
            (name,)
        ).fetchone()
        if existing:
            # Update existing probe with new chemical fields if they differ
            existing_probe = self.conn.execute(
                "SELECT * FROM probes WHERE probe_id = ?",
                (existing["probe_id"],)
            ).fetchone()
            if (existing_probe and
                (existing_probe.get("reactive_probe_flag") != reactive_probe_flag or
                 existing_probe.get("reactive_probe_name") != reactive_probe_name or
                 existing_probe.get("probe_origin") != probe_origin or
                 existing_probe.get("probe_link_type") != probe_link_type or
                 existing_probe.get("chromophore_center_atom") != chromophore_center_atom)):
                now = _utc_now()
                with self.conn:
                    self.conn.execute(
                        """UPDATE probes SET reactive_probe_flag = ?, reactive_probe_name = ?,
                           probe_origin = ?, probe_link_type = ?, chromophore_center_atom = ?,
                           category = ?, description = ?, updated_at = ? WHERE probe_id = ?""",
                        (reactive_probe_flag, reactive_probe_name, probe_origin, probe_link_type,
                         chromophore_center_atom, category, description, now, existing["probe_id"])
                    )
            return existing["probe_id"]

        # Create new probe with all chemical fields
        now = _utc_now()
        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO probes (chromophore_name, category, description, 
                   reactive_probe_flag, reactive_probe_name, probe_origin, probe_link_type,
                   chromophore_center_atom, created_at, updated_at, deleted_at) """
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, category, description, reactive_probe_flag, reactive_probe_name,
                 probe_origin, probe_link_type, chromophore_center_atom, now, now, None)
            )
            return cursor.lastrowid

    def get_probes(self, category=None, probe_type_id=None, include_inactive=False):
        query = "SELECT * FROM probes WHERE 1=1 AND deleted_at IS NULL"
        params = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if probe_type_id:
            query += " AND type_id = ?"
            params.append(probe_type_id)
        if not include_inactive:
            pass
        query += " ORDER BY probe_id"
        return self.conn.execute(query, params).fetchall()

    def find_probes_by_cas(self, cas: str) -> list[dict]:
        """Return probes whose CAS registry number matches ``cas``.

        CAS numbers are stored as the canonical ``cas`` optical property. The
        match ignores surrounding whitespace so ``"71-43-2"`` and ``" 71-43-2 "``
        are equivalent.
        """
        norm = str(cas or "").strip()
        if not norm:
            return []
        opt_cols = [r[1] for r in self.conn.execute("PRAGMA table_info(optical_properties)").fetchall()]
        opt_key = "probe_id" if "probe_id" in opt_cols else "item_id"
        rows = self.conn.execute(
            f"""SELECT p.* FROM probes p
                JOIN optical_properties o ON o.{opt_key} = p.probe_id
                WHERE o.property_name = 'cas'
                  AND TRIM(o.property_value) = ?
                  AND p.deleted_at IS NULL AND o.deleted_at IS NULL
                ORDER BY p.chromophore_name""",
            (norm,),
        ).fetchall()
        return [dict(r) for r in rows]

    def add_probe(self, probe_id, uuid_str=None, name=None, category=None, probe_type_id=None, probe_origin=None, probe_link_type=None, fluorophore_type=None, reactive_probe_flag=None, is_active=1, description=None, details=None, **kwargs):
        import uuid as _uuid
        is_legacy = False
        if isinstance(probe_id, str) and (isinstance(uuid_str, int) or (uuid_str is None and "type_id" in kwargs)):
            is_legacy = True
        elif isinstance(probe_id, str) and uuid_str is not None:
            try:
                _uuid.UUID(str(uuid_str))
            except ValueError:
                if str(uuid_str).isdigit():
                    is_legacy = True

        if is_legacy:
            chromophore_name = probe_id
            type_id = int(uuid_str) if uuid_str is not None else kwargs.get("type_id")
            category = category or "other"
            description = description or ""
            row = self.conn.execute(
                "SELECT probe_id FROM probes WHERE chromophore_name = ? AND type_id = ?",
                (chromophore_name, type_id)
            ).fetchone()
            if row:
                return row["probe_id"]
            with self.conn:
                now = _utc_now()
                cursor = self.conn.execute(
                    "INSERT INTO probes (chromophore_name, type_id, category, description, reactive_probe_flag, probe_origin, probe_link_type, fluorophore_type, "
                    "created_at, updated_at, deleted_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (chromophore_name, type_id, category, description, reactive_probe_flag or "no", probe_origin or "extrinsic", probe_link_type or "covalent", fluorophore_type or "unspecified",
                     now, now, None,)
                )
                return cursor.lastrowid
        else:
            c_name = name or probe_id
            if isinstance(c_name, int):
                c_name = name or f"probe_{probe_id}"
            t_id = probe_type_id or uuid_str
            if isinstance(t_id, str) and t_id.isdigit():
                t_id = int(t_id)
            elif not isinstance(t_id, int):
                t_id = None

            cols = ["chromophore_name"]
            vals = [c_name]
            if isinstance(probe_id, int):
                cols.append("probe_id")
                vals.append(probe_id)
            if t_id is not None:
                cols.append("type_id")
                vals.append(t_id)
            if category is not None:
                cols.append("category")
                vals.append(category)
            if description is not None:
                cols.append("description")
                vals.append(description)
            if probe_origin is not None:
                cols.append("probe_origin")
                vals.append(probe_origin)
            if probe_link_type is not None:
                cols.append("probe_link_type")
                vals.append(probe_link_type)
            if fluorophore_type is not None:
                cols.append("fluorophore_type")
                vals.append(fluorophore_type)
            if reactive_probe_flag is not None:
                cols.append("reactive_probe_flag")
                vals.append(reactive_probe_flag)

            now = _utc_now()
            cols += ["created_at", "updated_at", "deleted_at"]
            vals += [now, now, None]
            placeholders = ", ".join(["?"] * len(cols))
            with self.conn:
                cursor = self.conn.execute(
                    f"INSERT OR REPLACE INTO probes ({', '.join(cols)}) VALUES ({placeholders})",
                    tuple(vals)
                )
                return cursor.lastrowid or probe_id

    def update_probe(self, probe_id, **kwargs):
        allowed = {"name", "category", "probe_type_id", "probe_origin", "probe_link_type", "fluorophore_type", "reactive_probe_flag", "is_active", "description", "details"}
        if not kwargs:
            return
        cols, vals = [], []
        for key, value in kwargs.items():
            if key not in allowed:
                raise ValueError(f"Unsupported probe column: {key}")
            cols.append(f"{key} = ?")
            vals.append(value)
        vals.append(probe_id)
        with self.conn:
            self.conn.execute(f"UPDATE probes SET {', '.join(cols)}, updated_at = ? WHERE probe_id = ?", vals[:-1] + [_utc_now(), probe_id])

    def delete_probe(self, probe_id):
        with self.conn:
            self.dao.soft_delete("probes", probe_id, pk_column="probe_id", deleted_at=_utc_now())

    # -- probe verification / approval (PRD-06 Tasks 8) --

    def _set_probe_column(self, probe_id: int, column: str, value: str | None) -> None:
        now = _utc_now()
        with self.conn:
            self.conn.execute(
                f"UPDATE probes SET {column} = ?, updated_at = ? WHERE probe_id = ?",
                (value, now, probe_id),
            )

    def approve_probe(
        self, probe_id: int, verified_by: str | None = None
    ) -> None:
        """Mark a probe as approved.

        Parameters
        ----------
        probe_id : int
            Probe identifier.
        verified_by : str, optional
            User or system that approved the probe.
        """
        now = _utc_now()
        with self.conn:
            self.conn.execute(
                """UPDATE probes SET verification_status = 'approved',
                   is_curated = 1, verified_by = ?, verified_at = ?,
                   updated_at = ? WHERE probe_id = ?""",
                (verified_by, now, now, probe_id),
            )

    def reject_probe(
        self, probe_id: int, verified_by: str | None = None
    ) -> None:
        """Mark a probe as rejected."""
        now = _utc_now()
        with self.conn:
            self.conn.execute(
                """UPDATE probes SET verification_status = 'rejected',
                   verified_by = ?, verified_at = ?, updated_at = ?
                   WHERE probe_id = ?""",
                (verified_by, now, now, probe_id),
            )

    def mark_probe_under_review(self, probe_id: int) -> None:
        """Mark a probe as under review."""
        self._set_probe_column(probe_id, "verification_status", "under_review")

    def set_probe_quality(
        self, probe_id: int, quality: str
    ) -> None:
        """Set the quality grade for a probe.

        Parameters
        ----------
        probe_id : int
            Probe identifier.
        quality : str
            Quality grade: 'unknown', 'low', 'medium', or 'high'.
        """
        if quality not in ("unknown", "low", "medium", "high"):
            raise ValueError(f"Invalid quality grade: {quality!r}")
        self._set_probe_column(probe_id, "quality", quality)

    def get_probes_approved_only(
        self, category: str | None = None
    ) -> list[dict]:
        """Return approved probes suitable for downstream consumers.

        Parameters
        ----------
        category : str, optional
            Filter by probe category (e.g. 'organic_dye').

        Returns
        -------
        list of dict
            Approved probe rows.
        """
        query = "SELECT * FROM probes WHERE deleted_at IS NULL AND verification_status = 'approved'"
        params: list = []
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY chromophore_name"
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # -- import reference set from spectra.db (PRD-06 Task 7.2) --

    @staticmethod
    def _derive_probe_source(origin: str, name: str, prop_map: dict) -> tuple[str, str]:
        """Map a scraped probe to a real (source, source_ref).

        The reference ``spectra.db`` has no ``source`` column, but the scraper
        recorded provenance as optical properties — ``Origin`` (e.g.
        ``"FPbase (Protein)"``, ``"Chroma (Fluorochrome)"``, ``"Thorlabs (…)"``),
        ``fpbase_slug`` and ``PhotochemCAD Index``. Derive the canonical source
        from those (falling back to the name prefix, then ``spectra_db``).
        """
        o = (origin or "").lower()
        if "fpbase" in o:
            return "fpbase", str(prop_map.get("fpbase_slug", "") or "")
        if "chroma" in o:
            return "chroma", ""
        if "thorlabs" in o:
            return "thorlabs", ""
        if "omega" in o:
            return "omega", ""
        if "photochemcad" in o or prop_map.get("PhotochemCAD Index"):
            return "photochemcad", str(prop_map.get("PhotochemCAD Index", "") or "")
        if "atto" in o or name.upper().startswith("ATTO"):
            return "atto", ""
        return "spectra_db", ""

    @staticmethod
    def _derive_component_category(origin: str, type_name: str, name: str) -> str:
        """Normalize a scraped probe to an optical-component category.

        Drives the "Spectra" admin radio tabs. Derived from the scraped
        ``Origin`` (e.g. ``"Chroma (Chroma Emission Filter)"``) and probe
        ``type_name`` (e.g. ``"thorlabs_bandpass"``, ``"detector"``).
        """
        text = f"{origin} {type_name}".lower()
        if any(k in text for k in ("dichroic", "beamsplitter", "mirror")):
            return "dichroic"
        if "detector" in text or "apd" in text or "responsivity" in text:
            return "detector"
        if "light source" in text or "lightsource" in text:
            return "light_source"
        if "filter" in text or any(
            k in text for k in (
                "bandpass", "longpass", "shortpass", "notch", "_nd", "astronomy",
                "machine_vision", "tristimulus", "emission", "excitation",
            )
        ):
            return "filter"
        if any(
            k in text for k in (
                "fluorochrome", "protein", "organic dye", "organic_dye", "fpbase",
                "atto", "photochemcad", "fluorophore", "dye",
            )
        ):
            return "fluorophore"
        return "other"

    def purge_reference_probes(self) -> dict[str, int]:
        """Hard-delete all probes and their spectra / optical properties / images.

        Clears the optical-component reference set so it can be rebuilt cleanly
        from a freshly scraped ``spectra.db``. Foreign keys are checked first:
        if any sample / FRET / reagent row references a probe this raises rather
        than orphaning user data — only the self-contained probe tables
        (probes + spectra + optical_properties + images) are removed.
        """
        guard_tables = [
            ("flr_sample_probe", "probe_id"),
            ("flr_poly_probe_position", "probe_id"),
            ("flr_fret_forster_radius", "donor_probe_id"),
            ("flr_fret_forster_radius", "acceptor_probe_id"),
            ("flr_fret_distance_restraint", "probe_id_1"),
            ("flr_fret_distance_restraint", "probe_id_2"),
            ("mfdb_reagent_lot", "probe_id"),
        ]
        existing = {r["name"] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        for table, col in guard_tables:
            if table not in existing:
                continue
            n = self.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {col} IS NOT NULL"
            ).fetchone()[0]
            if n:
                raise RuntimeError(
                    f"Refusing to purge probes: {n} row(s) in {table}.{col} "
                    f"reference probes. Remove that data first."
                )
        counts = {}
        with self.conn:
            for table in ("spectra", "optical_properties", "images"):
                if table in existing:
                    counts[table] = self.conn.execute(f"DELETE FROM {table}").rowcount
            counts["probes"] = self.conn.execute("DELETE FROM probes").rowcount
        return counts

    def import_reference_set(
        self,
        source_path: str | None = None,
        *,
        mark_verified: bool = False,
        replace: bool = False,
    ) -> dict[str, int]:
        """Import fluorophore reference data from the scraped spectra.db.

        Copies probes, spectra, and optical properties from the ``_dev``
        spectra.db into this MFDB, carrying the scraper-assigned category and
        provenance through, then de-duplicates.

        Parameters
        ----------
        source_path : str, optional
            Path to the source ``spectra.db``. Defaults to the ``_dev``
            plugin-local database.
        mark_verified : bool, default=False
            If True, stamp imported probes as approved.
        replace : bool, default=False
            If True, hard-delete the existing reference probes (and their
            spectra / optical properties / images) first via
            :meth:`purge_reference_probes`, so the set is rebuilt cleanly
            instead of merged into the messy existing data.

        Returns
        -------
        dict
            Counts of imported probes, spectra, and optical properties
            (plus ``purged`` when ``replace=True``).
        """
        purged = None
        if replace:
            purged = self.purge_reference_probes()
        if source_path is None:
            source_path = _default_reference_spectra_path()
        source = sqlite3.connect(str(source_path))
        source.row_factory = sqlite3.Row
        try:
            # Fetch source probe types and build a map to MFDB type_ids
            src_type_rows = source.execute(
                "SELECT type_id, type_name FROM probe_types"
            ).fetchall()
            src_type_map = {r["type_id"]: r["type_name"] for r in src_type_rows}

            # Fetch MFDB probe types, creating missing ones on the fly
            mfdb_types = {
                r["type_name"]: r["type_id"]
                for r in self.conn.execute("SELECT type_id, type_name FROM probe_types").fetchall()
            }

            def _resolve_mfdb_type(source_type_id: int) -> int:
                """Map a source type_id to the corresponding MFDB type_id."""
                src_type_name = src_type_map.get(source_type_id, "organic_dye")
                # Strip path prefix from old-style type names (e.g. "E:\\dev\\...atto" → "atto")
                short_name = src_type_name.split("\\")[-1].split("/")[-1].lower()
                # Map source categories to MFDB canonical type names
                mfdb_type_name_map = {
                    "atto": "organic_dye",
                    "fluorophore": "organic_dye",
                    "photochemcad_common_compounds": "organic_dye",
                    "fpbase": "organic_dye",
                    "chroma_fluorochrome": "organic_dye",
                }
                canonical = mfdb_type_name_map.get(short_name, short_name)
                if canonical not in mfdb_types:
                    # Create the type if it doesn't exist
                    self.conn.execute(
                        "INSERT OR IGNORE INTO probe_types (type_name, display_name) VALUES (?, ?)",
                        (canonical, canonical.replace("_", " ").title()),
                    )
                    mfdb_types[canonical] = self.conn.execute(
                        "SELECT type_id FROM probe_types WHERE type_name = ?", (canonical,)
                    ).fetchone()["type_id"]
                return mfdb_types[canonical]

            # Fetch ALL source probes — fluorophores AND optical components
            # (filters/dichroics/detectors/light sources). The component type is
            # normalized into ``category`` below so the admin can manage them all.
            src_probes = source.execute(
                "SELECT * FROM probes WHERE deleted_at IS NULL"
            ).fetchall()

            # Check if probe_id or item_id is used in optical_properties and spectra of the source
            opt_cols = [row[1] for row in source.execute("PRAGMA table_info(optical_properties)").fetchall()]
            opt_key = "probe_id" if "probe_id" in opt_cols else "item_id"

            spec_cols = [row[1] for row in source.execute("PRAGMA table_info(spectra)").fetchall()]
            spec_key = "probe_id" if "probe_id" in spec_cols else "item_id"

            # The probe-name column differs between the canonical spectra.db
            # (``chromophore_name``) and the minimal test schema (``name``).
            probe_cols = [row[1] for row in source.execute("PRAGMA table_info(probes)").fetchall()]
            name_col = "chromophore_name" if "chromophore_name" in probe_cols else "name"

            imported_probes = 0
            imported_spectra = 0
            imported_props = 0
            skipped = 0

            now = _utc_now()
            verification = "approved" if mark_verified else "unverified"

            for src_p in src_probes:
                name = str(src_p[name_col] or "").strip()
                if not name:
                    skipped += 1
                    continue

                # Load this probe's optical properties first (keyed by probe_id,
                # which is always populated — item_id is NULL for many rows). The
                # scraped provenance lives here (Origin / fpbase_slug / …).
                src_props = source.execute(
                    f"SELECT * FROM optical_properties WHERE {opt_key} = ? AND deleted_at IS NULL",
                    (int(src_p["probe_id"]),),
                ).fetchall()
                prop_map = {
                    str(p["property_name"] or "").strip(): str(p["property_value"] or "").strip()
                    for p in src_props
                }
                derived_source, source_ref = self._derive_probe_source(
                    prop_map.get("Origin", ""), name, prop_map
                )
                # Prefer the provenance the scraper already recorded on the
                # staging row; fall back to the Origin-derived values.
                if "source" in probe_cols and str(src_p["source"] or "").strip():
                    derived_source = str(src_p["source"]).strip()
                if "source_ref" in probe_cols and str(src_p["source_ref"] or "").strip():
                    source_ref = str(src_p["source_ref"]).strip()
                # Prefer the category the scraper already assigned (the canonical
                # ingestion contract sets protein / organic_dye / filter / dichroic
                # / detector / light_source). Only fall back to deriving it from
                # the Origin/type when the staging row has none — so the staging
                # DB and the live MFDB stay aligned.
                src_category = ""
                if "category" in probe_cols:
                    src_category = str(src_p["category"] or "").strip().lower()
                src_type_name = src_type_map.get(int(src_p["type_id"]), "")
                if src_category and src_category != "other":
                    component_category = src_category
                else:
                    component_category = self._derive_component_category(
                        prop_map.get("Origin", ""), src_type_name, name
                    )

                mfdb_type_id = _resolve_mfdb_type(int(src_p["type_id"]))

                with self.conn:
                    # Normalize name: "ATTO-647N" -> "ATTO 647N"
                    chromophore_name = name.replace("-", " ").replace("_", " ")

                    existing = self.conn.execute(
                        "SELECT probe_id FROM probes WHERE chromophore_name = ? AND deleted_at IS NULL",
                        (chromophore_name,),
                    ).fetchone()
                    if existing:
                        probe_id = int(existing["probe_id"])
                        # Update the category, source, etc. on import
                        self.conn.execute(
                            "UPDATE probes SET category = ?, source = ?, source_ref = ?, type_id = ?, updated_at = ? WHERE probe_id = ?",
                            (
                                component_category,
                                derived_source,
                                source_ref,
                                mfdb_type_id,
                                now,
                                probe_id,
                            ),
                        )
                    else:
                        cursor = self.conn.cursor()
                        cursor.execute(
                            """INSERT INTO probes (chromophore_name, type_id, category,
                               description, is_curated, quality_flag,
                               verification_status, verified_by, verified_at,
                               quality, source, source_ref, created_at, updated_at, deleted_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                chromophore_name,
                                mfdb_type_id,
                                component_category,
                                str(src_p["description"] or ""),
                                1 if mark_verified else 0,
                                1,
                                verification,
                                "admin" if mark_verified else None,
                                now if mark_verified else None,
                                "unknown",
                                derived_source,
                                source_ref,
                                now,
                                now,
                                None,
                            ),
                        )
                        probe_id = int(cursor.lastrowid)
                        imported_probes += 1

                # Copy optical properties (already fetched above)
                for prop in src_props:
                    pname = str(prop["property_name"] or "").strip()
                    pval = str(prop["property_value"] or "").strip()
                    if not pname or not pval:
                        continue
                    with self.conn:
                        self.dao.upsert("optical_properties", {
                            "probe_id": probe_id,
                            "property_name": pname,
                            "property_value": pval,
                            "unit": str(prop["unit"] or ""),
                            "details": str(prop["details"] or ""),
                            "deleted_at": None,
                        }, conflict=["probe_id", "property_name"])
                        imported_props += 1

                # Copy spectra (by probe_id — item_id is NULL for many rows)
                src_spectra = source.execute(
                    f"SELECT * FROM spectra WHERE {spec_key} = ? AND deleted_at IS NULL",
                    (int(src_p["probe_id"]),),
                ).fetchall()
                for spec in src_spectra:
                    stype = str(spec["spectrum_type"] or "").strip()
                    wl = spec["wavelengths"]
                    iv = spec["intensity_values"]
                    if not stype or not wl or not iv:
                        continue
                    with self.conn:
                        self.dao.upsert("spectra", {
                            "probe_id": probe_id,
                            "spectrum_type": stype,
                            "wavelengths": wl,
                            "intensity_values": iv,
                            "wavelength_unit": "nm",
                            "intensity_unit": "normalized",
                            "details": f"Imported from spectra_db (source probe_id={src_p['probe_id']})",
                            "deleted_at": None,
                        }, conflict=["probe_id", "spectrum_type"])
                        imported_spectra += 1

            # Consolidate duplicate probes and merge their spectra/properties
            consolidation = self.consolidate_probes()

            return {
                "probes": imported_probes,
                "spectra": imported_spectra,
                "optical_properties": imported_props,
                "skipped": skipped,
                "consolidated": consolidation,
                "purged": purged,
            }
        finally:
            source.close()

    def consolidate_probes(self, aggressive: bool = False) -> dict[str, int]:
        """Merge duplicate probes, then delete the secondary copies.

        Two passes, simplest-first (per the maintainer's rule "prefer simple
        dedups over complex ones"):

        - **Simple (default).** Group **within a category** by a normalized name
          (case-folded, punctuation/space removed). This reliably collapses the
          same catalogue part scraped from two sources (e.g. a ``FB340-10``
          filter from both Thorlabs and 3DOptix) without ever merging across
          categories.
        - **Aggressive (opt-in, ``aggressive=True``).** Additionally strips
          reactive-group suffixes (``NHS ester``, ``maleimide``, …) so dye
          conjugates collapse onto the parent chromophore. This is **never**
          applied to fluorescent proteins (``category`` ``protein``/
          ``fluorophore``), which look alike by name but are distinct molecules.
        """
        # Discover all tables and columns that reference 'probes(probe_id)'
        referencing_cols = []
        for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            tname = r["name"]
            if tname.startswith("sqlite_"):
                continue
            fks = self.conn.execute(f"PRAGMA foreign_key_list({tname})").fetchall()
            for fk in fks:
                if fk["table"] == "probes":
                    referencing_cols.append((tname, fk["from"]))

        # Check column names dynamically to support both probe_id and item_id (for tests)
        spec_cols = [row[1] for row in self.conn.execute("PRAGMA table_info(spectra)").fetchall()]
        spec_key = "probe_id" if "probe_id" in spec_cols else "item_id"

        prop_cols = [row[1] for row in self.conn.execute("PRAGMA table_info(optical_properties)").fetchall()]
        prop_key = "probe_id" if "probe_id" in prop_cols else "item_id"

        rows = self.conn.execute(
            "SELECT probe_id, chromophore_name, category, type_id, description, "
            "source, source_ref, retrieved_at FROM probes WHERE deleted_at IS NULL"
        ).fetchall()

        # Fluorescent proteins are never fuzzy-merged: distinct proteins share
        # very similar names, so the aggressive suffix pass is suppressed for them.
        _PROTEIN_CATS = {"protein", "fluorophore"}
        _REACTIVE_SUFFIXES = (
            " nhs ester", " nhs-ester", " nhs",
            " maleimide",
            " carboxylic acid", " carboxylic",
            " succinimidyl ester", " succinimidyl",
            " sodium salt",
            " perchlorate",
            " tetrafluoroborate",
            " iodide",
            " chloride",
            " ester",
            " (nhs ester)",
            " (maleimide)",
        )

        def normalize_name(name: str, strip_suffixes: bool) -> str:
            if not name:
                return ""
            name_lower = " ".join(name.lower().split())
            if strip_suffixes:
                for suffix in _REACTIVE_SUFFIXES:
                    if name_lower.endswith(suffix):
                        name_lower = name_lower[: -len(suffix)]
            # Collapse to alphanumerics so "FB340-10" == "FB340 10".
            return "".join(c for c in name_lower if c.isalnum())

        # Group probes by (category, normalized name) so merges never cross a
        # category boundary. The aggressive suffix pass is skipped for proteins.
        groups: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            p = dict(r)
            category = (p.get("category") or "").lower()
            strip_suffixes = aggressive and category not in _PROTEIN_CATS
            norm = normalize_name(p["chromophore_name"], strip_suffixes)
            if norm:
                groups.setdefault((category, norm), []).append(p)

        merged_count = 0
        deleted_count = 0

        # Temporarily disable foreign keys constraint during database consolidation
        was_enforced = self.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        if was_enforced:
            self.conn.execute("PRAGMA foreign_keys=OFF")

        try:
            for (_category, _norm), p_list in groups.items():
                if len(p_list) <= 1:
                    continue

                # Choose the primary probe based on spectrum and property counts
                scored_probes = []
                for p in p_list:
                    probe_id = p["probe_id"]
                    n_specs = self.conn.execute(
                        f"SELECT COUNT(*) FROM spectra WHERE {spec_key} = ? AND deleted_at IS NULL",
                        (probe_id,)
                    ).fetchone()[0]
                    n_props = self.conn.execute(
                        f"SELECT COUNT(*) FROM optical_properties WHERE {prop_key} = ? AND deleted_at IS NULL",
                        (probe_id,)
                    ).fetchone()[0]
                    name_len = len(p["chromophore_name"])
                    # Score formula: prefer more spectra, properties, and shorter/cleaner names
                    score = (n_specs * 100) + (n_props * 10) - name_len
                    scored_probes.append((score, p))

                scored_probes.sort(key=lambda x: x[0], reverse=True)
                primary = scored_probes[0][1]
                secondaries = [x[1] for x in scored_probes[1:]]

                primary_id = primary["probe_id"]

                for sec in secondaries:
                    sec_id = sec["probe_id"]

                    # A. Merge spectra
                    sec_spectra = self.conn.execute(
                        f"SELECT id, spectrum_type FROM spectra WHERE {spec_key} = ? AND deleted_at IS NULL",
                        (sec_id,)
                    ).fetchall()
                    for spec in sec_spectra:
                        spec_id = spec["id"]
                        stype = spec["spectrum_type"]
                        existing_spec = self.conn.execute(
                            f"SELECT id FROM spectra WHERE {spec_key} = ? AND spectrum_type = ? AND deleted_at IS NULL",
                            (primary_id, stype)
                        ).fetchone()
                        if existing_spec is None:
                            self.conn.execute(
                                f"UPDATE spectra SET {spec_key} = ?, updated_at = ? WHERE id = ?",
                                (primary_id, _utc_now(), spec_id)
                            )
                        else:
                            self.conn.execute("DELETE FROM spectra WHERE id = ?", (spec_id,))

                    # B. Merge optical properties — union, keeping metadata. On a
                    # name conflict the secondary's value is dropped, UNLESS the
                    # primary's value is empty (then the secondary fills it in).
                    sec_props = self.conn.execute(
                        f"SELECT id, property_name, property_value FROM optical_properties "
                        f"WHERE {prop_key} = ? AND deleted_at IS NULL",
                        (sec_id,)
                    ).fetchall()
                    for prop in sec_props:
                        prop_id = prop["id"]
                        pname = prop["property_name"]
                        existing_prop = self.conn.execute(
                            f"SELECT id, property_value FROM optical_properties "
                            f"WHERE {prop_key} = ? AND property_name = ? AND deleted_at IS NULL",
                            (primary_id, pname)
                        ).fetchone()
                        if existing_prop is None:
                            self.conn.execute(
                                f"UPDATE optical_properties SET {prop_key} = ?, updated_at = ? WHERE id = ?",
                                (primary_id, _utc_now(), prop_id)
                            )
                        else:
                            primary_val = (existing_prop["property_value"] or "").strip()
                            sec_val = (prop["property_value"] or "").strip()
                            if not primary_val and sec_val:
                                self.conn.execute(
                                    "UPDATE optical_properties SET property_value = ?, updated_at = ? WHERE id = ?",
                                    (sec_val, _utc_now(), existing_prop["id"]),
                                )
                            self.conn.execute("DELETE FROM optical_properties WHERE id = ?", (prop_id,))

                    # C. Merge probe-level metadata so nothing is lost:
                    #  - description: fill if the primary has none;
                    #  - source: keep the UNION of contributing sources;
                    #  - source_ref / retrieved_at: fill if the primary has none;
                    #  - the secondary's distinct name is preserved as a synonym.
                    if not primary["description"] and sec["description"]:
                        self.conn.execute(
                            "UPDATE probes SET description = ?, updated_at = ? WHERE probe_id = ?",
                            (sec["description"], _utc_now(), primary_id),
                        )

                    cur = self.conn.execute(
                        "SELECT source, source_ref, retrieved_at FROM probes WHERE probe_id = ?",
                        (primary_id,),
                    ).fetchone()
                    src_parts = [s for s in str(cur["source"] or "").split(",") if s]
                    for s in str(sec["source"] or "").split(","):
                        if s and s not in src_parts:
                            src_parts.append(s)
                    self.conn.execute(
                        "UPDATE probes SET source = ?, "
                        "source_ref = COALESCE(NULLIF(source_ref, ''), ?), "
                        "retrieved_at = COALESCE(retrieved_at, ?), updated_at = ? "
                        "WHERE probe_id = ?",
                        (
                            ",".join(src_parts) or None,
                            sec["source_ref"],
                            sec["retrieved_at"],
                            _utc_now(),
                            primary_id,
                        ),
                    )
                    if (sec["chromophore_name"] or "") != (primary["chromophore_name"] or ""):
                        self.conn.execute(
                            f"INSERT OR IGNORE INTO optical_properties "
                            f"({prop_key}, property_name, property_value, created_at, updated_at) "
                            f"VALUES (?, 'synonym', ?, ?, ?)",
                            (primary_id, sec["chromophore_name"], _utc_now(), _utc_now()),
                        )

                    # E. Update/clean up all other referencing tables dynamically
                    for tname, colname in referencing_cols:
                        self.conn.execute(
                            f"UPDATE OR IGNORE {tname} SET {colname} = ? WHERE {colname} = ?",
                            (primary_id, sec_id)
                        )
                        self.conn.execute(
                            f"DELETE FROM {tname} WHERE {colname} = ?",
                            (sec_id,)
                        )

                    # D. Physically delete secondary probe
                    self.conn.execute("DELETE FROM probes WHERE probe_id = ?", (sec_id,))
                    deleted_count += 1

                merged_count += 1

            self.conn.commit()
        finally:
            if was_enforced:
                self.conn.execute("PRAGMA foreign_keys=ON")

        return {"merged_groups": merged_count, "deleted_probes": deleted_count}

    # -- Forster radius lookup (PRD-06 Task 5) --

    def lookup_forster_radius(
        self, donor_name: str, acceptor_name: str
    ) -> float | None:
        """Look up the Forster radius for a donor-acceptor pair.

        Reads from ``flr_fret_forster_radius`` via probe names.
        Only considers approved probes.

        Parameters
        ----------
        donor_name : str
            Donor probe name.
        acceptor_name : str
            Acceptor probe name.

        Returns
        -------
        float or None
            R0 in Angstrom, or None if not found or not approved.
        """
        row = self.conn.execute(
            """SELECT fr.forster_radius
               FROM flr_fret_forster_radius fr
               JOIN probes d ON d.probe_id = fr.donor_probe_id
               JOIN probes a ON a.probe_id = fr.acceptor_probe_id
               WHERE d.chromophore_name = ?
                 AND a.chromophore_name = ?
                 AND d.verification_status = 'approved'
                 AND a.verification_status = 'approved'
                 AND d.deleted_at IS NULL
                 AND a.deleted_at IS NULL""",
            (donor_name, acceptor_name),
        ).fetchone()
        return float(row[0]) if row else None

    def get_spectra_for_forster(
        self, probe_name: str
    ) -> dict | None:
        """Get absorption and emission spectra for a named probe.

        Only returns data for approved probes.

        Parameters
        ----------
        probe_name : str
            Probe name.

        Returns
        -------
        dict or None
            Dict with 'absorption' and 'emission' keys, each with
            'wavelengths' and 'intensity' arrays, or None if not found.
        """
        probe = self.conn.execute(
            "SELECT probe_id FROM probes WHERE chromophore_name = ? AND verification_status = 'approved' AND deleted_at IS NULL",
            (probe_name,),
        ).fetchone()
        if not probe:
            return None
        probe_id = int(probe["probe_id"])
        result = {}
        for stype in ("absorption", "emission"):
            row = self.conn.execute(
                "SELECT wavelengths, intensity_values FROM spectra WHERE probe_id = ? AND spectrum_type = ? AND deleted_at IS NULL",
                (probe_id, stype),
            ).fetchone()
            if row:
                result[stype] = {
                    "wavelengths": np.frombuffer(row["wavelengths"], dtype=np.float64),
                    "intensity": np.frombuffer(row["intensity_values"], dtype=np.float64),
                }
        if not result:
            return None
        return result

    # -- spectra --

    def get_spectra(self, probe_id=None, spectrum_type=None):
        query = "SELECT * FROM spectra WHERE 1=1 AND deleted_at IS NULL"
        params = []
        if probe_id is not None:
            query += " AND probe_id = ?"
            params.append(probe_id)
        if spectrum_type is not None:
            query += " AND spectrum_type = ?"
            params.append(spectrum_type)
        query += " ORDER BY spectrum_id"
        return self.conn.execute(query, params).fetchall()

    def get_spectrum(self, spectrum_id):
        return self.conn.execute("SELECT * FROM spectra WHERE spectrum_id = ?", (spectrum_id,)).fetchone()

    def add_spectrum(self, *args, **kwargs):
        probe_id = None
        spectrum_type = None
        wavelengths = None
        intensity_values = None
        wavelength_unit = kwargs.get("wavelength_unit", "nm")
        intensity_unit = kwargs.get("intensity_unit", "normalized")
        details = kwargs.get("details", None)

        if len(args) >= 4 and isinstance(args[1], str) and args[1] in ("emission", "excitation", "absorption"):
            probe_id = args[0]
            spectrum_type = args[1]
            wavelengths = args[2]
            intensity_values = args[3]
            if len(args) > 4:
                wavelength_unit = args[4]
            if len(args) > 5:
                intensity_unit = args[5]
            if len(args) > 6:
                details = args[6]
        elif len(args) >= 5 and isinstance(args[2], str) and args[2] in ("emission", "excitation", "absorption"):
            probe_id = args[1]
            spectrum_type = args[2]
            wavelengths = args[3]
            intensity_values = args[4]
            if len(args) > 5:
                details = args[5]
        else:
            probe_id = kwargs.get("probe_id", args[0] if len(args) > 0 else None)
            spectrum_type = kwargs.get("spectrum_type", args[1] if len(args) > 1 else None)
            wavelengths = kwargs.get("wavelengths", kwargs.get("wavelength_nm", args[2] if len(args) > 2 else None))
            intensity_values = kwargs.get("intensity_values", kwargs.get("value", args[3] if len(args) > 3 else None))

        if isinstance(wavelengths, (list, np.ndarray)):
            wavelengths = np.asarray(wavelengths, dtype=np.float64)
        if isinstance(intensity_values, (list, np.ndarray)):
            intensity_values = np.asarray(intensity_values, dtype=np.float64)

        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO spectra "
                "(probe_id, spectrum_type, wavelengths, intensity_values, wavelength_unit, intensity_unit, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (probe_id, spectrum_type, wavelengths, intensity_values, wavelength_unit, intensity_unit, details,
                 now, now, None)
            )

    def get_spectrum_record(self, probe_id, spectrum_type):
        row = self.conn.execute(
            "SELECT * FROM spectra WHERE probe_id = ? AND spectrum_type = ?",
            (probe_id, spectrum_type)
        ).fetchone()
        if row is None:
            return None
        res = dict(row)
        if res.get("wavelengths"):
            res["wavelengths"] = np.frombuffer(res["wavelengths"], dtype=np.float64)
        if res.get("intensity_values"):
            res["intensity_values"] = np.frombuffer(res["intensity_values"], dtype=np.float64)
        return res

    def delete_spectrum(self, spectrum_id):
        with self.conn:
            self.dao.soft_delete("spectra", spectrum_id, pk_column="spectrum_id", deleted_at=_utc_now())
