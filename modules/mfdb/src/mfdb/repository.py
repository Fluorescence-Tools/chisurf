import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from mfdb import schema
from mfdb._sqlutil import (
    _json_dumps,
    _json_hash,
    _json_loads,
    _utc_now,
)
from mfdb.base import MFDBClientBase
from mfdb.database_resolver import resolve_database_path
from mfdb.graph import map_legacy_node_type
from mfdb.models import (
    DIRECTIONS,
    OPERATION_TYPES,
    PARAMETER_TYPES,
    RELATIONSHIP_TYPES,
    STATUS_VALUES,
    STORAGE_MODES,
    VALIDATION_STATUS_VALUES,
    validate_vocabulary,
)
from mfdb.queries.parameters import ParameterMixin
from mfdb.queries.studies import StudyMixin
from mfdb.transactions import transaction as _transaction

logger = logging.getLogger(__name__)


def _default_reference_spectra_path() -> Path:
    """Resolve the default fluorophore reference ``spectra.db`` path.

    Returns
    -------
    pathlib.Path
        Existing reference database path.
    """
    env_path = os.environ.get("MFDB_REFERENCE_SPECTRA_DB")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    package_root = Path(__file__).resolve().parent
    candidates.extend(
        [
            package_root / "data" / "spectra.db",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No default fluorophore reference spectra.db found; set MFDB_REFERENCE_SPECTRA_DB"
    )


def _validate_checksum(checksum: str | None, algorithm: str | None) -> None:
    """Validate checksum length and hexadecimal encoding for supported algorithms."""
    if checksum is None or algorithm is None:
        return
    checksum_text = str(checksum)
    algorithm_text = str(algorithm).lower()
    if not re.fullmatch(r"[0-9a-fA-F]+", checksum_text):
        return
    expected_lengths = {"md5": 32, "sha256": 64}
    expected = expected_lengths.get(algorithm_text)
    if expected is not None and len(checksum_text) != expected:
        raise ValueError(f"{algorithm} checksum must be {expected} hexadecimal characters")
    int(checksum_text, 16)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _exists(conn: sqlite3.Connection, table: str, column: str, value: Any) -> bool:
    row = conn.execute(f"SELECT 1 FROM {table} WHERE {column} = ?", (value,)).fetchone()
    return row is not None


_migrated_db_paths: set[str] = set()


class MFDatabase(ParameterMixin, StudyMixin, MFDBClientBase):

    _VALID_ENUMS = {
        # ``category`` is the canonical optical-component class used by the
        # mfdb-admin "Spectra" radio tabs. Fluorophores keep the finer
        # ``organic_dye``/``protein``/… distinction; optical components use the
        # coarse ``filter``/``dichroic``/``detector``/``light_source`` classes.
        "category": [
            "fluorophore", "organic_dye", "protein", "nanoparticle", "quantum_dot",
            "filter", "dichroic", "detector", "light_source", "other",
        ],
        "probe_origin": ["extrinsic", "intrinsic", "other"],
        "probe_link_type": ["covalent", "non-covalent", "other"],
        "fluorophore_type": ["unspecified", "small_molecule", "protein_domain"],
        "reactive_probe_flag": ["yes", "no"],
    }

    _PROP_ALIASES = {
        "abs_max": ["abs_max", "absorption maximum", "λabs", "excitation max", "ex_max", "abs_peak", "λex"],
        "em_max": ["em_max", "emission maximum", "λfl", "emission max", "em_max", "em_peak", "λem"],
        "qy": ["qy", "fluorescence quantum yield", "ηfl", "quantum yield", "phi_acceptor", "phi", "qy_d", "phi_d"],
        "lifetime": ["lifetime", "fluorescence lifetime", "τfl", "tau", "tau_d", "tau_0"],
        "ext_coeff": ["ext_coeff", "molar extinction coefficient", "εmax", "extinction coefficient", "epsilon", "molar_ec"],
        # Optical-component (filter / dichroic / detector) properties — kept under
        # canonical keys so every scraper surfaces them identically in the GUI.
        "cut_on": ["cut_on", "cut-on", "cut on", "cut-on wavelength (nm)", "cut-on wavelength", "cuton"],
        "cut_off": ["cut_off", "cut-off", "cut off", "cut-off wavelength (nm)", "cut-off wavelength", "cutoff"],
        "center_wavelength": [
            "center_wavelength", "center wavelength", "center wavelength (nm)",
            "central wavelength", "cwl", "wavelength (nm)",
        ],
        "bandwidth": ["bandwidth", "bandwidth (nm)", "fwhm", "fwhm (nm)", "notch bandwidth (nm)", "bandwidth fwhm (nm)"],
        "optical_density": ["optical_density", "optical density", "od"],
        # CAS Registry Number — chemical identity for dyes/compounds.
        "cas": ["cas", "cas number", "cas_number", "cas nbr", "cas no", "cas#",
                "cas registry number", "casrn"],
    }

    def __init__(self, db_path: str | os.PathLike | None = None, readonly: bool = False, connection: sqlite3.Connection | None = None, enforce_foreign_keys: bool = True):
        import os as _os
        self._os = _os
        if db_path is None:
            db_path = resolve_database_path()
        self.db_path = str(db_path) if db_path == ":memory:" else (_os.path.abspath(str(db_path)) if db_path else None)
        self.readonly = readonly
        self.enforce_foreign_keys = enforce_foreign_keys
        self._conn: sqlite3.Connection | None = None
        self.migration_report = None
        if connection is not None:
            self._conn = connection
            self.readonly = readonly
        elif db_path is not None:
            self.connect()
            db_key = self.db_path
            if db_key == ":memory:" or db_key not in _migrated_db_paths:
                self.migration_report = schema.migrate_schema(self.conn)
                if not self.readonly and hasattr(schema, "_ensure_lifecycle_columns"):
                    schema._ensure_lifecycle_columns(self.conn)
                if db_key != ":memory:":
                    _migrated_db_paths.add(db_key)
            # Defensive column ensure — runs on every open so module
            # reloads and reconnect both repair DBs that silently missed
            # column additions (the old try/except OperationalError: pass).
            if not self.readonly and hasattr(schema, "_ensure_mfdb_setup_columns"):
                schema._ensure_mfdb_setup_columns(self.conn)

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    @property
    def dao(self) -> "DictionaryDao":
        """Dictionary-/schema-driven CRUD accessor (PRD-26 Task 1).

        Provides parameterised, schema-whitelisted ``insert/get/list/update/
        soft_delete`` over the live tables, built lazily from the connection
        after the schema is reconciled. CRUD-shaped repository methods are being
        migrated onto this to remove hand-written/f-string SQL.
        """
        dao = getattr(self, "_dao", None)
        if dao is None:
            from mfdb.dao import DictionaryDao

            dao = DictionaryDao.from_connection(self.conn)
            self._dao = dao
        return dao

    @property
    def lineage(self) -> "Lineage":
        """Lineage query API over the operation graph (PRD-21 Task 1).

        Artifact-centric ancestry/descent and the provenance-graph projection
        (``ancestors``/``descendants``/``what_used``/``provenance_graph``), built
        lazily from the connection. Call sites should use this instead of rolling
        their own ``mfdb_operation_artifact`` traversal.
        """
        lineage = getattr(self, "_lineage", None)
        if lineage is None:
            from mfdb.lineage import Lineage

            lineage = Lineage.from_connection(self.conn)
            self._lineage = lineage
        return lineage

    def connect(self):
        if self._conn is not None:
            return
        self._dao = None
        self._lineage = None
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        if self.enforce_foreign_keys:
            self._conn.execute("PRAGMA foreign_keys=ON")
        else:
            self._conn.execute("PRAGMA foreign_keys=OFF")
        self._conn.execute("PRAGMA busy_timeout=5000")

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def backup_database(self, target_path: str | os.PathLike) -> str:
        """Create a SQLite backup at ``target_path``.

        Parameters
        ----------
        target_path : str or os.PathLike
            Destination path for the copied SQLite database.

        Returns
        -------
        str
            Absolute path to the backup database.
        """
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        dest = sqlite3.connect(str(target))
        try:
            self.conn.backup(dest)
        finally:
            dest.close()
        return str(target)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _transaction(self):
        """Return a savepoint-backed transaction context for compound writes."""
        @contextmanager
        def _wrapper():
            with _transaction(self.conn):
                yield self
        return _wrapper()

    def transaction(self):
        """Transaction context manager for the database connection."""
        from contextlib import contextmanager

        from mfdb.transactions import transaction as _transaction
        @contextmanager
        def _wrapper():
            with _transaction(self.conn):
                yield self
        return _wrapper()

    def validate_extensible_vocab(self, field_name: str, value: str | None) -> None:
        """Validate that value is active in mfdb_vocabulary for field_name."""
        if value is None:
            return
        row = self.conn.execute(
            "SELECT is_active FROM mfdb_vocabulary WHERE field_name = ? AND value = ?",
            (field_name, value)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown extensible vocabulary value {value!r} for field {field_name!r}")
        if not row["is_active"]:
            raise ValueError(f"Inactive extensible vocabulary value {value!r} for field {field_name!r}")

    def register_vocabulary_value(
        self,
        field_name: str,
        value: str,
        display_name: str | None = None,
        description: str | None = None,
        is_builtin: bool = False,
        is_active: bool = True,
    ) -> None:
        """Register an extensible vocabulary value."""
        with self._transaction():
            now = _utc_now()
            self.conn.execute(
                """INSERT OR REPLACE INTO mfdb_vocabulary (
                    field_name, value, display_name, description, is_builtin, is_active,
                    created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (field_name, value, display_name or value, description, int(is_builtin), int(is_active),
                 now, now, None)
            )

    # -- schema / migration --

    def create_tables(self):
        schema.create_tables(self.conn)

    def migrate(self):
        schema.migrate(self.conn)

    def register_migration(self, version, description, applied_by):
        schema.register_migration(self.conn, version, description, applied_by)

    def get_schema_version(self):
        return schema.get_schema_version(self.conn)

    # -- legacy alias --
    _get_schema_version = get_schema_version

    # -- materialized views --

    def refresh_materialized_views(self):
        for view_name in schema.MATERIALIZED_VIEWS:
            self.conn.execute("DELETE FROM " + view_name)
            self.conn.execute("INSERT INTO " + view_name + " SELECT * FROM " + view_name + "__source")

    # -- external files --

    def add_external_file(self, file_path, file_format=None, content_type=None, file_size_bytes=None, md5=None, details=None):
        if not file_path:
            raise ValueError("file_path is required")
        file_uuid = str(uuid.uuid4())
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO ihm_external_files "
                "(reference_id, file_path, file_format, content_type, file_size_bytes, md5, uuid, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (None, str(file_path), file_format, content_type, file_size_bytes, md5, file_uuid, details,
                 now, now, None)
            )
            row = self.conn.execute("SELECT id FROM ihm_external_files WHERE uuid = ?", (file_uuid,)).fetchone()
            return int(row["id"]) if row else 0

    def get_external_file(self, file_id):
        return self.conn.execute("SELECT * FROM ihm_external_files WHERE id = ?", (file_id,)).fetchone()

    def resolve_external_path(self, file_path: str) -> str:
        raw = str(file_path)
        if raw.startswith("~"):
            raw = self._os.path.expanduser(raw)
        if not self._os.path.isabs(raw) and self.db_path:
            raw = self._os.path.join(self._os.path.dirname(self.db_path), raw)
        return self._os.path.normpath(raw)

    # -- CiteULike / citations --

    def get_citations(self, citeulike_ids=None):
        if citeulike_ids is None:
            return self.conn.execute("SELECT * FROM citeulike ORDER BY authors, title").fetchall()
        placeholders = ",".join("?" for _ in citeulike_ids)
        return self.conn.execute(f"SELECT * FROM citeulike WHERE citeulike_id IN ({placeholders}) ORDER BY authors, title", citeulike_ids).fetchall()

    def add_citation(self, citeulike_id, title, authors, journal=None, year=None, volume=None, number=None, pages=None, doi=None, pmid=None, pmcid=None, details=None):
        if not citeulike_id:
            raise ValueError("citeulike_id is required")
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO citeulike "
                "(citeulike_id, title, authors, journal, year, volume, number, pages, doi, pmid, pmcid, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (citeulike_id, title, authors, journal, year, volume, number, pages, doi, pmid, pmcid, details,
                 now, now, None)
            )

    def delete_citation(self, citeulike_id):
        with self.conn:
            self.dao.soft_delete("citeulike", citeulike_id, pk_column="citeulike_id", deleted_at=_utc_now())

    # -- probes / spectra / entities --

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
                        self.conn.execute(
                            """INSERT OR REPLACE INTO optical_properties
                               (probe_id, property_name, property_value, unit, details,
                                created_at, updated_at, deleted_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                probe_id,
                                pname,
                                pval,
                                str(prop["unit"] or ""),
                                str(prop["details"] or ""),
                                now,
                                now,
                                None,
                            ),
                        )
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
                        self.conn.execute(
                            """INSERT OR REPLACE INTO spectra
                               (probe_id, spectrum_type, wavelengths, intensity_values,
                                wavelength_unit, intensity_unit, details,
                                created_at, updated_at, deleted_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                probe_id,
                                stype,
                                wl,
                                iv,
                                "nm",
                                "normalized",
                                f"Imported from spectra_db (source probe_id={src_p['probe_id']})",
                                now,
                                now,
                                None,
                            ),
                        )
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

    # -- entities --

    def get_entities(self):
        return self.conn.execute("SELECT * FROM entities WHERE deleted_at IS NULL ORDER BY entity_id").fetchall()

    def add_entity(self, entity_id, name, sequence=None, entity_type=None, organism=None, entity_source=None, details=None):
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO entities "
                "(entity_id, type, description, formula_weight, src_method, "
                "number_of_molecules, common_name, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entity_id, entity_type or "polymer", details or name, None, None, 1, name,
                 now, now, None)
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

    # -- mfdb operation management --

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

    # -- mfdb artifact management --

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

    # -- artifacts linked to operations --

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

    # -- provenance edges (mfdb) --

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

    # -- legacy fdb provenance for graph traversal (read-only) --

    def get_downstream_dependencies(
        self, node_type: str, node_id: str
    ) -> list[dict[str, Any]]:
        if not node_type or not node_id:
            return []
        from mfdb.graph import traverse_canonical_graph
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
        from mfdb.graph import traverse_canonical_graph
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
        from mfdb.graph import normalize_node_type, traverse_canonical_graph

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

    # -- artifact lineage (PRD-21 Task 1; over the operation graph) --

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
        from mfdb.compute_spec import get_compute_spec

        return get_compute_spec(self, artifact_id)

    # -- lifecycle state machine (PRD-12) --

    def get_state(self, entity_type: str, entity_id: str) -> str | None:
        """Return an entity's current lifecycle state, or ``None``.

        The current state is the latest non-deleted transition's ``to_state`` — the
        transition log is the source of truth (no separate mutable status flag).
        """
        row = self.conn.execute(
            "SELECT to_state FROM mfdb_state_transition "
            "WHERE entity_type = ? AND entity_id = ? AND deleted_at IS NULL "
            "ORDER BY created_at DESC, transition_id DESC LIMIT 1",
            (entity_type, entity_id),
        ).fetchone()
        return row[0] if row else None

    def get_state_history(
        self, entity_type: str, entity_id: str
    ) -> list[dict[str, Any]]:
        """Return an entity's transitions in chronological order (oldest first)."""
        rows = self.conn.execute(
            "SELECT transition_id, entity_type, entity_id, from_state, to_state, "
            "reason, operator_user_id, created_at FROM mfdb_state_transition "
            "WHERE entity_type = ? AND entity_id = ? AND deleted_at IS NULL "
            "ORDER BY created_at, transition_id",
            (entity_type, entity_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def _state_transition_allowed(
        self, entity_type: str, from_state: str | None, to_state: str
    ) -> bool:
        """Is ``(entity_type, from_state, to_state)`` a declared transition rule?

        A NULL ``from_state`` rule declares an allowed initial state.
        """
        row = self.conn.execute(
            "SELECT 1 FROM mfdb_state_transition_rule "
            "WHERE entity_type = ? AND to_state = ? AND deleted_at IS NULL "
            "AND ((from_state IS NULL AND ? IS NULL) OR from_state = ?) LIMIT 1",
            (entity_type, to_state, from_state, from_state),
        ).fetchone()
        return row is not None

    def transition_state(
        self,
        entity_type: str,
        entity_id: str,
        to_state: str,
        *,
        reason: str = "",
        operator_user_id: str | None = None,
    ) -> bool:
        """Move an entity to ``to_state``, recording the transition (PRD-12).

        Validates against ``mfdb_state_transition_rule`` (raises
        :class:`~mfdb.lifecycle.StateTransitionError` on an illegal jump,
        surfaced not swallowed), is an **idempotent no-op** when already in
        ``to_state`` (returns ``False``), and otherwise records a transition row and
        publishes PRD-21's ``state.changed`` event post-commit (best-effort; a
        subscriber can never break the transition). Returns ``True`` when a transition
        was recorded.
        """
        from mfdb.lifecycle import StateTransitionError

        current = self.get_state(entity_type, entity_id)
        if current == to_state:
            return False
        if not self._state_transition_allowed(entity_type, current, to_state):
            raise StateTransitionError(
                f"illegal {entity_type} transition {current!r} -> {to_state!r} "
                "(no matching mfdb_state_transition_rule)"
            )
        now = _utc_now()
        with self._transaction():
            next_id = (
                self.conn.execute(
                    "SELECT COALESCE(MAX(transition_id), 0) FROM mfdb_state_transition"
                ).fetchone()[0]
                + 1
            )
            self.conn.execute(
                "INSERT INTO mfdb_state_transition "
                "(transition_id, entity_type, entity_id, from_state, to_state, reason, "
                "operator_user_id, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    next_id, entity_type, entity_id, current, to_state,
                    reason or None, operator_user_id, now, now, None,
                ),
            )
            self.add_audit_log(
                action="transition",
                target_type=f"state:{entity_type}",
                target_id=entity_id,
                operator_user_id=operator_user_id,
                details={"from": current, "to": to_state, "reason": reason},
            )
        # Post-commit, best-effort event (PRD-21 Task 3); never breaks the transition.
        from mfdb.events import EVENT_STATE_CHANGED, publish

        publish(
            EVENT_STATE_CHANGED,
            entity_type=entity_type,
            entity_id=entity_id,
            from_state=current,
            to_state=to_state,
            reason=reason or "",
            operator_user_id=operator_user_id or "",
        )
        return True

    # -- protocols (PRD-14): named, versioned procedures --

    #: Allowed protocol categories (declared as the .dic enumeration on
    #: mfdb_protocol.category).
    PROTOCOL_CATEGORIES = ("measurement", "processing", "analysis")

    def create_protocol(
        self,
        name: str,
        category: str,
        *,
        operation_type: str | None = None,
        setup_id: str | None = None,
        description: str = "",
        is_public: bool = False,
        created_by_user_id: str | None = None,
    ) -> tuple[str, int]:
        """Create a protocol (or a new version of an existing name); append-only.

        Returns ``(protocol_id, version)``. Editing a protocol means calling this
        again with the same ``name`` — it never mutates an existing row; a new row
        with ``version = max(version for name) + 1`` is recorded, so operations keep
        the exact version they ran. Owner defaults to the active user.
        """
        if category not in self.PROTOCOL_CATEGORIES:
            raise ValueError(
                f"unknown protocol category {category!r}; "
                f"expected one of {self.PROTOCOL_CATEGORIES}"
            )
        if not name:
            raise ValueError("protocol name is required")
        if created_by_user_id is None:
            from mfdb.session import configured_default_user_id
            created_by_user_id = configured_default_user_id()
        now = _utc_now()
        with self._transaction():
            prev = self.conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM mfdb_protocol "
                "WHERE name = ? AND deleted_at IS NULL",
                (name,),
            ).fetchone()[0]
            version = int(prev) + 1
            protocol_id = str(uuid.uuid4())
            self.conn.execute(
                "INSERT INTO mfdb_protocol (protocol_id, name, version, category, "
                "description, operation_type, setup_id, created_by_user_id, is_public, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    protocol_id, name, version, category, description or None,
                    operation_type, setup_id, created_by_user_id,
                    1 if is_public else 0, now, now, None,
                ),
            )
            self.add_audit_log(
                action="create",
                target_type="protocol",
                target_id=protocol_id,
                operator_user_id=created_by_user_id,
                details={"name": name, "version": version, "category": category},
            )
        return protocol_id, version

    def get_protocol(
        self, name: str, version: int | str = "latest"
    ) -> dict[str, Any] | None:
        """Return a protocol by ``name`` and ``version`` (default the latest)."""
        if version == "latest":
            row = self.conn.execute(
                "SELECT * FROM mfdb_protocol WHERE name = ? AND deleted_at IS NULL "
                "ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM mfdb_protocol WHERE name = ? AND version = ? "
                "AND deleted_at IS NULL",
                (name, int(version)),
            ).fetchone()
        return dict(row) if row else None

    def get_protocol_by_id(self, protocol_id: str) -> dict[str, Any] | None:
        """Return a specific protocol version row by its ``protocol_id``."""
        row = self.conn.execute(
            "SELECT * FROM mfdb_protocol WHERE protocol_id = ? AND deleted_at IS NULL",
            (protocol_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_protocol_versions(self, name: str) -> list[dict[str, Any]]:
        """Return all versions of a protocol ``name``, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM mfdb_protocol WHERE name = ? AND deleted_at IS NULL "
            "ORDER BY version",
            (name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_protocols(
        self, scope: str = "all", owner_id: str | None = None
    ) -> list[dict[str, Any]]:
        """List the latest version of each protocol, scoped own/public/all.

        ``scope``: ``'own'`` (owned by ``owner_id``), ``'public'`` (``is_public=1``),
        or ``'all'`` (public + own). ``owner_id`` defaults to the active user.
        """
        if owner_id is None and scope in ("own", "all"):
            from mfdb.session import configured_default_user_id
            owner_id = configured_default_user_id()
        latest = (
            "version = (SELECT MAX(p2.version) FROM mfdb_protocol p2 "
            "WHERE p2.name = mfdb_protocol.name AND p2.deleted_at IS NULL)"
        )
        where = [f"deleted_at IS NULL", latest]
        params: list[Any] = []
        if scope == "own":
            where.append("created_by_user_id = ?")
            params.append(owner_id)
        elif scope == "public":
            where.append("is_public = 1")
        else:  # all
            where.append("(is_public = 1 OR created_by_user_id = ?)")
            params.append(owner_id)
        rows = self.conn.execute(
            f"SELECT * FROM mfdb_protocol WHERE {' AND '.join(where)} ORDER BY name",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_protocol_parameter_schema(
        self, protocol: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the declared parameter schema for a protocol (no forked stack).

        The schema is the protocol's ``operation_type`` schema from PRD-11
        (``mfdb_operation_parameter_def``) — ``{name: OperationParameterDef}`` — so a
        protocol pins a named procedure to an operation kind without duplicating the
        parameter declarations.
        """
        from mfdb.operation_parameters import get_operation_parameter_defs

        operation_type = (protocol or {}).get("operation_type") or ""
        if not operation_type:
            return {}
        return get_operation_parameter_defs(self.conn, operation_type)

    # -- mfdb setup / setup definitions --

    def add_setup(self, setup_id, name, description=None, setup_type=None, config_json=None, details=None):
        if not setup_id:
            raise ValueError("setup_id is required")
        configuration = config_json or {}
        if setup_type is not None:
            configuration["setup_type"] = setup_type
        if details is not None:
            configuration["details"] = details
        return self.save_setup(
            setup_id=setup_id,
            name=name,
            description=description,
            configuration=configuration,
        )

    def get_setups(self, setup_type=None):
        setups = self.list_setups()
        if setup_type is None:
            return setups
        return [
            setup
            for setup in setups
            if _json_loads(setup.get("configuration_json")).get("setup_type") == setup_type
        ]
    def delete_setup(self, setup_id):
        # PRD-26 Task 2: schema-driven soft-delete (was a hand UPDATE). The explicit
        # _utc_now() value keeps the stored deleted_at marker format identical.
        with self.conn:
            self.dao.soft_delete("mfdb_setup", setup_id, pk_column="setup_id", deleted_at=_utc_now())

    def list_detector_channels(self, setup_id: str) -> list[dict[str, Any]]:
        """List detector channel definitions for a setup.

        Parameters
        ----------
        setup_id : str
            Setup identifier.

        Returns
        -------
        list of dict
            Detector channel rows.
        """
        rows = self.conn.execute(
            "SELECT * FROM mfdb_setup_detector_channel "
            "WHERE setup_id = ? AND deleted_at IS NULL ORDER BY id",
            (setup_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_pie_windows(self, setup_id: str) -> list[dict[str, Any]]:
        """List PIE/micro-time window definitions for a setup.

        Parameters
        ----------
        setup_id : str
            Setup identifier.

        Returns
        -------
        list of dict
            PIE window rows.
        """
        rows = self.conn.execute(
            "SELECT * FROM mfdb_setup_pie_window "
            "WHERE setup_id = ? AND deleted_at IS NULL ORDER BY id",
            (setup_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- setup calibration --

    def add_setup_calibration(
        self,
        setup_id: str,
        channel_name: str,
        g_factor: float | None = None,
        l1: float | None = None,
        l2: float | None = None,
        g_factor_channels: list[int] | None = None,
        g_factor_calibration_id: str | None = None,
        calibrated_at: str | None = None,
        method: str | None = "manual",
        created_by_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Append a calibration snapshot for one detector channel.

        This is an append-only insert — existing rows are never updated.
        The corresponding ``mfdb_setup_detector_channel`` row is updated as
        a cache of the latest snapshot so that legacy readers continue to work.

        Parameters
        ----------
        setup_id : str
            Setup identifier.
        channel_name : str
            Detector channel name.
        g_factor : float or None, optional
            G-factor value.
        l1 : float or None, optional
            Leakage parameter l1.
        l2 : float or None, optional
            Leakage parameter l2.
        g_factor_channels : list of int or None, optional
            Channel indices used for G-factor calculation.
        g_factor_calibration_id : str or None, optional
            Reference to the MFDB calibration artifact.
        calibrated_at : str or None, optional
            ISO-8601 timestamp. Defaults to current UTC time.
        method : str or None, optional
            Calibration method (e.g. ``manual``, ``migrated``,
            ``jordi_g_factor``). Defaults to ``manual``.
        created_by_user_id : str or None, optional
            User creating this snapshot.

        Returns
        -------
        dict
            The inserted row as a dictionary.
        """
        now = calibrated_at or _utc_now()
        with self._transaction():
            cur = self.conn.execute(
                """INSERT INTO mfdb_setup_calibration
                    (setup_id, channel_name, g_factor, l1, l2,
                     g_factor_channels, g_factor_calibration_id,
                     calibrated_at, method, created_by_user_id,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    setup_id, channel_name, g_factor, l1, l2,
                    _json_dumps(g_factor_channels) if g_factor_channels is not None else None,
                    g_factor_calibration_id,
                    now, method, created_by_user_id,
                    now, now,
                ),
            )
            snapshot_id = cur.lastrowid

            # Update the detector channel cache row
            self.conn.execute(
                """UPDATE mfdb_setup_detector_channel SET
                    g_factor = ?, l1 = ?, l2 = ?,
                    g_factor_channels = ?, g_factor_calibration_id = ?,
                    updated_at = ?
                WHERE setup_id = ? AND name = ? AND deleted_at IS NULL""",
                (
                    g_factor, l1, l2,
                    _json_dumps(g_factor_channels) if g_factor_channels is not None else None,
                    g_factor_calibration_id,
                    now, setup_id, channel_name,
                ),
            )

            return dict(
                self.conn.execute(
                    "SELECT * FROM mfdb_setup_calibration WHERE id = ?",
                    (snapshot_id,),
                ).fetchone()
            )

    def list_setup_calibration_dates(
        self, setup_id: str
    ) -> list[str]:
        """Return distinct calibration timestamps for a setup, newest first.

        Parameters
        ----------
        setup_id : str
            Setup identifier.

        Returns
        -------
        list of str
            ISO-8601 timestamps.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT calibrated_at FROM mfdb_setup_calibration "
            "WHERE setup_id = ? ORDER BY calibrated_at DESC",
            (setup_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_setup_calibration(
        self,
        setup_id: str,
        calibrated_at: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return calibration snapshots for a setup at a given timestamp.

        Parameters
        ----------
        setup_id : str
            Setup identifier.
        calibrated_at : str or None, optional
            ISO-8601 timestamp. If ``None``, returns the latest snapshot
            per channel.

        Returns
        -------
        list of dict
            Calibration snapshot rows.
        """
        if calibrated_at:
            rows = self.conn.execute(
                "SELECT * FROM mfdb_setup_calibration "
                "WHERE setup_id = ? AND calibrated_at = ? "
                "ORDER BY channel_name",
                (setup_id, calibrated_at),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT sc.* FROM mfdb_setup_calibration sc
                    INNER JOIN (
                        SELECT channel_name, MAX(calibrated_at) AS latest
                        FROM mfdb_setup_calibration
                        WHERE setup_id = ?
                        GROUP BY channel_name
                    ) latest
                    ON sc.channel_name = latest.channel_name
                    AND sc.calibrated_at = latest.latest
                    WHERE sc.setup_id = ?
                    ORDER BY sc.channel_name""",
                (setup_id, setup_id),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- analysis runs --

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
    # -- audit log --

    def add_audit_entry(self, action, entity_type, entity_id, user_id=None, old_values=None, new_values=None, details=None):
        details_json = details or {}
        if isinstance(details_json, dict):
            details_json = details_json.copy()
            details_json.update({"old_values": old_values, "new_values": new_values})
        return self.add_audit_log(
            action=action,
            target_type=entity_type,
            target_id=entity_id,
            operator_user_id=user_id,
            details=details_json,
        )

    def get_audit_log(self, entity_type=None, entity_id=None, action=None, limit=100):
        return self.get_audit_logs(
            action=action,
            target_type=entity_type,
            target_id=entity_id,
            limit=limit,
        )

    # -- products / standards --

    def get_product_categories(self):
        return self.conn.execute("SELECT * FROM product_categories ORDER BY name").fetchall()

    def add_product_category(self, name, description=None, details=None):
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO product_categories (name, description, details, created_at, updated_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?)",
                (name, description, details, now, now, None)
            )

    def get_products(self, category_id=None, supplier_id=None):
        query = "SELECT * FROM products WHERE 1=1"
        params = []
        if category_id:
            query += " AND category_id = ?"
            params.append(category_id)
        if supplier_id:
            query += " AND supplier_id = ?"
            params.append(supplier_id)
        query += " ORDER BY product_id"
        return self.conn.execute(query, params).fetchall()

    def add_product(self, product_id, name, catalog_number=None, supplier_id=None, category_id=None, cas_number=None, description=None, details=None):
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO products "
                "(product_id, name, catalog_number, supplier_id, category_id, cas_number, description, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (product_id, name, catalog_number, supplier_id, category_id, cas_number, description, details,
                 now, now, None)
            )

    def get_standards(self):
        return self.conn.execute("SELECT * FROM standards ORDER BY name").fetchall()

    def add_standard(self, standard_id, name, probe_id=None, reference_id=None, certification_details=None, valid_until=None, description=None, details=None):
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO standards "
                "(standard_id, name, probe_id, reference_id, certification_details, valid_until, description, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (standard_id, name, probe_id, reference_id, certification_details, valid_until, description, details,
                 now, now, None)
            )

    # -- flr sample / experiment (read-only wrappers for Core API) --

    def add_photon_stream(self, analysis_id, file_path, file_format=None, content_type=None, stream_id=None, detector_id=None, description=None, details=None):
        external_id = self.add_external_file(file_path, file_format, content_type, details=details)
        if stream_id is None:
            stream_id = f"stream_{external_id}"
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_photon_stream "
                "(stream_id, analysis_id, external_file_id, detector_id, description, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (stream_id, analysis_id, external_id, detector_id, description, details,
                 now, now, None)
            )
        return stream_id

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

    def get_sample(self, sample_id):
        # PRD-26 Task 2: schema-driven get-by-PK (was a hand SELECT of a column
        # subset returning a sqlite3.Row). Now returns a dict (or None) covering all
        # flr_sample columns — a superset of the former subset; callers index by key
        # (some already rely on dict `.get(...)`). include_deleted=True preserves the
        # former "return regardless of soft-delete" semantics (no deleted_at filter).
        return self.dao.get("flr_sample", sample_id, include_deleted=True)

    def add_sample(self, sample_id, uuid=None, description="", details="", num_of_probes=None, solvent_phase=None, sample_condition_id=None, entity_assembly_id=None, project_id=None, measured_by_user_id=None, measured_by_device_id=None, measured_at=None):
        if measured_by_user_id is None:
            from mfdb.session import configured_default_user_id
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

    def set_experiment_key_value(self, experiment_id: str, key: str, value: str, details: str | None = None):
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_experiment_key_value (experiment_id, key, value, details, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (experiment_id, key, value, details, now, now, None)
            )

    def clear_experiment_key_values(self, experiment_id: str) -> None:
        self.conn.execute(
            "DELETE FROM flr_experiment_key_value WHERE experiment_id = ?",
            (experiment_id,),
        )

    def get_experiment_key_values(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT key, value, details FROM flr_experiment_key_value WHERE experiment_id = ? AND deleted_at IS NULL",
            (experiment_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_experiment_full(self, experiment_id: str) -> dict[str, Any] | None:
        row = self.get_experiment(experiment_id)
        if row is None:
            return None
        res = dict(row)
        data_rows = self.conn.execute(
            "SELECT * FROM flr_experiment_data WHERE experiment_id = ? AND deleted_at IS NULL",
            (experiment_id,)
        ).fetchall()
        res["data"] = [dict(r) for r in data_rows]
        res["key_values"] = self.get_experiment_key_values(experiment_id)
        return res

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

    def get_raw_data(self, raw_data_id: str) -> dict[str, Any] | None:
        return self.get_artifact(raw_data_id)

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

    def get_photon_streams(self, analysis_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT ps.*, ef.file_path, ef.file_format, ef.content_type, ef.file_size_bytes
               FROM flr_photon_stream ps
               LEFT JOIN ihm_external_files ef ON ef.id = ps.external_file_id
               WHERE ps.analysis_id = ? AND ps.deleted_at IS NULL
               ORDER BY ps.stream_id""",
            (analysis_id,),
        ).fetchall()

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

    def export_flr_cif(self, path, analysis_id: str | None = None, include_extension: bool = True):
        import io

        import ihm.format
        is_stream = isinstance(path, io.TextIOBase)

        if analysis_id is None:
            row = self.conn.execute(
                "SELECT analysis_id FROM flr_fret_analysis WHERE deleted_at IS NULL ORDER BY analysis_id LIMIT 1"
            ).fetchone()
            analysis_id = row["analysis_id"] if row else "analysis_1"

        analysis = dict(
            self.conn.execute(
                "SELECT * FROM flr_fret_analysis WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()
            or {}
        )
        sample_id = analysis.get("sample_id") or analysis_id

        def _row_dict(row):
            return dict(row) if row is not None else {}

        _row_dict(
            self.conn.execute(
                "SELECT * FROM flr_sample WHERE sample_id = ?", (sample_id,)
            ).fetchone()
        )
        probes = [
            dict(row)
            for row in self.conn.execute("SELECT * FROM probes WHERE deleted_at IS NULL ORDER BY probe_id").fetchall()
        ]
        positions = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM flr_poly_probe_position WHERE deleted_at IS NULL ORDER BY id"
            ).fetchall()
        ]
        sample_probes = [dict(row) for row in self.get_sample_probe_mappings(sample_id=sample_id)]
        if not sample_probes:
            first_probe = probes[0] if probes else None
            first_position = positions[0] if positions else None
            if first_probe is not None or first_position is not None:
                sample_probes = [
                    {
                        "sample_probe_id": 1,
                        "sample_id": sample_id,
                        "probe_id": first_probe.get("probe_id") if first_probe is not None else None,
                        "poly_probe_position_id": first_position.get("id") if first_position is not None else None,
                        "chromophore_name": first_probe.get("chromophore_name") if first_probe is not None else None,
                        "fluorophore_type": "unspecified",
                        "description": "ChiSurf legacy sample probe mapping",
                    }
                ]
        distances = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM flr_fret_distance_restraint WHERE analysis_id = ? AND deleted_at IS NULL ORDER BY id",
                (analysis_id,),
            ).fetchall()
        ]
        forster = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM flr_fret_forster_radius WHERE sample_id = ? AND deleted_at IS NULL ORDER BY id",
                (sample_id,)
            ).fetchall()
        ]
        metadata = self.get_analysis_metadata(analysis_id)
        streams = [dict(row) for row in self.get_photon_streams(analysis_id)]
        external_files = [
            dict(row)
            for row in self.conn.execute("SELECT * FROM ihm_external_files ORDER BY id").fetchall()
        ]
        properties = [
            dict(row)
            for row in self.conn.execute("SELECT * FROM optical_properties WHERE deleted_at IS NULL ORDER BY id").fetchall()
        ]
        spectra = [
            dict(row)
            for row in self.conn.execute("SELECT * FROM spectra WHERE deleted_at IS NULL ORDER BY id").fetchall()
        ]
        analysis_data = [dict(row) for row in self.get_analysis_data(analysis_id)]
        struct_refs = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM struct_ref WHERE deleted_at IS NULL ORDER BY ref_id"
            ).fetchall()
        ]
        struct_ref_seqs = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM struct_ref_seq WHERE deleted_at IS NULL ORDER BY align_id"
            ).fetchall()
        ]
        struct_ref_seq_difs = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM struct_ref_seq_dif WHERE deleted_at IS NULL ORDER BY pdbx_ordinal, seq_num"
            ).fetchall()
        ]

        def _array_to_text(blob, dtype=np.float64):
            if not blob:
                return ""
            try:
                return " ".join(f"{float(v):.8g}" for v in np.frombuffer(blob, dtype=dtype))
            except ValueError:
                pass
            try:
                values = json.loads(blob)
                return " ".join(f"{float(v):.8g}" for v in values)
            except (json.JSONDecodeError, TypeError, ValueError):
                return str(blob)

        def _write_content(writer):
            writer.start_block("chisurf_flr_export")

            with writer.loop(
                "_flr_probe_list",
                [
                    "probe_id",
                    "chromophore_name",
                    "reactive_probe_flag",
                    "reactive_probe_name",
                    "probe_origin",
                    "probe_link_type",
                    "fluorophore_type",
                    "chromophore_chem_descriptor_id",
                    "reactive_probe_chem_descriptor_id",
                    "chromophore_center_atom",
                    "details",
                ],
            ) as loop:
                for row in probes:
                    loop.write(
                        probe_id=row.get("probe_id"),
                        chromophore_name=row.get("chromophore_name"),
                        reactive_probe_flag=row.get("reactive_probe_flag"),
                        reactive_probe_name=row.get("reactive_probe_name"),
                        probe_origin=row.get("probe_origin"),
                        probe_link_type=row.get("probe_link_type"),
                        fluorophore_type=row.get("fluorophore_type"),
                        chromophore_chem_descriptor_id=row.get("chromophore_chem_descriptor_id"),
                        reactive_probe_chem_descriptor_id=row.get("reactive_probe_chem_descriptor_id"),
                        chromophore_center_atom=row.get("chromophore_center_atom"),
                        details=row.get("description"),
                    )

            with writer.loop(
                "_flr_poly_probe_position",
                ["id", "entity_id", "asym_id", "residue_number", "residue_name", "details"],
            ) as loop:
                for row in positions:
                    loop.write(
                        id=row.get("id"),
                        entity_id=row.get("entity_id"),
                        asym_id=row.get("asym_id"),
                        residue_number=row.get("residue_number"),
                        residue_name=row.get("residue_name"),
                        details=row.get("description"),
                    )

            with writer.loop(
                "_flr_sample_probe_details",
                [
                    "sample_probe_id",
                    "sample_id",
                    "probe_id",
                    "poly_probe_position_id",
                    "fluorophore_type",
                    "description",
                ],
            ) as loop:
                for row in sample_probes:
                    loop.write(
                        sample_probe_id=row.get("sample_probe_id"),
                        sample_id=row.get("sample_id"),
                        probe_id=row.get("probe_id"),
                        poly_probe_position_id=row.get("poly_probe_position_id"),
                        fluorophore_type=row.get("fluorophore_type"),
                        description=row.get("description"),
                    )

            with writer.loop(
                "_flr_fret_forster_radius",
                [
                    "id",
                    "donor_probe_id",
                    "acceptor_probe_id",
                    "forster_radius",
                    "forster_radius_error_plus",
                    "forster_radius_error_minus",
                    "kappa_squared_mode",
                    "refractive_index",
                    "citation_id",
                    "details",
                ],
            ) as loop:
                for row in forster:
                    loop.write(
                        id=row.get("id"),
                        donor_probe_id=row.get("donor_probe_id"),
                        acceptor_probe_id=row.get("acceptor_probe_id"),
                        forster_radius=row.get("forster_radius"),
                        forster_radius_error_plus=row.get("forster_radius_error_plus"),
                        forster_radius_error_minus=row.get("forster_radius_error_minus"),
                        kappa_squared_mode=row.get("kappa_squared_mode"),
                        refractive_index=row.get("refractive_index"),
                        citation_id=row.get("citation_id"),
                        details=row.get("details"),
                    )

            with writer.loop(
                "_flr_fret_analysis",
                [
                    "analysis_id",
                    "experiment_id",
                    "sample_id",
                    "type",
                    "method",
                    "sample_probe_id_1",
                    "sample_probe_id_2",
                    "forster_radius_id",
                    "dataset_list_id",
                    "external_file_id",
                    "software_id",
                    "details",
                ],
            ) as loop:
                loop.write(
                    analysis_id=analysis.get("analysis_id"),
                    experiment_id=analysis.get("experiment_id"),
                    sample_id=analysis.get("sample_id"),
                    type=analysis.get("type"),
                    method=analysis.get("method"),
                    sample_probe_id_1=analysis.get("sample_probe_id_1"),
                    sample_probe_id_2=analysis.get("sample_probe_id_2"),
                    forster_radius_id=analysis.get("forster_radius_id"),
                    dataset_list_id=analysis.get("dataset_list_id"),
                    external_file_id=analysis.get("external_file_id"),
                    software_id=analysis.get("software_id"),
                    details=analysis.get("details"),
                )

            with writer.loop(
                "_struct_ref",
                [
                    "ref_id",
                    "entity_id",
                    "db_name",
                    "db_code",
                    "pdbx_db_accession",
                    "pdbx_db_isoform",
                    "pdbx_seq_one_letter_code",
                    "organism",
                    "details",
                ],
            ) as loop:
                for row in struct_refs:
                    loop.write(
                        ref_id=row.get("ref_id"),
                        entity_id=row.get("entity_id"),
                        db_name=row.get("db_name"),
                        db_code=row.get("db_code"),
                        pdbx_db_accession=row.get("pdbx_db_accession"),
                        pdbx_db_isoform=row.get("pdbx_db_isoform"),
                        pdbx_seq_one_letter_code=row.get("pdbx_seq_one_letter_code"),
                        organism=row.get("organism"),
                        details=row.get("details"),
                    )

            with writer.loop(
                "_struct_ref_seq",
                [
                    "align_id",
                    "ref_id",
                    "seq_align_beg",
                    "seq_align_end",
                    "db_align_beg",
                    "db_align_end",
                    "pdbx_db_accession",
                    "details",
                ],
            ) as loop:
                for row in struct_ref_seqs:
                    loop.write(
                        align_id=row.get("align_id"),
                        ref_id=row.get("ref_id"),
                        seq_align_beg=row.get("seq_align_beg"),
                        seq_align_end=row.get("seq_align_end"),
                        db_align_beg=row.get("db_align_beg"),
                        db_align_end=row.get("db_align_end"),
                        pdbx_db_accession=row.get("pdbx_db_accession"),
                        details=row.get("details"),
                    )

            with writer.loop(
                "_struct_ref_seq_dif",
                [
                    "id",
                    "align_id",
                    "seq_num",
                    "mon_id",
                    "db_mon_id",
                    "details",
                    "pdbx_seq_db_name",
                    "pdbx_seq_db_accession_code",
                    "pdbx_ordinal",
                ],
            ) as loop:
                for row in struct_ref_seq_difs:
                    loop.write(
                        id=row.get("id"),
                        align_id=row.get("align_id"),
                        seq_num=row.get("seq_num"),
                        mon_id=row.get("mon_id"),
                        db_mon_id=row.get("db_mon_id"),
                        details=row.get("details"),
                        pdbx_seq_db_name=row.get("pdbx_seq_db_name"),
                        pdbx_seq_db_accession_code=row.get("pdbx_seq_db_accession_code"),
                        pdbx_ordinal=row.get("pdbx_ordinal"),
                    )

            sample_probe_by_probe = {
                row.get("probe_id"): row.get("sample_probe_id")
                for row in sample_probes
                if row.get("probe_id") is not None
            }
            with writer.loop(
                "_flr_fret_distance_restraint",
                [
                    "ordinal_id",
                    "id",
                    "group_id",
                    "sample_probe_id_1",
                    "sample_probe_id_2",
                    "state_id",
                    "analysis_id",
                    "distance",
                    "distance_error_plus",
                    "distance_error_minus",
                    "distance_type",
                    "population_fraction",
                    "peak_assignment_id",
                ],
            ) as loop:
                for i, row in enumerate(distances, 1):
                    loop.write(
                        ordinal_id=i,
                        id=row.get("id"),
                        group_id=1,
                        sample_probe_id_1=row.get("sample_probe_id_1")
                        or sample_probe_by_probe.get(row.get("probe_id_1")),
                        sample_probe_id_2=row.get("sample_probe_id_2")
                        or sample_probe_by_probe.get(row.get("probe_id_2")),
                        state_id=row.get("state_id"),
                        analysis_id=row.get("analysis_id") or analysis_id,
                        distance=row.get("distance"),
                        distance_error_plus=row.get("distance_error_plus"),
                        distance_error_minus=row.get("distance_error_minus"),
                        distance_type=row.get("distance_type"),
                        population_fraction=row.get("population_fraction"),
                        peak_assignment_id=row.get("peak_assignment_id"),
                    )

            with writer.loop(
                "_ihm_dataset_list", ["id", "data_type", "details", "database_hosted"]
            ) as loop:
                loop.write(
                    id=1,
                    data_type="analysis_metadata",
                    details="ChiSurf FLR analysis metadata",
                    database_hosted="no",
                )

            with writer.loop(
                "_ihm_external_files",
                [
                    "id",
                    "reference_id",
                    "file_path",
                    "file_format",
                    "content_type",
                    "file_size_bytes",
                    "md5",
                    "uuid",
                    "details",
                ],
            ) as loop:
                for external in external_files:
                    loop.write(
                        **{
                            k: external.get(k)
                            for k in [
                                "id",
                                "reference_id",
                                "file_path",
                                "file_format",
                                "content_type",
                                "file_size_bytes",
                                "md5",
                                "uuid",
                                "details",
                            ]
                        }
                    )

            if include_extension:
                with writer.loop(
                    "_chisurf_analysis_metadata", ["analysis_id", "key", "value"]
                ) as loop:
                    for key, value in sorted(metadata.items()):
                        loop.write(analysis_id=analysis_id, key=key, value=value)

                with writer.loop(
                    "_chisurf_probe_property",
                    ["probe_id", "property_name", "property_value", "unit", "details"],
                ) as loop:
                    for prop in properties:
                        loop.write(
                            **{
                                k: prop.get(k)
                                for k in [
                                    "probe_id",
                                    "property_name",
                                    "property_value",
                                    "unit",
                                    "details",
                                ]
                            }
                        )

                with writer.loop(
                    "_chisurf_probe_spectrum",
                    [
                        "probe_id",
                        "spectrum_type",
                        "wavelengths",
                        "intensity_values",
                        "wavelength_unit",
                        "intensity_unit",
                        "details",
                    ],
                ) as loop:
                    for spec in spectra:
                        loop.write(
                            probe_id=spec.get("probe_id"),
                            spectrum_type=spec.get("spectrum_type"),
                            wavelengths=_array_to_text(spec.get("wavelengths")),
                            intensity_values=_array_to_text(spec.get("intensity_values")),
                            wavelength_unit=spec.get("wavelength_unit") or "nm",
                            intensity_unit=spec.get("intensity_unit") or "normalized",
                            details=spec.get("details"),
                        )

                with writer.loop(
                    "_chisurf_photon_stream",
                    [
                        "stream_id",
                        "analysis_id",
                        "external_file_id",
                        "detector_id",
                        "description",
                        "details",
                    ],
                ) as loop:
                    for stream in streams:
                        loop.write(
                            **{
                                k: stream.get(k)
                                for k in [
                                    "stream_id",
                                    "analysis_id",
                                    "external_file_id",
                                    "detector_id",
                                    "description",
                                    "details",
                                ]
                            }
                        )

                with writer.loop(
                    "_chisurf_analysis_data",
                    [
                        "analysis_id",
                        "data_type",
                        "data_name",
                        "x_values",
                        "y_values",
                        "x_unit",
                        "y_unit",
                        "details",
                    ],
                ) as loop:
                    for data in analysis_data:
                        loop.write(
                            analysis_id=analysis_id,
                            data_type=data.get("data_type"),
                            data_name=data.get("data_name"),
                            x_values=_array_to_text(data.get("x_values")),
                            y_values=_array_to_text(data.get("y_values")),
                            x_unit=data.get("x_unit"),
                            y_unit=data.get("y_unit"),
                            details=data.get("details"),
                        )

        if is_stream:
            writer = ihm.format.CifWriter(path)
            _write_content(writer)
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as out:
                writer = ihm.format.CifWriter(out)
                _write_content(writer)
            return path

    def export_flr_cif_to_text(self, analysis_id: str | None = None, include_extension: bool = True) -> str:
        import io
        buffer = io.StringIO()
        self.export_flr_cif(buffer, analysis_id=analysis_id, include_extension=include_extension)
        return buffer.getvalue()

    def get_raw_data_references(self, experiment_id=None, data_type=None):
        query = "SELECT * FROM mfdb_artifact WHERE artifact_kind = 'raw_data' AND deleted_at IS NULL"
        params = []
        if experiment_id is not None:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        rows = self.conn.execute(query, params).fetchall()
        res = []
        for r in rows:
            d = dict(r)
            meta = _json_loads(d.get("metadata_json")) or {}
            if data_type is not None:
                if meta.get("data_type") != data_type:
                    continue
            res.append(d)
        return res

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
        import uuid
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

    # -- chem_descriptors / optical_properties / images --

    def get_chemical_descriptors(self, probe_id=None):
        return self.conn.execute(
            "SELECT * FROM chem_descriptors WHERE probe_id = ? ORDER BY descriptor_type, descriptor_id",
            (probe_id,)
        ).fetchall()

    def add_chemical_descriptor(self, probe_id, descriptor_type, value, unit=None, method=None, details=None):
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO chem_descriptors "
                "(probe_id, descriptor_type, value, unit, method, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (probe_id, descriptor_type, value, unit, method, details,
                 now, now, None)
            )

    def get_optical_properties(self, probe_id=None):
        if probe_id is not None:
            rows = self.conn.execute(
                "SELECT id, probe_id, property_name, property_value, unit, details FROM optical_properties WHERE probe_id = ? AND deleted_at IS NULL ORDER BY property_name, id",
                (probe_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, probe_id, property_name, property_value, unit, details FROM optical_properties WHERE deleted_at IS NULL ORDER BY property_name, id"
            ).fetchall()
        new_rows = []
        for r in rows:
            d = dict(r)
            d["property_id"] = d["id"]
            d["property_type"] = d["property_name"]
            d["value"] = d["property_value"]
            new_rows.append(d)
        return new_rows

    def add_optical_property(self, probe_id, property_type, value, unit=None, method=None, condition_json=None, details=None):
        extra = {}
        if method:
            extra["method"] = method
        if condition_json:
            extra["condition"] = condition_json
        details_str = details
        if extra:
            details_str = (details or "") + " " + _json_dumps(extra)
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO optical_properties "
                "(probe_id, property_name, property_value, unit, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (probe_id, property_type, value, unit, details_str,
                 now, now, None)
            )

    def get_images(self, probe_id=None):
        return self.conn.execute(
            "SELECT * FROM images WHERE probe_id = ? AND deleted_at IS NULL ORDER BY image_id",
            (probe_id,)
        ).fetchall()

    def add_image(self, probe_id, image_path, image_type, description=None, details=None):
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                "INSERT OR REPLACE INTO images "
                "(probe_id, image_path, image_type, description, details, "
                "created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (probe_id, image_path, image_type, description, details,
                 now, now, None)
            )

    # -- _decode helpers (adapted for mfdb_* metadata columns) --

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

    def _decode_raw_data_row(self, row: dict[str, Any] | sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        metadata_raw = data.pop("metadata_json", None) or data.pop("metadata", None) or data.pop("settings_json", None)
        if metadata_raw:
            data["metadata"] = _json_loads(metadata_raw)
        if "artifact_id" in data and "raw_data_id" not in data:
            data["raw_data_id"] = data["artifact_id"]
        if "data_format" in data and "data_type" not in data:
            data["data_type"] = data["data_format"]
        return data

    def _decode_processed_data_row(self, row: dict[str, Any] | sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        settings_raw = data.pop("settings_json", None) or data.pop("settings", None) or data.pop("metadata_json", None)
        if settings_raw:
            data["settings"] = _json_loads(settings_raw)
        if "artifact_id" in data and "processed_data_id" not in data:
            data["processed_data_id"] = data["artifact_id"]
        return data

    # -- update_processing_run_status --

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

    # ============================================================
    # Canonical API — matches the signatures expected by api.py,
    # pipeline.py, and the PRD specification.
    # ============================================================

    def default_runtime_environment(self) -> dict[str, Any]:
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

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

    def _get_object_store(self):
        """Return the shared ObjectStore instance, creating it if needed."""
        if not hasattr(self, "_object_store") or self._object_store is None:
            from mfdb.database_resolver import object_store_root
            from mfdb.object_store import ObjectStore
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

    def browse_datasets(
        self,
        scope: str = "all",
        query: str | None = None,
        kinds: list[str] | None = None,
        formats: list[str] | None = None,
        sample_id: str | None = None,
        owner_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        study_id: str | None = None,
    ) -> dict[str, Any]:
        """Browse datasets with scope-based access control, search, and
        pagination.

        Parameters
        ----------
        scope : str, default='all'
            One of ``'own'`` (only owned by *owner_id*), ``'public'``
            (``is_public = 1``), or ``'all'`` (public + owned by *owner_id*).
        query : str, optional
            Free-text search against artifact_id, artifact_kind, data_format,
            and metadata_json.
        kinds : list of str, optional
            Only include artifacts with these artifact_kind values.
        formats : list of str, optional
            Only include artifacts with these data_format values.
        sample_id : str, optional
            Only include artifacts linked to this sample via mfdb_edge.
        owner_id : str, optional
            User ID for ``'own'`` scope or to supplement ``'all'``.
        limit : int, default=50
            Maximum results per page.
        offset : int, default=0
            Pagination offset.

        Returns
        -------
        dict
            Keys:
              ``datasets`` : list of artifact dicts,
              ``total`` : int (total matching count),
              ``sample_counts`` : dict[str, int] (artifact count per sample).
        """
        where_clauses: list[str] = [
            "a.deleted_at IS NULL",
            "a.artifact_id NOT IN ("
            "  SELECT source_node_id FROM mfdb_edge"
            "  WHERE source_node_type = 'artifact'"
            "    AND target_node_type = 'artifact'"
            "    AND relationship_type = 'grouped_in'"
            "    AND deleted_at IS NULL"
            ")"
        ]
        params: list[Any] = []

        # A dataset can be co-owned (mfdb_artifact_owner). "own" matches the
        # original creator OR any co-owner so every owner sees it under Mine.
        owner_match = (
            "(a.created_by_user_id = ? OR a.artifact_id IN "
            "(SELECT artifact_id FROM mfdb_artifact_owner "
            " WHERE user_id = ? AND deleted_at IS NULL))"
        )
        if scope == "own":
            if not owner_id:
                owner_id = ""
            where_clauses.append(owner_match)
            params.extend([owner_id, owner_id])
        elif scope == "public":
            where_clauses.append("a.is_public = 1")
        elif scope == "all":
            if owner_id:
                where_clauses.append(f"(a.is_public = 1 OR {owner_match})")
                params.extend([owner_id, owner_id])
            else:
                where_clauses.append("a.is_public = 1")

        if query:
            pattern = f"%{query}%"
            where_clauses.append(
                "(a.artifact_id LIKE ? OR a.artifact_kind LIKE ? "
                "OR a.data_format LIKE ? OR a.metadata_json LIKE ?)"
            )
            params.extend([pattern, pattern, pattern, pattern])

        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            where_clauses.append(f"a.artifact_kind IN ({placeholders})")
            params.extend(kinds)

        if formats:
            # Stored data_format is the dot-less suffix (e.g. "ptu"); accept
            # callers that pass either ".ptu" or "ptu".
            norm_formats = [str(f).lstrip(".").lower() for f in formats]
            placeholders = ",".join("?" for _ in norm_formats)
            where_clauses.append(f"LOWER(a.data_format) IN ({placeholders})")
            params.extend(norm_formats)

        if sample_id:
            where_clauses.append(
                "a.artifact_id IN ( "
                "SELECT source_node_id FROM mfdb_edge "
                "WHERE source_node_type = 'artifact' "
                "AND target_node_type = 'sample' "
                "AND target_node_id = ? AND deleted_at IS NULL"
                ")"
            )
            params.append(sample_id)

        if study_id:
            # The study facet (PRD-13): an artifact is in the study if it is a direct
            # member, or if its linked sample is a member (membership is many-to-many).
            where_clauses.append(
                "(a.artifact_id IN ("
                "  SELECT member_id FROM mfdb_study_member "
                "  WHERE study_id = ? AND member_type = 'artifact' AND deleted_at IS NULL"
                ") OR a.artifact_id IN ("
                "  SELECT e.source_node_id FROM mfdb_edge e "
                "  JOIN mfdb_study_member m ON m.member_id = e.target_node_id "
                "    AND m.member_type = 'sample' AND m.study_id = ? "
                "    AND m.deleted_at IS NULL "
                "  WHERE e.source_node_type = 'artifact' "
                "    AND e.target_node_type = 'sample' AND e.deleted_at IS NULL"
                "))"
            )
            params.extend([study_id, study_id])

        where_sql = " AND ".join(where_clauses)

        count_row = self.conn.execute(
            f"SELECT COUNT(*) FROM mfdb_artifact a WHERE {where_sql}",
            params,
        ).fetchone()
        total = count_row[0] if count_row else 0

        rows = self.conn.execute(
            f"SELECT a.*, "
            f"       o.refcount AS object_refcount, "
            f"       o.original_filename AS original_filename, "
            f"       s.description AS sample_name "
            f"FROM mfdb_artifact a "
            f"LEFT JOIN mfdb_object o ON o.object_uuid = a.object_uuid "
            f"LEFT JOIN mfdb_edge e ON e.source_node_id = a.artifact_id "
            f"  AND e.source_node_type = 'artifact' "
            f"  AND e.target_node_type = 'sample' "
            f"  AND e.deleted_at IS NULL "
            f"LEFT JOIN flr_sample s ON s.sample_id = e.target_node_id "
            f"  AND s.deleted_at IS NULL "
            f"WHERE {where_sql} "
            f"ORDER BY a.created_at DESC, a.artifact_id LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        datasets = [dict(r) for r in rows]

        sample_counts: dict[str, int] = {}
        if datasets:
            art_ids = [d["artifact_id"] for d in datasets]
            placeholders = ",".join("?" for _ in art_ids)
            sc_rows = self.conn.execute(
                f"SELECT e.target_node_id AS sample_id, COUNT(*) AS cnt "
                f"FROM mfdb_edge e "
                f"WHERE e.source_node_type = 'artifact' "
                f"AND e.target_node_type = 'sample' "
                f"AND e.deleted_at IS NULL "
                f"AND e.source_node_id IN ({placeholders}) "
                f"GROUP BY e.target_node_id",
                art_ids,
            ).fetchall()
            for sr in sc_rows:
                sample_counts[sr["sample_id"]] = sr["cnt"]

        return {
            "datasets": datasets,
            "total": total,
            "sample_counts": sample_counts,
        }

    def open_dataset(self, artifact_id: str) -> str:
        """Materialize a dataset artifact to a readable local file path.

        Parameters
        ----------
        artifact_id : str
            Artifact identifier.

        Returns
        -------
        str
            Local file path to the materialized dataset content.

        Raises
        ------
        KeyError
            If the artifact or its stored object is not found.
        """
        art = self.get_artifact(artifact_id)
        if art is None:
            raise KeyError(f"Artifact not found: {artifact_id}")

        object_uuid = art.get("object_uuid")
        if object_uuid:
            store = self._get_object_store()
            row = self.conn.execute(
                "SELECT content_md5, original_filename FROM mfdb_object WHERE object_uuid = ?",
                (object_uuid,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Object not found for artifact: {artifact_id}")
            blob_path = store.get_path(row["content_md5"])
            original_name = row["original_filename"] or artifact_id
            suffix = Path(original_name).suffix if "." in original_name else ""
            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, prefix="chisurf-ds-"
            )
            tmp.write(blob_path.read_bytes())
            tmp.close()
            return tmp.name

        file_path = art.get("file_path")
        if file_path and Path(file_path).exists():
            return str(file_path)

        # External references (e.g. a burst output directory or .bur file) carry
        # their path in metadata rather than the object store; materialize that.
        metadata = art.get("metadata") or art.get("metadata_json")
        if isinstance(metadata, str):
            try:
                metadata = _json_loads(metadata)
            except Exception:
                metadata = None
        if isinstance(metadata, dict):
            meta_path = metadata.get("path") or metadata.get("folder_path")
            if meta_path and Path(meta_path).exists():
                return str(meta_path)

        raise KeyError(
            f"Artifact {artifact_id} has no materializable content "
            f"(no object_uuid and no valid file_path)"
        )

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
            from mfdb.session import configured_default_user_id
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
                from mfdb.auth import create_default_acl_for_object
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

    def _normalize_parameter_payload(self, parameter: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(parameter, dict):
            raise ValueError("parameter payload must be a mapping")
        parameter_uuid = parameter.get("parameter_uuid", parameter.get("id"))
        name = parameter.get("name")
        if not parameter_uuid or not name:
            raise ValueError("parameter_uuid and name are required")
        parameter_type = parameter.get("parameter_type", "free")
        validate_vocabulary(parameter_type, PARAMETER_TYPES, "parameter_type")
        return {
            "parameter_uuid": parameter_uuid,
            "name": name,
            "value": parameter.get("value"),
            "standard_error": parameter.get("standard_error"),
            "confidence_interval_low": parameter.get("confidence_interval_low"),
            "confidence_interval_high": parameter.get("confidence_interval_high"),
            "initial_value": parameter.get("initial_value"),
            "lower_bound": parameter.get("lower_bound"),
            "upper_bound": parameter.get("upper_bound"),
            "bounds_on": parameter.get("bounds_on", False),
            "units": parameter.get("units"),
            "parameter_type": parameter_type,
            "expression": parameter.get("expression"),
            "prior": parameter.get("prior"),
            "mapping": parameter.get("mapping"),
            "metadata": parameter.get("metadata"),
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

    def record_parameter(
        self,
        parameter_uuid: str,
        operation_id: str,
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
        role: str | None = None,
    ) -> str:
        validate_vocabulary(parameter_type, PARAMETER_TYPES, "parameter_type")
        now = _utc_now()
        with self._transaction():
            self.conn.execute(
                """INSERT INTO mfdb_parameter (
                    parameter_uuid, operation_id, name, value, standard_error,
                    confidence_interval_low, confidence_interval_high, initial_value,
                    lower_bound, upper_bound, bounds_on, units, parameter_type, role,
                    expression, prior_json, mapping_json, metadata_json,
                    created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(parameter_uuid) DO UPDATE SET
                    operation_id=excluded.operation_id,
                    name=excluded.name,
                    value=excluded.value,
                    standard_error=excluded.standard_error,
                    confidence_interval_low=excluded.confidence_interval_low,
                    confidence_interval_high=excluded.confidence_interval_high,
                    initial_value=excluded.initial_value,
                    lower_bound=excluded.lower_bound,
                    upper_bound=excluded.upper_bound,
                    bounds_on=excluded.bounds_on,
                    units=excluded.units,
                    parameter_type=excluded.parameter_type,
                    role=excluded.role,
                    expression=excluded.expression,
                    prior_json=excluded.prior_json,
                    mapping_json=excluded.mapping_json,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at,
                    deleted_at=excluded.deleted_at""",
                (
                    parameter_uuid,
                    operation_id,
                    name,
                    value,
                    standard_error,
                    confidence_interval_low,
                    confidence_interval_high,
                    initial_value,
                    lower_bound,
                    upper_bound,
                    1 if bounds_on else 0,
                    units,
                    parameter_type,
                    role,
                    expression,
                    _json_dumps(prior),
                    _json_dumps(mapping),
                    _json_dumps(metadata),
                    now,
                    now,
                    None,
                ),
            )
            self.add_audit_log(
                action="create",
                target_type="parameter",
                target_id=parameter_uuid,
                details={"operation_id": operation_id, "name": name},
            )
        return parameter_uuid

    def get_parameter(self, parameter_uuid: str) -> dict[str, Any] | None:
        # PRD-26 Task 2: parameterised, schema-driven get-by-PK (was a hand SELECT).
        return self.dao.get("mfdb_parameter", parameter_uuid, include_deleted=True)

    def list_parameters(
        self,
        operation_id: str | None = None,
        parameter_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List parameters with optional filters.

        Parameters
        ----------
        operation_id : str, optional
            Filter by operation identifier.
        parameter_type : str, optional
            Filter by parameter vocabulary value.

        Returns
        -------
        list of dict
            Parameter dictionaries.
        """
        query = "SELECT * FROM mfdb_parameter WHERE 1=1 AND deleted_at IS NULL"
        params: list[Any] = []
        if operation_id is not None:
            query += " AND operation_id = ?"
            params.append(operation_id)
        if parameter_type is not None:
            query += " AND parameter_type = ?"
            params.append(parameter_type)
        query += " ORDER BY parameter_id"
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    def save_setup(
        self,
        setup_id: str,
        name: str,
        version: int = 1,
        instrument_id: str | None = None,
        description: str | None = None,
        configuration: dict[str, Any] | None = None,
        detectors: dict[str, Any] | None = None,
        windows: dict[str, Any] | None = None,
        timing_calibration: dict[str, Any] | None = None,
        irf_definition: dict[str, Any] | None = None,
        dark_count: dict[str, Any] | None = None,
        timing_resolution: dict[str, Any] | None = None,
        burst_defaults: dict[str, Any] | None = None,
        fcs_calibration: dict[str, Any] | None = None,
        created_by_user_id: str | None = None,
        is_public: bool | int | None = None,
        fcs_pairs: dict[str, Any] | None = None,
        n_bins: int | None = None,
        n_casc: int | None = None,
        make_fine: bool | int | None = None,
    ) -> None:
        now = _utc_now()
        # Extract typed timing columns from the timing_resolution dict.
        # These are the dictionary-authoritative columns; timing_resolution_json
        # remains for backward compatibility.
        timing_dict = timing_resolution or {}
        _macro_t = timing_dict.get("macro_time_resolution")
        _micro_t = timing_dict.get("micro_time_resolution")
        _micro_b = timing_dict.get("micro_time_binning")
        # Normalize empty-string user_id to None so the FK constraint holds
        _owner = created_by_user_id or None
        with self._transaction():
            self.conn.execute(
                """INSERT INTO mfdb_setup (
                    setup_id, name, version, instrument_id, description,
                    configuration_json, detectors_json, timing_calibration_json,
                    irf_definition_json, dark_count_json, timing_resolution_json,
                    macro_time_resolution, micro_time_resolution, micro_time_binning,
                    n_bins, n_casc, make_fine,
                    burst_defaults_json, fcs_calibration_json,
                    created_by_user_id,
                    is_public,
                    created_at, updated_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(setup_id) DO UPDATE SET
                    name=excluded.name,
                    version=excluded.version,
                    instrument_id=excluded.instrument_id,
                    description=excluded.description,
                    configuration_json=excluded.configuration_json,
                    detectors_json=excluded.detectors_json,
                    timing_calibration_json=excluded.timing_calibration_json,
                    irf_definition_json=excluded.irf_definition_json,
                    dark_count_json=excluded.dark_count_json,
                    timing_resolution_json=excluded.timing_resolution_json,
                    macro_time_resolution=excluded.macro_time_resolution,
                    micro_time_resolution=excluded.micro_time_resolution,
                    micro_time_binning=excluded.micro_time_binning,
                    n_bins=excluded.n_bins,
                    n_casc=excluded.n_casc,
                    make_fine=excluded.make_fine,
                    burst_defaults_json=excluded.burst_defaults_json,
                    fcs_calibration_json=excluded.fcs_calibration_json,
                    created_by_user_id=excluded.created_by_user_id,
                    is_public=excluded.is_public,
                    updated_at=excluded.updated_at,
                    deleted_at=excluded.deleted_at""",
                (
                    setup_id,
                    name,
                    version,
                    instrument_id,
                    description,
                    _json_dumps(configuration),
                    _json_dumps(detectors),
                    _json_dumps(timing_calibration),
                    _json_dumps(irf_definition),
                    _json_dumps(dark_count),
                    _json_dumps(timing_resolution),
                    _macro_t,
                    _micro_t,
                    _micro_b,
                    n_bins,
                    n_casc,
                    1 if make_fine else 0 if make_fine is not None else None,
                    _json_dumps(burst_defaults),
                    _json_dumps(fcs_calibration),
                    _owner,
                    1 if is_public is True else 0,
                    now,
                    now,
                    None,
                ),
            )

            # Write structured detector channel rows
            if detectors:
                self.conn.execute(
                    "UPDATE mfdb_setup_detector_channel SET deleted_at = ? WHERE setup_id = ? AND deleted_at IS NULL",
                    (now, setup_id)
                )
                for det_name, det_data in detectors.items():
                    if not isinstance(det_data, dict):
                        continue
                    channels = det_data.get("channels") or det_data.get("chs")
                    mtr = det_data.get("micro_time_ranges")
                    g_factor = det_data.get("g_factor")
                    l1 = det_data.get("l1")
                    l2 = det_data.get("l2")
                    gfc = det_data.get("g_factor_channels")
                    gfc_id = det_data.get("g_factor_calibration_id")
                    self.conn.execute(
                        """INSERT INTO mfdb_setup_detector_channel
                            (setup_id, name, channels, micro_time_ranges,
                             g_factor, l1, l2, g_factor_channels, g_factor_calibration_id,
                             created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            setup_id, det_name,
                            _json_dumps(channels) if channels is not None else None,
                            _json_dumps(mtr) if mtr is not None else None,
                            g_factor, l1, l2,
                            _json_dumps(gfc) if gfc is not None else None,
                            gfc_id,
                            now, now
                        )
                    )
                    # Append a calibration snapshot only when calibration data is
                    # present AND differs from the latest snapshot for this
                    # channel. Plain structural re-saves must not create
                    # duplicate-factor snapshots (which would pollute the
                    # calibration-date history).
                    has_cal = g_factor is not None or l1 is not None or l2 is not None
                    latest = self.conn.execute(
                        "SELECT g_factor, l1, l2, g_factor_calibration_id "
                        "FROM mfdb_setup_calibration "
                        "WHERE setup_id = ? AND channel_name = ? AND deleted_at IS NULL "
                        "ORDER BY calibrated_at DESC LIMIT 1",
                        (setup_id, det_name),
                    ).fetchone()
                    unchanged = latest is not None and (
                        latest[0] == g_factor and latest[1] == l1
                        and latest[2] == l2 and latest[3] == gfc_id
                    )
                    if has_cal and not unchanged:
                        self.conn.execute(
                            """INSERT INTO mfdb_setup_calibration
                                (setup_id, channel_name, g_factor, l1, l2,
                                 g_factor_channels, g_factor_calibration_id,
                                 calibrated_at, method, created_by_user_id,
                                 created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                setup_id, det_name, g_factor, l1, l2,
                                _json_dumps(gfc) if gfc is not None else None,
                                gfc_id,
                                now, "manual", _owner,
                                now, now,
                            ),
                        )

            # Write structured PIE window rows
            if windows:
                self.conn.execute(
                    "UPDATE mfdb_setup_pie_window SET deleted_at = ? WHERE setup_id = ? AND deleted_at IS NULL",
                    (now, setup_id)
                )
                for win_name, bounds in windows.items():
                    if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
                        continue
                    start_val, end_val = int(bounds[0]), int(bounds[1])
                    self.conn.execute(
                        """INSERT INTO mfdb_setup_pie_window
                            (setup_id, name, start, end, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                        (setup_id, win_name, start_val, end_val, now, now)
                    )

            # Write structured FCS channel pair rows
            if fcs_pairs:
                self.conn.execute(
                    "UPDATE mfdb_setup_fcs_pair SET deleted_at = ? WHERE setup_id = ? AND deleted_at IS NULL",
                    (now, setup_id)
                )
                for pair_name, pair_data in fcs_pairs.items():
                    if isinstance(pair_data, dict):
                        channel_a = pair_data.get("channel_a", "")
                        channel_b = pair_data.get("channel_b", "")
                        kind = pair_data.get("kind")
                        pair_n_bins = pair_data.get("n_bins")
                        pair_n_casc = pair_data.get("n_casc")
                        pair_make_fine = pair_data.get("make_fine")
                    elif isinstance(pair_data, (list, tuple)) and len(pair_data) >= 2:
                        channel_a = str(pair_data[0]) if pair_data[0] else ""
                        channel_b = str(pair_data[1]) if pair_data[1] else ""
                        kind = str(pair_data[2]) if len(pair_data) > 2 and pair_data[2] else None
                        pair_n_bins = pair_n_casc = pair_make_fine = None
                    else:
                        continue
                    if not channel_a or not channel_b:
                        continue
                    self.conn.execute(
                        """INSERT INTO mfdb_setup_fcs_pair
                            (setup_id, name, channel_a, channel_b, kind,
                             n_bins, n_casc, make_fine,
                             created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            setup_id, pair_name, channel_a, channel_b, kind,
                            pair_n_bins, pair_n_casc,
                            1 if pair_make_fine else 0 if pair_make_fine is not None else None,
                            now, now,
                        )
                    )

            self.add_audit_log(
                action="create",
                target_type="setup",
                target_id=setup_id,
                details={"name": name, "version": version},
            )

    def get_setup(
        self,
        setup_id: str,
        calibrated_at: str | None = None,
    ) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM mfdb_setup WHERE setup_id = ?", (setup_id,)
        ).fetchone()
        result = _row_to_dict(row)
        if result is not None:
            result["detector_channels"] = [
                dict(r) for r in self.conn.execute(
                    "SELECT * FROM mfdb_setup_detector_channel WHERE setup_id = ? AND deleted_at IS NULL ORDER BY id",
                    (setup_id,)
                ).fetchall()
            ]
            result["pie_windows"] = [
                dict(r) for r in self.conn.execute(
                    "SELECT * FROM mfdb_setup_pie_window WHERE setup_id = ? AND deleted_at IS NULL ORDER BY id",
                    (setup_id,)
                ).fetchall()
            ]
            result["fcs_pairs"] = [
                dict(r) for r in self.conn.execute(
                    "SELECT * FROM mfdb_setup_fcs_pair WHERE setup_id = ? AND deleted_at IS NULL ORDER BY id",
                    (setup_id,)
                ).fetchall()
            ]
            result["calibration_dates"] = self.list_setup_calibration_dates(setup_id)
            result["calibration"] = self.get_setup_calibration(setup_id, calibrated_at=calibrated_at)
        return result

    def list_setups(self) -> list[dict[str, Any]]:
        """List setup snapshots.

        Returns
        -------
        list of dict
            Setup dictionaries.
        """
        rows = self.conn.execute("SELECT * FROM mfdb_setup WHERE deleted_at IS NULL ORDER BY name, version").fetchall()
        return [dict(r) for r in rows]

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

    def add_audit_log(
        self,
        action: str,
        target_type: str,
        target_id: str,
        operator_user_id: str | None = None,
        details: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> int:
        if operator_user_id is None:
            from mfdb.session import configured_default_user_id
            operator_user_id = configured_default_user_id()
        now = timestamp or _utc_now()
        with self._transaction():
            self.conn.execute(
                """INSERT INTO mfdb_audit_log
                   (action, target_type, target_id, operator_user_id, details_json, timestamp, created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (action, target_type, target_id, operator_user_id, _json_dumps(details), now, now, now, None),
            )
            return int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def get_audit_logs(
        self,
        action: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM mfdb_audit_log WHERE 1=1 AND deleted_at IS NULL"
        params: list[Any] = []
        if action is not None:
            query += " AND action = ?"
            params.append(action)
        if target_type is not None:
            query += " AND target_type = ?"
            params.append(target_type)
        if target_id is not None:
            query += " AND target_id = ?"
            params.append(target_id)
        query += " ORDER BY timestamp DESC, log_id DESC LIMIT ?"
        params.append(limit)
        logs = []
        for row in self.conn.execute(query, params).fetchall():
            data = dict(row)
            data["details"] = _json_loads(data.get("details_json"))
            logs.append(data)
        return logs

    # ── Legacy backward-compat stubs ───────────────────────────────────

    def add_raw_data_reference(
        self,
        raw_data_id: str | None = None,
        experiment_id: str | None = None,
        data_type: str = "PTU",
        storage_mode: str = "local_file",
        file_path: str | None = None,
        acquired_at: str | None = None,
        validation_status: str = "unvalidated",
        **kwargs,
    ) -> str:
        import uuid
        art_id = raw_data_id or f"raw_{uuid.uuid4().hex[:12]}"
        checksum = kwargs.pop("checksum", None)
        size_bytes = kwargs.pop("size_bytes", None)
        checksum_algorithm = kwargs.pop("checksum_algorithm", "sha256")
        mime_type = kwargs.pop("mime_type", None)
        row_count = kwargs.pop("row_count", None)
        return self.register_artifact(
            artifact_id=art_id,
            artifact_kind="raw_data",
            storage_mode=storage_mode,
            experiment_id=experiment_id,
            file_path=file_path,
            validation_status=validation_status,
            checksum=checksum,
            size_bytes=size_bytes,
            checksum_algorithm=checksum_algorithm,
            mime_type=mime_type,
            row_count=row_count,
            metadata={"data_type": data_type, "acquired_at": acquired_at, **kwargs},
        )


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

    def add_processed_data_product(
        self,
        processing_id: str | None = None,
        product_type: str | None = None,
        storage_mode: str | None = None,
        product: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        import uuid
        prod_dict = product if product else kwargs
        pt = product_type or prod_dict.get("product_type", "processed_data")
        sm = storage_mode or prod_dict.get("storage_mode", "local_file")
        prod_id = prod_dict.get("processed_data_id") or f"prod_{uuid.uuid4().hex[:12]}"
        pid = processing_id or prod_dict.get("processing_id") or "unknown"
        with self._transaction():
            self.register_artifact(
                artifact_id=prod_id,
                artifact_kind=pt,
                storage_mode=sm,
                file_path=prod_dict.get("file_path"),
                folder_path=prod_dict.get("folder_path"),
                checksum=prod_dict.get("checksum"),
                row_count=prod_dict.get("row_count"),
                validation_status=prod_dict.get("validation_status", "unvalidated"),
                metadata=prod_dict,
            )
            self.record_operation_link(
                operation_id=pid,
                artifact_id=prod_id,
                direction="output",
            )
            self.add_audit_log(
                action="create",
                target_type="processed_data",
                target_id=prod_id,
                details={"processing_id": pid, "product_type": pt},
            )
        return prod_id

    def get_processed_data(self, data_id: str) -> dict[str, Any] | None:
        return self.get_artifact(data_id)

    def get_processed_data_products(
        self, processing_id: str | None = None, **kwargs
    ) -> list[dict[str, Any]]:
        if processing_id:
            links = self.conn.execute(
                "SELECT artifact_id FROM mfdb_operation_artifact WHERE operation_id = ? AND direction = 'output' AND deleted_at IS NULL",
                (processing_id,),
            ).fetchall()
            return [self.get_artifact(r["artifact_id"]) for r in links if self.get_artifact(r["artifact_id"])]
        return self.list_artifacts(**kwargs)

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

    def _decode_processed_data_row(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        row = dict(row)
        if "artifact_kind" in row and "product_type" not in row:
            row["product_type"] = row["artifact_kind"]
        if "artifact_id" in row and "processed_data_id" not in row:
            row["processed_data_id"] = row["artifact_id"]
        return row

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

    def add_setup_definition(self, setup_id: str, name: str, **kwargs):
        result = self.save_setup(setup_id=setup_id, name=name, **kwargs)
        self.add_audit_log(
            action="create",
            target_type="setup_definition",
            target_id=setup_id,
            details={"name": name},
        )
        return result

    def get_setup_definition(self, setup_id: str, **kwargs):
        return self.get_setup(setup_id, **kwargs)

    def delete_setup_definition(self, setup_id: str, **kwargs):
        return self.delete_setup(setup_id)

    def _decode_setup_definition_row(self, row):
        if row is None:
            return None
        row = dict(row)
        for json_field in ("configuration", "detectors", "timing_calibration", "irf_definition",
                           "dark_count", "timing_resolution", "burst_defaults", "fcs_calibration"):
            col = f"{json_field}_json"
            if col in row and isinstance(row[col], str):
                try:
                    row[json_field] = json.loads(row[col])
                except (json.JSONDecodeError, TypeError):
                    pass
            if col in row:
                del row[col]
        return row

    def list_setup_definitions(self, **kwargs):
        return self.get_setups(**kwargs)

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

    def trace_processed_data(self, processed_data_id: str, **kwargs) -> dict[str, Any] | None:
        product = self.get_processed_data(processed_data_id)
        if product is None:
            return None
        product = self._decode_processed_data_row(product)

        # Find the operation linked to this artifact
        op_row = self.conn.execute(
            "SELECT operation_id FROM mfdb_operation_artifact WHERE artifact_id = ? AND direction = 'output' AND deleted_at IS NULL",
            (processed_data_id,)
        ).fetchone()
        processing_id = op_row["operation_id"] if op_row else None

        if not processing_id:
            raise ValueError(f"No producing operation found for processed_data_id {processed_data_id!r}")

        product["processing_id"] = processing_id

        run = None
        if processing_id:
            run = self.get_processing_run_full(processing_id)

        included_edges = [
            dict(row)
            for row in self.get_provenance_edges(
                source_node_type="processed_data",
                source_node_id=processed_data_id,
                relationship_type="included_in",
            )
        ]
        return {
            "processed_data": product,
            "processing_run": run,
            "included_in": included_edges,
        }

    def add_trace_processed_data(self, processed_data_id: str, **kwargs):
        return self.get_processed_data(processed_data_id)

    def export_burst_processing_manifest(self, processing_id: str) -> dict[str, Any]:
        run = self.get_processing_run_full(processing_id)
        if run is None:
            raise KeyError(f"processing run not found: {processing_id}")
        experiment = _row_to_dict(self.get_experiment(str(run.get("experiment_id"))))
        products = [
            self._decode_processed_data_row(row)
            for row in self.get_processed_data_products(processing_id=processing_id)
        ]
        raw_data = [self._decode_raw_data_row(row) for row in run.get("input_raw_data", [])]
        edges = [
            self._decode_provenance_edge_row(row)
            for row in self.get_provenance_edges(processing_id=processing_id)
        ]
        return {
            "schema": "mfdb.burst_processing_manifest.v1",
            "exported_at": _utc_now(),
            "experiment": experiment,
            "processing_run": run,
            "raw_data": raw_data,
            "processed_data": products,
            "provenance_edges": edges,
        }

    def register_archive_manifest(
        self,
        processing_id: str,
        manifest: dict[str, Any] | None = None,
        output_path: str | None = None,
    ) -> str:
        manifest = manifest or self.export_burst_processing_manifest(processing_id)
        data_json = _json_dumps(manifest)
        product_id = self.add_processed_data_product(
            processing_id,
            "archive_manifest",
            "local_file" if output_path else "embedded_json",
            file_path=output_path,
            mime_type="application/json",
            size_bytes=len(data_json.encode("utf-8")) if data_json else None,
            checksum=hashlib.sha256(data_json.encode("utf-8")).hexdigest()
            if data_json
            else None,
            data_json=data_json,
            validation_status="valid",
        )
        run = self.get_processing_run(processing_id)
        for product in self.get_processed_data_products(processing_id=processing_id):
            other_id = product.get("processed_data_id") or product.get("artifact_id")
            if other_id == product_id:
                continue
            self.add_provenance_edge(
                source_node_type="processed_data",
                source_node_id=other_id,
                target_node_type="processed_data",
                target_node_id=product_id,
                relationship_type="included_in",
                processing_id=processing_id,
                settings_hash=run.get("settings_hash") if run else None,
                software_version=run.get("software_version") if run else None,
                checksum_snapshot={
                    "source": product.get("checksum"),
                    "archive_manifest": hashlib.sha256(data_json.encode("utf-8")).hexdigest()
                    if data_json
                    else None,
                },
            )
        return product_id


    # ── PRD-named aliases ──────────────────────────────────────────────

    link_operation_artifact = record_operation_link
    save_setup_snapshot = save_setup

    def register_sample(self, sample_id, **kwargs):
        return self.add_sample(sample_id, **kwargs)

    def register_experiment(self, experiment_id, **kwargs):
        return self.add_experiment(experiment_id, **kwargs)

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
        from mfdb.graph import traverse_canonical_graph
        return traverse_canonical_graph(
            self.conn,
            node_type,
            node_id,
            direction="upstream",
            max_depth=max_depth,
            canonical=True,
        )

    def graph_downstream(self, node_type: str, node_id: str, max_depth: int = 100) -> list[dict[str, Any]]:
        from mfdb.graph import traverse_canonical_graph
        return traverse_canonical_graph(
            self.conn,
            node_type,
            node_id,
            direction="downstream",
            max_depth=max_depth,
            canonical=True,
        )

    list_audit_logs = get_audit_logs

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

    def link_grouped_fits(self, local_fit_uuid: str, global_fit_uuid: str) -> None:
        local_run = self.get_analysis_run(local_fit_uuid)
        self.add_provenance_edge(
            source_node_type="analysis_run",
            source_node_id=global_fit_uuid,
            target_node_type="analysis_run",
            target_node_id=local_fit_uuid,
            relationship_type="grouped_in",
            processing_id=global_fit_uuid,
            software_version=local_run["software_version"] if local_run else None,
        )

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
