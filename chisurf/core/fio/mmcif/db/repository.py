import sqlite3
import logging
import hashlib
import json
import platform
import uuid
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from . import schema
from .database_resolver import backup_database_before_migration, resolve_database_path
from .models import (
    Probe,
    ProbeType,
    Entity,
    SequenceResidue,
    PolyProbePosition,
    SampleCondition,
    SampleProbe,
    EntityAssembly,
    OpticalProperty,
    Spectrum,
    ExternalFile,
    PhotonStream,
    AnalysisMetadata,
)

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format.

    Returns
    -------
    str
        Timezone-aware UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str | None:
    """Serialize a value as stable JSON text.

    Parameters
    ----------
    value : Any
        JSON-compatible value.

    Returns
    -------
    str or None
        Serialized JSON, or ``None`` for unset values.
    """
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _json_loads(value: str | None) -> Any:
    """Deserialize JSON text when present.

    Parameters
    ----------
    value : str or None
        JSON text.

    Returns
    -------
    Any
        Parsed JSON value, or ``None``.
    """
    if not value:
        return None
    return json.loads(value)


def _json_hash(value: Any) -> str | None:
    """Return a SHA-256 hash for a JSON-compatible value.

    Parameters
    ----------
    value : Any
        JSON-compatible value.

    Returns
    -------
    str or None
        Stable SHA-256 digest, or ``None`` when value is unset.
    """
    text = _json_dumps(value)
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert a SQLite row to a dictionary.

    Parameters
    ----------
    row : sqlite3.Row or None
        Row returned by SQLite.

    Returns
    -------
    dict or None
        Dictionary representation, or ``None`` for missing rows.
    """
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


class FluorophoreDatabase:
    """Spectral reference library and sample management repository."""

    _VALID_ENUMS = {
        "category": ["organic_dye", "protein", "nanoparticle", "quantum_dot", "other"],
        "probe_origin": ["extrinsic", "intrinsic", "other"],
        "probe_link_type": ["covalent", "non-covalent", "other"],
        "fluorophore_type": ["unspecified", "small_molecule", "protein_domain"],
        "reactive_probe_flag": ["yes", "no"],
    }

    _PROP_ALIASES = {
        "abs_max": [
            "abs_max",
            "absorption maximum",
            "λabs",
            "excitation max",
            "ex_max",
            "abs_peak",
            "λex",
        ],
        "em_max": ["em_max", "emission maximum", "λfl", "emission max", "em_max", "em_peak", "λem"],
        "qy": [
            "qy",
            "fluorescence quantum yield",
            "ηfl",
            "quantum yield",
            "phi_acceptor",
            "phi",
            "qy_d",
            "phi_d",
        ],
        "lifetime": ["lifetime", "fluorescence lifetime", "τfl", "tau", "tau_d", "tau_0"],
        "ext_coeff": [
            "ext_coeff",
            "molar extinction coefficient",
            "εmax",
            "extinction coefficient",
            "epsilon",
            "molar_ec",
        ],
    }

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """Initialize the FluorophoreDatabase.

        Parameters
        ----------
        db_path : str or Path, optional
            Path to the SQLite database file. Defaults to
            sample_management.db in the plugin directory.
        """
        if db_path is None:
            db_path = resolve_database_path()

        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._ensure_connection()
        backup_database_before_migration(self.db_path, schema.SCHEMA_VERSION)
        schema.migrate_schema(self.conn)
        self._normalize_optical_property_keys()

    def _ensure_connection(self):
        """Ensure the database connection is established."""
        if self.conn is None:
            self.connect()

    def connect(self):
        """Open a connection to the database."""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            if str(self.db_path) != ":memory:":
                try:
                    self.conn.execute("PRAGMA journal_mode = WAL")
                except sqlite3.OperationalError:
                    pass

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Enter context manager and ensure connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, committing or rolling back."""
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()

    def _get_schema_version(self) -> int:
        """Return the current schema version."""
        return schema.get_schema_version(self.conn)

    def _validate_enum(self, key: str, value: str):
        """Validate an enum value against known valid options.

        Parameters
        ----------
        key : str
            Enum field name.
        value : str
            Value to validate.

        Raises
        ------
        ValueError
            If value is not valid for the given key.
        """
        if key in self._VALID_ENUMS:
            valid = self._VALID_ENUMS[key]
            if value not in valid:
                raise ValueError(f"Invalid {key}: {value}. Must be one of {valid}")

    def _normalize_optical_property_keys(self):
        """Normalize optical property names to canonical keys."""
        canonical_map = {}
        for canonical, aliases in self._PROP_ALIASES.items():
            for alias in aliases:
                canonical_map[alias.lower()] = canonical

        # Detach results immediately
        rows = [
            dict(r)
            for r in self.conn.execute(
                "SELECT id, probe_id, property_name FROM optical_properties"
            ).fetchall()
        ]
        if not rows:
            return

        with self.conn:
            for row in rows:
                name_low = row["property_name"].lower()
                if name_low in canonical_map:
                    new_name = canonical_map[name_low]
                    if new_name != row["property_name"]:
                        exists = self.conn.execute(
                            "SELECT id FROM optical_properties WHERE probe_id = ? AND property_name = ?",
                            (row["probe_id"], new_name),
                        ).fetchone()
                        if exists:
                            self.conn.execute(
                                "DELETE FROM optical_properties WHERE id = ?", (row["id"],)
                            )
                        else:
                            self.conn.execute(
                                "UPDATE optical_properties SET property_name = ? WHERE id = ?",
                                (new_name, row["id"]),
                            )

    # ── CRUD Operations ──

    def add_probe_type(self, type_name: str, display_name: str) -> int:
        """Add a probe type.

        Parameters
        ----------
        type_name : str
            Internal type name.
        display_name : str
            Human-readable display name.

        Returns
        -------
        int
            The type_id.
        """
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO probe_types (type_name, display_name) VALUES (?, ?)",
                (type_name, display_name),
            )
        row = self.conn.execute(
            "SELECT type_id FROM probe_types WHERE type_name = ?", (type_name,)
        ).fetchone()
        return row["type_id"]

    def get_probe_types(self) -> List[sqlite3.Row]:
        """Return all probe types.

        Returns
        -------
        list of sqlite3.Row
        """
        return self.conn.execute(
            "SELECT type_id, type_name, display_name FROM probe_types"
        ).fetchall()

    def add_probe(
        self,
        chromophore_name: str,
        type_id: int,
        description: str = "",
        category: str = "other",
        **kwargs,
    ) -> int:
        """Add a new probe.

        Parameters
        ----------
        chromophore_name : str
            Name of the chromophore.
        type_id : int
            Probe type identifier.
        description : str
            Optional description.
        category : str
            Probe category.
        **kwargs
            Additional fields.

        Returns
        -------
        int
            The probe_id.
        """
        self._validate_enum("category", category)
        for k, v in kwargs.items():
            self._validate_enum(k, v)

        cols = ["chromophore_name", "type_id", "description", "category"]
        vals = [chromophore_name, type_id, description, category]
        for k, v in kwargs.items():
            cols.append(k)
            vals.append(v)

        placeholders = ", ".join(["?"] * len(cols))
        with self.conn:
            self.conn.execute(
                f"INSERT OR IGNORE INTO probes ({', '.join(cols)}) VALUES ({placeholders})",
                tuple(vals),
            )
        row = self.conn.execute(
            "SELECT probe_id FROM probes WHERE chromophore_name = ? AND type_id = ?",
            (chromophore_name, type_id),
        ).fetchone()
        return row["probe_id"]

    def get_probes(self, type_id: int = None, curated_only: bool = False) -> List[sqlite3.Row]:
        """Get probes, optionally filtered by type or curation status.

        Parameters
        ----------
        type_id : int, optional
            Filter by probe type.
        curated_only : bool
            If True, return only curated probes.

        Returns
        -------
        list of sqlite3.Row
        """
        query = "SELECT * FROM probes WHERE 1=1"
        params = []
        if type_id is not None:
            query += " AND type_id = ?"
            params.append(type_id)
        if curated_only:
            query += " AND is_curated = 1"
        return self.conn.execute(query, params).fetchall()

    def get_probe_by_id(self, probe_id: int) -> Optional[sqlite3.Row]:
        """Get a single probe by its ID.

        Parameters
        ----------
        probe_id : int
            Probe identifier.

        Returns
        -------
        sqlite3.Row or None
        """
        return self.conn.execute("SELECT * FROM probes WHERE probe_id = ?", (probe_id,)).fetchone()

    def update_probe(self, probe_id: int, **kwargs):
        """Update probe fields.

        Parameters
        ----------
        probe_id : int
            Probe identifier.
        **kwargs
            Field name/value pairs to update.
        """
        if not kwargs:
            return
        for k, v in kwargs.items():
            self._validate_enum(k, v)
        cols = [f"{k} = ?" for k in kwargs.keys()]
        with self.conn:
            self.conn.execute(
                f"UPDATE probes SET {', '.join(cols)} WHERE probe_id = ?",
                (*kwargs.values(), probe_id),
            )

    def delete_probe(self, probe_id: int):
        """Delete a probe and all related data.

        Parameters
        ----------
        probe_id : int
            Probe identifier.
        """
        with self.conn:
            for t in ["spectra", "optical_properties", "images", "flr_poly_probe_position"]:
                self.conn.execute(f"DELETE FROM {t} WHERE probe_id = ?", (probe_id,))
            self.conn.execute("DELETE FROM probes WHERE probe_id = ?", (probe_id,))

    def add_optical_property(
        self, probe_id: int, name: str, value: str, unit: Optional[str] = None
    ):
        """Add or update an optical property for a probe.

        Parameters
        ----------
        probe_id : int
            Probe identifier.
        name : str
            Property name.
        value : str
            Property value.
        unit : str, optional
            Unit of the property.
        """
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO optical_properties (probe_id, property_name, property_value, unit) VALUES (?, ?, ?, ?)",
                (probe_id, name, str(value), unit),
            )

    def get_optical_properties(self, probe_id: int) -> Dict[str, str]:
        """Get all optical properties for a probe.

        Parameters
        ----------
        probe_id : int
            Probe identifier.

        Returns
        -------
        dict of str to str
        """
        rows = self.conn.execute(
            "SELECT property_name, property_value FROM optical_properties WHERE probe_id = ?",
            (probe_id,),
        ).fetchall()
        return {r["property_name"]: r["property_value"] for r in rows}

    def add_spectrum(
        self,
        probe_id: int,
        spec_type: str,
        wavelengths: np.ndarray,
        values: np.ndarray,
        wavelength_unit: str = "nm",
        intensity_unit: str = "normalized",
        details: Optional[str] = None,
    ):
        """Add or replace a spectrum for a probe.

        Parameters
        ----------
        probe_id : int
            Probe identifier.
        spec_type : str
            Spectrum type (e.g. 'absorption', 'emission').
        wavelengths : np.ndarray
            Wavelength array.
        values : np.ndarray
            Intensity values.
        wavelength_unit : str
            Unit for wavelength values.
        intensity_unit : str
            Unit for intensity values.
        details : str, optional
            Optional details.
        """
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        w_blob = wavelengths.tobytes()
        v_blob = values.tobytes()
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO spectra (probe_id, spectrum_type, wavelengths, intensity_values, wavelength_unit, intensity_unit, details) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (probe_id, spec_type, w_blob, v_blob, wavelength_unit, intensity_unit, details),
            )

    def get_spectrum(self, probe_id: int, spec_type: str) -> Optional[tuple]:
        """Get a spectrum for a probe.

        Parameters
        ----------
        probe_id : int
            Probe identifier.
        spec_type : str
            Spectrum type.

        Returns
        -------
        tuple of np.ndarray or None
            (wavelengths, intensity_values).
        """
        row = self.conn.execute(
            "SELECT wavelengths, intensity_values FROM spectra WHERE probe_id = ? AND spectrum_type = ?",
            (probe_id, spec_type),
        ).fetchone()
        if row:
            w = np.frombuffer(row["wavelengths"], dtype=np.float64)
            v = np.frombuffer(row["intensity_values"], dtype=np.float64)
            return w, v
        return None

    def get_spectrum_record(self, probe_id: int, spec_type: str) -> Optional[sqlite3.Row]:
        """Return the full spectrum database row.

        Parameters
        ----------
        probe_id : int
            Probe identifier.
        spec_type : str
            Spectrum type.

        Returns
        -------
        sqlite3.Row or None
            Full spectrum row.
        """
        return self.conn.execute(
            "SELECT * FROM spectra WHERE probe_id = ? AND spectrum_type = ?", (probe_id, spec_type)
        ).fetchone()

    def add_analysis_metadata(
        self, analysis_id: str, key: str, value: Any, details: Optional[str] = None
    ):
        """Add or replace user-provided metadata for an analysis.

        Parameters
        ----------
        analysis_id : str
            Analysis identifier.
        key : str
            Metadata key.
        value : Any
            Metadata value.
        details : str, optional
            Optional human-readable details.
        """
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO analysis_metadata (analysis_id, key, value, details) VALUES (?, ?, ?, ?)",
                (analysis_id, key, str(value), details),
            )

    def get_analysis_metadata(self, analysis_id: str) -> Dict[str, str]:
        """Return metadata for an analysis.

        Parameters
        ----------
        analysis_id : str
            Analysis identifier.

        Returns
        -------
        dict of str to str
            Metadata key/value pairs.
        """
        rows = self.conn.execute(
            "SELECT key, value FROM analysis_metadata WHERE analysis_id = ? ORDER BY key",
            (analysis_id,),
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_analysis_metadata(self, analysis_id: str, metadata: Dict[str, Any]):
        """Replace all metadata for an analysis.

        Parameters
        ----------
        analysis_id : str
            Analysis identifier.
        metadata : dict
            Metadata key/value pairs.
        """
        old = self.get_analysis_metadata(analysis_id)
        with self.conn:
            for key in set(old) - set(metadata):
                self.conn.execute(
                    "DELETE FROM analysis_metadata WHERE analysis_id = ? AND key = ?",
                    (analysis_id, key),
                )
            for key, value in metadata.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO analysis_metadata (analysis_id, key, value, details) VALUES (?, ?, ?, ?)",
                    (analysis_id, key, str(value), None),
                )

    def delete_analysis_metadata(self, analysis_id: str, key: str):
        """Delete one metadata entry.

        Parameters
        ----------
        analysis_id : str
            Analysis identifier.
        key : str
            Metadata key.
        """
        with self.conn:
            self.conn.execute(
                "DELETE FROM analysis_metadata WHERE analysis_id = ? AND key = ?",
                (analysis_id, key),
            )

    def add_external_file(
        self,
        file_path: Union[str, Path],
        file_format: Optional[str] = None,
        content_type: Optional[str] = None,
        reference_id: Optional[str] = None,
        details: Optional[str] = None,
    ) -> int:
        """Register an external file without embedding its contents.

        Parameters
        ----------
        file_path : str or Path
            Path to the external file.
        file_format : str, optional
            File format, for example ``tttr`` or ``ptu``.
        content_type : str, optional
            MIME-like content type.
        reference_id : str, optional
            Stable external reference identifier.
        details : str, optional
            Additional details.

        Returns
        -------
        int
            External file identifier.
        """
        path = Path(file_path)
        size = path.stat().st_size if path.exists() else None
        md5 = None
        if path.exists() and path.is_file():
            digest = hashlib.md5()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
            md5 = digest.hexdigest()
        file_uuid = str(uuid.uuid4())
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO ihm_external_files
                   (reference_id, file_path, file_format, content_type, file_size_bytes, md5, uuid, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (reference_id, str(path), file_format, content_type, size, md5, file_uuid, details),
            )
        row = self.conn.execute(
            "SELECT id FROM ihm_external_files WHERE file_path = ? ORDER BY id DESC LIMIT 1",
            (str(path),),
        ).fetchone()
        return row["id"]

    def get_external_file(self, file_id: int) -> Optional[sqlite3.Row]:
        """Return an external file row by id.

        Parameters
        ----------
        file_id : int
            External file identifier.

        Returns
        -------
        sqlite3.Row or None
            External file row.
        """
        return self.conn.execute(
            "SELECT * FROM ihm_external_files WHERE id = ?", (file_id,)
        ).fetchone()

    def add_photon_stream(
        self,
        analysis_id: str,
        file_path: Union[str, Path],
        file_format: Optional[str] = None,
        content_type: Optional[str] = None,
        stream_id: Optional[str] = None,
        detector_id: Optional[str] = None,
        description: Optional[str] = None,
        details: Optional[str] = None,
    ) -> str:
        """Register a large photon stream as an external file reference.

        Parameters
        ----------
        analysis_id : str
            Analysis identifier.
        file_path : str or Path
            Path to the photon stream file.
        file_format : str, optional
            File format.
        content_type : str, optional
            MIME-like content type.
        stream_id : str, optional
            Stable stream identifier.
        detector_id : str, optional
            Detector/channel identifier.
        description : str, optional
            Short description.
        details : str, optional
            Additional details.

        Returns
        -------
        str
            Photon stream identifier.
        """
        external_id = self.add_external_file(file_path, file_format, content_type, details=details)
        if stream_id is None:
            stream_id = f"stream_{external_id}"
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO flr_photon_stream
                   (stream_id, analysis_id, external_file_id, detector_id, description, details)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (stream_id, analysis_id, external_id, detector_id, description, details),
            )
        return stream_id

    def get_photon_streams(self, analysis_id: str) -> List[sqlite3.Row]:
        """Return photon streams registered for an analysis.

        Parameters
        ----------
        analysis_id : str
            Analysis identifier.

        Returns
        -------
        list of sqlite3.Row
            Photon stream rows.
        """
        return self.conn.execute(
            """SELECT ps.*, ef.file_path, ef.file_format, ef.content_type, ef.file_size_bytes
               FROM flr_photon_stream ps
               LEFT JOIN ihm_external_files ef ON ef.id = ps.external_file_id
               WHERE ps.analysis_id = ?
               ORDER BY ps.stream_id""",
            (analysis_id,),
        ).fetchall()

    def add_analysis_data(
        self,
        analysis_id: str,
        data_type: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        data_name: Optional[str] = None,
        x_unit: Optional[str] = None,
        y_unit: Optional[str] = None,
        details: Optional[str] = None,
    ) -> int:
        """Add or replace small analysis data for embedding in mmCIF.

        Parameters
        ----------
        analysis_id : str
            Analysis identifier.
        data_type : str
            Data type, e.g. 'spectrum', 'decay', 'correlation'.
        x_values : np.ndarray
            X-axis values.
        y_values : np.ndarray
            Y-axis values.
        data_name : str, optional
            Human-readable data name.
        x_unit : str, optional
            Unit for x-axis values.
        y_unit : str, optional
            Unit for y-axis values.
        details : str, optional
            Additional details.

        Returns
        -------
        int
            Data row identifier.
        """
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
            self.conn.execute(f"DELETE FROM analysis_data WHERE {where}", params)
            self.conn.execute(
                """INSERT OR REPLACE INTO analysis_data
                   (analysis_id, data_type, data_name, x_values, y_values, x_unit, y_unit, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (analysis_id, data_type, data_name, x_blob, y_blob, x_unit, y_unit, details),
            )
        row = self.conn.execute(
            "SELECT id FROM analysis_data WHERE analysis_id = ? AND data_type = ? AND data_name IS ? ORDER BY id DESC LIMIT 1",
            (analysis_id, data_type, data_name),
        ).fetchone()
        return int(row["id"]) if row else 0

    def get_analysis_data(self, analysis_id: str) -> List[sqlite3.Row]:
        """Return small analysis data that can be embedded in mmCIF.

        Parameters
        ----------
        analysis_id : str
            Analysis identifier.

        Returns
        -------
        list of sqlite3.Row
            Analysis data rows.
        """
        return self.conn.execute(
            "SELECT * FROM analysis_data WHERE analysis_id = ? ORDER BY data_type, data_name, id",
            (analysis_id,),
        ).fetchall()

    def list_samples(self) -> List[sqlite3.Row]:
        """List all known samples with fluorescence-relevant summary fields.

        Returns
        -------
        list of sqlite3.Row
            Sample summary rows.
        """
        return self.conn.execute(
            """SELECT s.sample_id, s.sample_uuid, s.description, s.details,
                      s.num_of_probes, s.solvent_phase, s.sample_condition_id,
                s.entity_assembly_id, s.project_id, s.measured_by_user_id,
                s.measured_by_device_id, s.measured_at,
                COUNT(sp.sample_probe_id) AS mapped_probe_count,
                u.display_name AS measured_by_user,
                d.name AS measured_by_device
               FROM flr_sample AS s
               LEFT JOIN flr_sample_probe AS sp ON sp.sample_id = s.sample_id
               LEFT JOIN flr_sample_users AS u ON u.user_id = s.measured_by_user_id
               LEFT JOIN flr_sample_devices AS d ON d.device_id = s.measured_by_device_id
               GROUP BY s.sample_id
               ORDER BY s.sample_id"""
        ).fetchall()

    def get_sample(self, sample_id: str) -> Optional[sqlite3.Row]:
        """Get a single sample by id.

        Parameters
        ----------
        sample_id : str
            Sample identifier.

        Returns
        -------
        sqlite3.Row or None
        """
        return self.conn.execute(
            """SELECT sample_id, sample_uuid, description, details,
                      num_of_probes, solvent_phase, sample_condition_id,
                      entity_assembly_id, project_id, measured_by_user_id,
                      measured_by_device_id, measured_at
               FROM flr_sample WHERE sample_id = ?""",
            (sample_id,),
        ).fetchone()

    def get_sample_full(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get a sample with condition, entities, mappings, probes, and positions."""
        sample = self.get_sample(sample_id)
        if sample is None:
            return None
        data = dict(sample)
        condition_id = data.get("sample_condition_id")
        data["condition"] = dict(
            self.conn.execute(
                "SELECT * FROM flr_sample_condition WHERE condition_id = ?",
                (condition_id,),
            ).fetchone()
            or {}
        )
        data["entity_assembly"] = dict(
            self.conn.execute(
                "SELECT * FROM flr_entity_assembly WHERE assembly_id = ?",
                (data.get("entity_assembly_id"),),
            ).fetchone()
            or {}
        )
        mappings = self.get_sample_probe_mappings(sample_id=sample_id)
        data["sample_probes"] = [dict(m) for m in mappings]
        data["entities"] = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM entities WHERE entity_id IN "
                "(SELECT DISTINCT entity_id FROM flr_poly_probe_position WHERE id IN "
                "(SELECT poly_probe_position_id FROM flr_sample_probe WHERE sample_id = ?))",
                (sample_id,),
            ).fetchall()
        ]
        data["key_values"] = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM flr_sample_key_value WHERE sample_id = ? ORDER BY key",
                (sample_id,),
            ).fetchall()
        ]
        return data

    def get_sample_key_values(self, sample_id: str) -> list[dict[str, Any]]:
        """Return sample-level key/value metadata."""
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM flr_sample_key_value WHERE sample_id = ? ORDER BY key",
                (sample_id,),
            ).fetchall()
        ]

    def set_sample_key_value(
        self,
        sample_id: str,
        key: str,
        value: Any,
        details: str | None = None,
    ) -> None:
        """Set one sample-level key/value metadata item."""
        if not key:
            raise ValueError("key is required")
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_sample_key_value "
                "(sample_id, key, value, details) VALUES (?, ?, ?, ?)",
                (sample_id, key, "" if value is None else str(value), details),
            )

    def clear_sample_key_values(self, sample_id: str) -> None:
        """Delete all sample-level key/value metadata items."""
        with self.conn:
            self.conn.execute(
                "DELETE FROM flr_sample_key_value WHERE sample_id = ?",
                (sample_id,),
            )

    def add_sample(
        self,
        sample_id: str,
        uuid: Optional[str] = None,
        description: str = "",
        details: str = "",
        num_of_probes: Optional[int] = None,
        solvent_phase: Optional[str] = None,
        sample_condition_id: Optional[str] = None,
        entity_assembly_id: Optional[str] = None,
        project_id: Optional[str] = None,
        measured_by_user_id: Optional[str] = None,
        measured_by_device_id: Optional[str] = None,
        measured_at: Optional[str] = None,
    ):
        """Add or replace a sample record."""
        import uuid as _uuid

        if uuid is None:
            existing = self.get_sample(sample_id)
            uuid = existing.get("sample_uuid") if existing else str(_uuid.uuid4())
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO flr_sample
                   (sample_id, sample_uuid, description, details, num_of_probes,
                    solvent_phase, sample_condition_id, entity_assembly_id,
                    project_id, measured_by_user_id, measured_by_device_id, measured_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sample_id,
                    uuid,
                    description,
                    details,
                    num_of_probes,
                    solvent_phase,
                    sample_condition_id,
                    entity_assembly_id,
                    project_id,
                    measured_by_user_id,
                    measured_by_device_id,
                    measured_at,
                ),
            )

    def update_sample(self, sample_id: str, **kwargs):
        """Update sample fields."""
        if not kwargs:
            return
        allowed = {
            "sample_uuid",
            "description",
            "details",
            "num_of_probes",
            "solvent_phase",
            "sample_condition_id",
            "entity_assembly_id",
        }
        cols, vals = [], []
        for key, value in kwargs.items():
            if key not in allowed:
                raise ValueError(f"Unsupported sample column: {key}")
            cols.append(f"{key} = ?")
            vals.append(value)
        vals.append(sample_id)
        with self.conn:
            self.conn.execute(f"UPDATE flr_sample SET {', '.join(cols)} WHERE sample_id = ?", vals)

    def delete_sample(self, sample_id: str):
        """Delete a sample and its explicit probe mappings."""
        with self.conn:
            self.conn.execute("DELETE FROM flr_sample_probe WHERE sample_id = ?", (sample_id,))
            self.conn.execute("DELETE FROM flr_sample WHERE sample_id = ?", (sample_id,))

    # ── Users ──────────────────────────────────────────────────────────────

    def get_users(self) -> list[sqlite3.Row]:
        """List all registered users.

        Returns
        -------
        list of sqlite3.Row
        """
        return self.conn.execute(
            "SELECT * FROM flr_sample_users ORDER BY user_id"
        ).fetchall()

    def add_user(
        self,
        user_id: str,
        display_name: str,
        email: str | None = None,
        affiliation: str | None = None,
        details: str | None = None,
    ) -> None:
        """Add or replace a user.

        Parameters
        ----------
        user_id : str
            Unique user identifier.
        display_name : str
            Human-readable display name.
        email : str, optional
            Email address.
        affiliation : str, optional
            Institution or lab affiliation.
        details : str, optional
            Additional notes.
        """
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO flr_sample_users
                   (user_id, display_name, email, affiliation, details)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, display_name, email, affiliation, details),
            )

    def delete_user(self, user_id: str) -> None:
        """Delete a user.

        Parameters
        ----------
        user_id : str
            User identifier.
        """
        with self.conn:
            self.conn.execute(
                "DELETE FROM flr_sample_users WHERE user_id = ?", (user_id,)
            )

    # ── Devices ────────────────────────────────────────────────────────────

    def get_devices(self) -> list[sqlite3.Row]:
        """List all registered devices.

        Returns
        -------
        list of sqlite3.Row
        """
        return self.conn.execute(
            "SELECT * FROM flr_sample_devices ORDER BY device_id"
        ).fetchall()

    def add_device(
        self,
        device_id: str,
        name: str,
        device_type: str | None = None,
        model: str | None = None,
        serial_number: str | None = None,
        location: str | None = None,
        owner: str | None = None,
        details: str | None = None,
    ) -> None:
        """Add or replace a device.

        Parameters
        ----------
        device_id : str
            Unique device identifier.
        name : str
            Human-readable device name.
        device_type : str, optional
            Device type (e.g. TCSPC, plate_reader).
        model : str, optional
            Model name/number.
        serial_number : str, optional
            Serial number.
        location : str, optional
            Physical location.
        owner : str, optional
            Responsible person or group.
        details : str, optional
            Additional notes.
        """
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO flr_sample_devices
                   (device_id, name, device_type, model, serial_number,
                    location, owner, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (device_id, name, device_type, model, serial_number,
                 location, owner, details),
            )

    def delete_device(self, device_id: str) -> None:
        """Delete a device.

        Parameters
        ----------
        device_id : str
            Device identifier.
        """
        with self.conn:
            self.conn.execute(
                "DELETE FROM flr_sample_devices WHERE device_id = ?", (device_id,)
            )

    # ── Experiments ──────────────────────────────────────────────────────────

    def add_experiment_type(
        self,
        name: str,
        category: str | None = None,
        description: str | None = None,
        details: str | None = None,
    ) -> int:
        """Add or replace an experiment type.

        Parameters
        ----------
        name : str
            Unique experiment type name.
        category : str, optional
            Broad category such as imaging or single-molecule.
        description : str, optional
            Short description.
        details : str, optional
            Additional notes.

        Returns
        -------
        int
            Experiment type identifier.
        """
        if not name:
            raise ValueError("experiment type name is required")
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO flr_experiment_type
                   (name, category, description, details)
                   VALUES (?, ?, ?, ?)""",
                (name, category, description, details),
            )
        row = self.conn.execute(
            "SELECT type_id FROM flr_experiment_type WHERE name = ?", (name,)
        ).fetchone()
        return int(row["type_id"])

    def get_experiment_types(self) -> list[sqlite3.Row]:
        """List experiment types.

        Returns
        -------
        list of sqlite3.Row
            Experiment type rows.
        """
        return self.conn.execute(
            "SELECT * FROM flr_experiment_type ORDER BY category, name"
        ).fetchall()

    def delete_experiment_type(self, type_id: int) -> None:
        """Delete an experiment type.

        Parameters
        ----------
        type_id : int
            Experiment type identifier.
        """
        with self.conn:
            self.conn.execute(
                "DELETE FROM flr_experiment_type WHERE type_id = ?", (type_id,)
            )

    def add_experiment(
        self,
        experiment_id: str,
        type_id: int | None = None,
        sample_id: str | None = None,
        project_id: str | None = None,
        measured_by_user_id: str | None = None,
        measured_by_device_id: str | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        status: str | None = None,
        details: str | None = None,
    ) -> None:
        """Add or replace an experiment record.

        Parameters
        ----------
        experiment_id : str
            Unique experiment identifier.
        type_id : int, optional
            Experiment type identifier.
        sample_id : str, optional
            Sample identifier.
        project_id : str, optional
            Project identifier.
        measured_by_user_id : str, optional
            Operator/user identifier.
        measured_by_device_id : str, optional
            Device/instrument identifier.
        started_at : str, optional
            ISO-like start timestamp.
        ended_at : str, optional
            ISO-like end timestamp.
        status : str, optional
            Experiment status.
        details : str, optional
            Additional notes.
        """
        if not experiment_id:
            raise ValueError("experiment_id is required")
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO flr_experiment
                   (experiment_id, type_id, sample_id, project_id, measured_by_user_id,
                    measured_by_device_id, started_at, ended_at, status, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    type_id,
                    sample_id,
                    project_id,
                    measured_by_user_id,
                    measured_by_device_id,
                    started_at,
                    ended_at,
                    status,
                    details,
                ),
            )

    def get_experiment(self, experiment_id: str) -> sqlite3.Row | None:
        """Return one experiment row.

        Parameters
        ----------
        experiment_id : str
            Experiment identifier.

        Returns
        -------
        sqlite3.Row or None
            Experiment row.
        """
        return self.conn.execute(
            """SELECT e.*, et.name AS experiment_type, et.category AS experiment_category,
                      s.description AS sample_description, u.display_name AS measured_by_user,
                      d.name AS measured_by_device
                 FROM flr_experiment AS e
                 LEFT JOIN flr_experiment_type AS et ON et.type_id = e.type_id
                 LEFT JOIN flr_sample AS s ON s.sample_id = e.sample_id
                 LEFT JOIN flr_sample_users AS u ON u.user_id = e.measured_by_user_id
                 LEFT JOIN flr_sample_devices AS d ON d.device_id = e.measured_by_device_id
                WHERE e.experiment_id = ?""",
            (experiment_id,),
        ).fetchone()

    def get_experiments(
        self,
        sample_id: str | None = None,
        project_id: str | None = None,
        type_id: int | None = None,
    ) -> list[sqlite3.Row]:
        """List experiments with type, sample, user, and device labels.

        Parameters
        ----------
        sample_id : str, optional
            Filter by sample identifier.
        project_id : str, optional
            Filter by project identifier.
        type_id : int, optional
            Filter by experiment type identifier.

        Returns
        -------
        list of sqlite3.Row
            Experiment rows.
        """
        query = (
            "SELECT e.*, et.name AS experiment_type, et.category AS experiment_category, "
            "s.description AS sample_description, u.display_name AS measured_by_user, "
            "d.name AS measured_by_device "
            "FROM flr_experiment AS e "
            "LEFT JOIN flr_experiment_type AS et ON et.type_id = e.type_id "
            "LEFT JOIN flr_sample AS s ON s.sample_id = e.sample_id "
            "LEFT JOIN flr_sample_users AS u ON u.user_id = e.measured_by_user_id "
            "LEFT JOIN flr_sample_devices AS d ON d.device_id = e.measured_by_device_id "
            "WHERE 1=1"
        )
        params: list[Any] = []
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

    def get_experiment_full(self, experiment_id: str) -> dict[str, Any] | None:
        """Return an experiment with key/value metadata and data records.

        Parameters
        ----------
        experiment_id : str
            Experiment identifier.

        Returns
        -------
        dict or None
            Experiment data with nested ``key_values`` and ``data`` lists.
        """
        row = self.get_experiment(experiment_id)
        if row is None:
            return None
        data = dict(row)
        data["key_values"] = [
            dict(kv)
            for kv in self.conn.execute(
                "SELECT * FROM flr_experiment_key_value WHERE experiment_id = ? ORDER BY key",
                (experiment_id,),
            ).fetchall()
        ]
        data["data"] = [dict(item) for item in self.get_experiment_data(experiment_id)]
        return data

    def set_experiment_key_value(
        self,
        experiment_id: str,
        key: str,
        value: Any,
        details: str | None = None,
    ) -> None:
        """Set one experiment-level key/value metadata item."""
        if not key:
            raise ValueError("key is required")
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_experiment_key_value "
                "(experiment_id, key, value, details) VALUES (?, ?, ?, ?)",
                (experiment_id, key, "" if value is None else str(value), details),
            )

    def clear_experiment_key_values(self, experiment_id: str) -> None:
        """Delete all experiment-level key/value metadata items."""
        with self.conn:
            self.conn.execute(
                "DELETE FROM flr_experiment_key_value WHERE experiment_id = ?",
                (experiment_id,),
            )

    def add_experiment_data(
        self,
        experiment_id: str,
        data_type: str,
        storage_mode: str,
        file_path: str | None = None,
        url: str | None = None,
        folder_path: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        checksum: str | None = None,
        data_json: str | None = None,
        data_blob: bytes | None = None,
        reading_options_json: str | None = None,
        details: str | None = None,
    ) -> int:
        """Add embedded or linked experiment data.

        Parameters
        ----------
        experiment_id : str
            Experiment identifier.
        data_type : str
            Data kind, e.g. fcs, tcspc, spectra, flim, alex, mfd.
        storage_mode : str
            One of ``embedded``, ``link``, or ``folder``.
        file_path : str, optional
            Local raw data file path.
        url : str, optional
            Remote raw data URL.
        folder_path : str, optional
            Local raw data folder path.
        mime_type : str, optional
            MIME/content type.
        size_bytes : int, optional
            File size in bytes.
        checksum : str, optional
            Checksum string.
        data_json : str, optional
            JSON payload for small embedded data.
        data_blob : bytes, optional
            Binary payload for small embedded data.
        details : str, optional
            Additional notes.

        Returns
        -------
        int
            Experiment data identifier.
        """
        if not data_type:
            raise ValueError("data_type is required")
        if not storage_mode:
            raise ValueError("storage_mode is required")
        with self.conn:
            self.conn.execute(
                """INSERT INTO flr_experiment_data
                   (experiment_id, data_type, storage_mode, file_path, url, folder_path,
                    mime_type, size_bytes, checksum, data_json, data_blob, reading_options_json,
                    details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    data_type,
                    storage_mode,
                    file_path,
                    url,
                    folder_path,
                    mime_type,
                    size_bytes,
                    checksum,
                    data_json,
                    data_blob,
                    reading_options_json,
                    details,
                ),
            )
            return int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def get_experiment_data(self, experiment_id: str) -> list[sqlite3.Row]:
        """Return experiment data records.

        Parameters
        ----------
        experiment_id : str
            Experiment identifier.

        Returns
        -------
        list of sqlite3.Row
            Experiment data rows.
        """
        return self.conn.execute(
            "SELECT * FROM flr_experiment_data WHERE experiment_id = ? ORDER BY data_type, data_id",
            (experiment_id,),
        ).fetchall()

    def update_experiment_data(
        self,
        data_id: int,
        experiment_id: str,
        data_type: str,
        storage_mode: str,
        file_path: str | None = None,
        url: str | None = None,
        folder_path: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        checksum: str | None = None,
        data_json: str | None = None,
        data_blob: bytes | None = None,
        reading_options_json: str | None = None,
        details: str | None = None,
    ) -> None:
        """Update one experiment data record."""
        with self.conn:
            self.conn.execute(
                """UPDATE flr_experiment_data
                   SET experiment_id=?, data_type=?, storage_mode=?, file_path=?, url=?,
                       folder_path=?, mime_type=?, size_bytes=?, checksum=?, data_json=?,
                       data_blob=?, reading_options_json=?, details=?
                   WHERE data_id=?""",
                (
                    experiment_id,
                    data_type,
                    storage_mode,
                    file_path,
                    url,
                    folder_path,
                    mime_type,
                    size_bytes,
                    checksum,
                    data_json,
                    data_blob,
                    reading_options_json,
                    details,
                    data_id,
                ),
            )

    def delete_experiment_data(self, data_id: int) -> None:
        """Delete one experiment data record.

        Parameters
        ----------
        data_id : int
            Experiment data identifier.
        """
        with self.conn:
            self.conn.execute(
                "DELETE FROM flr_experiment_data WHERE data_id = ?", (data_id,)
            )

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment and its data/metadata through cascades."""
        with self.conn:
            self.conn.execute(
                "DELETE FROM flr_experiment WHERE experiment_id = ?", (experiment_id,)
            )

    # ── fdb4chembio raw/process/product provenance ─────────────────────────

    def add_raw_data_reference(
        self,
        experiment_id: str,
        data_type: str,
        storage_mode: str,
        raw_data_id: str | None = None,
        file_path: str | None = None,
        url: str | None = None,
        folder_path: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        checksum: str | None = None,
        checksum_algorithm: str = "sha256",
        header_metadata: dict[str, Any] | None = None,
        detector_mapping: dict[str, Any] | None = None,
        acquisition_software: str | None = None,
        acquisition_software_version: str | None = None,
        acquired_at: str | None = None,
        validation_status: str = "unvalidated",
        validation_message: str | None = None,
    ) -> str:
        """Register a raw measured-data reference.

        Parameters
        ----------
        experiment_id : str
            Experiment that owns the raw data.
        data_type : str
            Raw-data type such as ``TTTR``, ``PTU``, ``SPC``, or ``BH``.
        storage_mode : str
            Storage location mode such as ``local_file``, ``url``, or ``folder``.
        raw_data_id : str, optional
            Stable identifier. A UUID-backed identifier is generated when omitted.

        Returns
        -------
        str
            Raw-data identifier.
        """
        if not experiment_id:
            raise ValueError("experiment_id is required")
        if not data_type:
            raise ValueError("data_type is required")
        if not storage_mode:
            raise ValueError("storage_mode is required")
        raw_data_id = raw_data_id or f"raw_{uuid.uuid4()}"
        now = _utc_now()
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO fdb_raw_data
                   (raw_data_id, experiment_id, data_type, storage_mode, file_path, url,
                    folder_path, mime_type, size_bytes, checksum, checksum_algorithm,
                    header_metadata_json, detector_mapping_json, acquisition_software,
                    acquisition_software_version, acquired_at, validation_status,
                    validation_message, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    raw_data_id,
                    experiment_id,
                    data_type,
                    storage_mode,
                    file_path,
                    url,
                    folder_path,
                    mime_type,
                    size_bytes,
                    checksum,
                    checksum_algorithm,
                    _json_dumps(header_metadata),
                    _json_dumps(detector_mapping),
                    acquisition_software,
                    acquisition_software_version,
                    acquired_at,
                    validation_status,
                    validation_message,
                    now,
                    now,
                ),
            )
        self.add_provenance_edge(
            "experiment",
            experiment_id,
            "raw_data",
            raw_data_id,
            "has_raw_data",
            checksum_snapshot={"raw_data": checksum} if checksum else None,
        )
        return raw_data_id

    def get_raw_data(self, raw_data_id: str) -> sqlite3.Row | None:
        """Return one raw-data reference.

        Parameters
        ----------
        raw_data_id : str
            Raw-data identifier.

        Returns
        -------
        sqlite3.Row or None
            Raw-data row.
        """
        return self.conn.execute(
            "SELECT * FROM fdb_raw_data WHERE raw_data_id = ?", (raw_data_id,)
        ).fetchone()

    def get_raw_data_references(
        self,
        experiment_id: str | None = None,
        data_type: str | None = None,
    ) -> list[sqlite3.Row]:
        """List raw-data references.

        Parameters
        ----------
        experiment_id : str, optional
            Restrict rows to one experiment.
        data_type : str, optional
            Restrict rows to one raw-data type.

        Returns
        -------
        list of sqlite3.Row
            Matching raw-data rows.
        """
        query = "SELECT * FROM fdb_raw_data WHERE 1=1"
        params: list[Any] = []
        if experiment_id is not None:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if data_type is not None:
            query += " AND data_type = ?"
            params.append(data_type)
        query += " ORDER BY created_at, raw_data_id"
        return self.conn.execute(query, params).fetchall()

    def add_processing_run(
        self,
        experiment_id: str,
        processing_type: str = "burst_selection",
        processing_id: str | None = None,
        input_raw_data_ids: list[str] | None = None,
        settings: dict[str, Any] | None = None,
        selected_setup_name: str | None = None,
        detector_definitions: dict[str, Any] | None = None,
        pie_window_definitions: dict[str, Any] | None = None,
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
        file_count: int | None = None,
        photon_count: int | None = None,
        selected_photon_count: int | None = None,
        burst_count: int | None = None,
    ) -> str:
        """Register a repeatable processing run.

        Parameters
        ----------
        experiment_id : str
            Experiment that owns the processing run.
        processing_type : str
            Processing kind. Phase 1 uses ``burst_selection``.
        processing_id : str, optional
            Stable identifier. A UUID-backed identifier is generated when omitted.
        input_raw_data_ids : list of str, optional
            Raw data consumed by the run.

        Returns
        -------
        str
            Processing-run identifier.
        """
        if not experiment_id:
            raise ValueError("experiment_id is required")
        if not processing_type:
            raise ValueError("processing_type is required")
        processing_id = processing_id or f"proc_{uuid.uuid4()}"
        input_raw_data_ids = input_raw_data_ids or []
        settings_hash = _json_hash(settings)
        now = _utc_now()
        runtime_environment = runtime_environment or self.default_runtime_environment()
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO fdb_processing_run
                   (processing_id, processing_type, experiment_id, settings_json,
                    settings_hash, selected_setup_name, detector_definitions_json,
                    pie_window_definitions_json, operator_user_id, software_package,
                    software_module, software_version, runtime_environment_json,
                    started_at, ended_at, status, error_message, traceback_summary,
                    file_count, photon_count, selected_photon_count, burst_count,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    processing_id,
                    processing_type,
                    experiment_id,
                    _json_dumps(settings),
                    settings_hash,
                    selected_setup_name,
                    _json_dumps(detector_definitions),
                    _json_dumps(pie_window_definitions),
                    operator_user_id,
                    software_package,
                    software_module,
                    software_version,
                    _json_dumps(runtime_environment),
                    started_at,
                    ended_at,
                    status,
                    error_message,
                    traceback_summary,
                    file_count,
                    photon_count,
                    selected_photon_count,
                    burst_count,
                    now,
                    now,
                ),
            )
            self.conn.execute(
                "DELETE FROM fdb_processing_input WHERE processing_id = ?",
                (processing_id,),
            )
            for ordinal, raw_data_id in enumerate(input_raw_data_ids):
                self.conn.execute(
                    """INSERT OR REPLACE INTO fdb_processing_input
                       (processing_id, raw_data_id, ordinal) VALUES (?, ?, ?)""",
                    (processing_id, raw_data_id, ordinal),
                )
        for raw_data_id in input_raw_data_ids:
            raw_row = self.get_raw_data(raw_data_id)
            self.add_provenance_edge(
                "raw_data",
                raw_data_id,
                "processing_run",
                processing_id,
                "input_to",
                processing_id=processing_id,
                settings_hash=settings_hash,
                software_version=software_version,
                checksum_snapshot={
                    "raw_data": raw_row["checksum"] if raw_row else None,
                    "settings": settings_hash,
                },
            )
        return processing_id

    def update_processing_run_status(
        self,
        processing_id: str,
        status: str,
        ended_at: str | None = None,
        error_message: str | None = None,
        traceback_summary: str | None = None,
        photon_count: int | None = None,
        selected_photon_count: int | None = None,
        burst_count: int | None = None,
    ) -> None:
        """Update status and aggregate counts for a processing run.

        Parameters
        ----------
        processing_id : str
            Processing-run identifier.
        status : str
            New status value.
        ended_at : str, optional
            Completion timestamp.
        """
        with self.conn:
            self.conn.execute(
                """UPDATE fdb_processing_run
                   SET status = ?, ended_at = COALESCE(?, ended_at),
                       error_message = COALESCE(?, error_message),
                       traceback_summary = COALESCE(?, traceback_summary),
                       photon_count = COALESCE(?, photon_count),
                       selected_photon_count = COALESCE(?, selected_photon_count),
                       burst_count = COALESCE(?, burst_count),
                       updated_at = ?
                   WHERE processing_id = ?""",
                (
                    status,
                    ended_at,
                    error_message,
                    traceback_summary,
                    photon_count,
                    selected_photon_count,
                    burst_count,
                    _utc_now(),
                    processing_id,
                ),
            )

    def get_processing_run(self, processing_id: str) -> sqlite3.Row | None:
        """Return one processing run.

        Parameters
        ----------
        processing_id : str
            Processing-run identifier.

        Returns
        -------
        sqlite3.Row or None
            Processing-run row.
        """
        return self.conn.execute(
            "SELECT * FROM fdb_processing_run WHERE processing_id = ?",
            (processing_id,),
        ).fetchone()

    def get_processing_runs(
        self,
        experiment_id: str | None = None,
        processing_type: str | None = None,
        status: str | None = None,
    ) -> list[sqlite3.Row]:
        """List processing runs.

        Parameters
        ----------
        experiment_id : str, optional
            Restrict rows to one experiment.
        processing_type : str, optional
            Restrict rows to one processing type.
        status : str, optional
            Restrict rows to one status.

        Returns
        -------
        list of sqlite3.Row
            Matching processing runs.
        """
        query = "SELECT * FROM fdb_processing_run WHERE 1=1"
        params: list[Any] = []
        if experiment_id is not None:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if processing_type is not None:
            query += " AND processing_type = ?"
            params.append(processing_type)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY started_at, created_at, processing_id"
        return self.conn.execute(query, params).fetchall()

    def get_processing_run_full(self, processing_id: str) -> dict[str, Any] | None:
        """Return a processing run with inputs, products, and edges.

        Parameters
        ----------
        processing_id : str
            Processing-run identifier.

        Returns
        -------
        dict or None
            Expanded processing-run record.
        """
        run = _row_to_dict(self.get_processing_run(processing_id))
        if run is None:
            return None
        run["settings"] = _json_loads(run.pop("settings_json", None))
        run["detector_definitions"] = _json_loads(
            run.pop("detector_definitions_json", None)
        )
        run["pie_window_definitions"] = _json_loads(
            run.pop("pie_window_definitions_json", None)
        )
        run["runtime_environment"] = _json_loads(
            run.pop("runtime_environment_json", None)
        )
        run["input_raw_data"] = [
            dict(row)
            for row in self.conn.execute(
                """SELECT rd.*
                   FROM fdb_processing_input AS pi
                   JOIN fdb_raw_data AS rd ON rd.raw_data_id = pi.raw_data_id
                   WHERE pi.processing_id = ?
                   ORDER BY pi.ordinal, rd.raw_data_id""",
                (processing_id,),
            ).fetchall()
        ]
        run["processed_data"] = [
            dict(row) for row in self.get_processed_data_products(processing_id=processing_id)
        ]
        run["provenance_edges"] = [
            dict(row) for row in self.get_provenance_edges(processing_id=processing_id)
        ]
        return run

    def add_processed_data_product(
        self,
        processing_id: str,
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
    ) -> str:
        """Register a product generated by a processing run.

        Parameters
        ----------
        processing_id : str
            Processing run that produced the data.
        product_type : str
            Product type such as ``bur``, ``hdf5``, ``zip``, or ``archive_manifest``.
        storage_mode : str
            Storage location mode.

        Returns
        -------
        str
            Processed-data identifier.
        """
        if not processing_id:
            raise ValueError("processing_id is required")
        if not product_type:
            raise ValueError("product_type is required")
        if not storage_mode:
            raise ValueError("storage_mode is required")
        processed_data_id = processed_data_id or f"prod_{uuid.uuid4()}"
        now = _utc_now()
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO fdb_processed_data
                   (processed_data_id, processing_id, product_type, storage_mode,
                    file_path, url, folder_path, mime_type, size_bytes, checksum,
                    checksum_algorithm, row_count, product_summary_json, metadata_json,
                    data_json, data_blob, validation_status, validation_message,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    processed_data_id,
                    processing_id,
                    product_type,
                    storage_mode,
                    file_path,
                    url,
                    folder_path,
                    mime_type,
                    size_bytes,
                    checksum,
                    checksum_algorithm,
                    row_count,
                    _json_dumps(product_summary),
                    _json_dumps(metadata),
                    data_json,
                    data_blob,
                    validation_status,
                    validation_message,
                    now,
                    now,
                ),
            )
        run = self.get_processing_run(processing_id)
        self.add_provenance_edge(
            "processing_run",
            processing_id,
            "processed_data",
            processed_data_id,
            "produced",
            processing_id=processing_id,
            settings_hash=run["settings_hash"] if run else None,
            software_version=run["software_version"] if run else None,
            checksum_snapshot={"processed_data": checksum} if checksum else None,
        )
        return processed_data_id

    def get_processed_data(self, processed_data_id: str) -> sqlite3.Row | None:
        """Return one processed-data product.

        Parameters
        ----------
        processed_data_id : str
            Processed-data identifier.

        Returns
        -------
        sqlite3.Row or None
            Processed-data row.
        """
        return self.conn.execute(
            "SELECT * FROM fdb_processed_data WHERE processed_data_id = ?",
            (processed_data_id,),
        ).fetchone()

    def get_processed_data_products(
        self,
        processing_id: str | None = None,
        product_type: str | None = None,
    ) -> list[sqlite3.Row]:
        """List processed-data products.

        Parameters
        ----------
        processing_id : str, optional
            Restrict rows to one processing run.
        product_type : str, optional
            Restrict rows to one product type.

        Returns
        -------
        list of sqlite3.Row
            Matching processed-data rows.
        """
        query = "SELECT * FROM fdb_processed_data WHERE 1=1"
        params: list[Any] = []
        if processing_id is not None:
            query += " AND processing_id = ?"
            params.append(processing_id)
        if product_type is not None:
            query += " AND product_type = ?"
            params.append(product_type)
        query += " ORDER BY created_at, processed_data_id"
        return self.conn.execute(query, params).fetchall()

    def add_provenance_edge(
        self,
        source_node_type: str,
        source_node_id: str,
        target_node_type: str,
        target_node_id: str,
        relationship_type: str,
        processing_id: str | None = None,
        settings_hash: str | None = None,
        timestamp: str | None = None,
        software_version: str | None = None,
        checksum_snapshot: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Register one provenance graph edge.

        Parameters
        ----------
        source_node_type : str
            Source node type.
        source_node_id : str
            Source node identifier.
        target_node_type : str
            Target node type.
        target_node_id : str
            Target node identifier.
        relationship_type : str
            Relationship label.

        Returns
        -------
        int
            Provenance edge identifier.
        """
        for name, value in {
            "source_node_type": source_node_type,
            "source_node_id": source_node_id,
            "target_node_type": target_node_type,
            "target_node_id": target_node_id,
            "relationship_type": relationship_type,
        }.items():
            if not value:
                raise ValueError(f"{name} is required")
        with self.conn:
            self.conn.execute(
                """INSERT INTO fdb_provenance_edge
                   (source_node_type, source_node_id, target_node_type, target_node_id,
                    relationship_type, processing_id, settings_hash, timestamp,
                    software_version, checksum_snapshot_json, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    source_node_type,
                    source_node_id,
                    target_node_type,
                    target_node_id,
                    relationship_type,
                    processing_id,
                    settings_hash,
                    timestamp or _utc_now(),
                    software_version,
                    _json_dumps(checksum_snapshot),
                    _json_dumps(metadata),
                ),
            )
            return int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])

    def get_provenance_edges(
        self,
        source_node_type: str | None = None,
        source_node_id: str | None = None,
        target_node_type: str | None = None,
        target_node_id: str | None = None,
        relationship_type: str | None = None,
        processing_id: str | None = None,
    ) -> list[sqlite3.Row]:
        """List provenance edges.

        Parameters
        ----------
        source_node_type : str, optional
            Restrict by source type.
        source_node_id : str, optional
            Restrict by source identifier.
        target_node_type : str, optional
            Restrict by target type.
        target_node_id : str, optional
            Restrict by target identifier.

        Returns
        -------
        list of sqlite3.Row
            Matching provenance edges.
        """
        query = "SELECT * FROM fdb_provenance_edge WHERE 1=1"
        filters = {
            "source_node_type": source_node_type,
            "source_node_id": source_node_id,
            "target_node_type": target_node_type,
            "target_node_id": target_node_id,
            "relationship_type": relationship_type,
            "processing_id": processing_id,
        }
        params: list[Any] = []
        for key, value in filters.items():
            if value is not None:
                query += f" AND {key} = ?"
                params.append(value)
        query += " ORDER BY timestamp, edge_id"
        return self.conn.execute(query, params).fetchall()

    def trace_processed_data(self, processed_data_id: str) -> dict[str, Any] | None:
        """Trace a product back to its processing run and raw inputs.

        Parameters
        ----------
        processed_data_id : str
            Processed-data identifier.

        Returns
        -------
        dict or None
            Provenance trace.
        """
        product = _row_to_dict(self.get_processed_data(processed_data_id))
        if product is None:
            return None
        processing_id = product["processing_id"]
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

    def export_burst_processing_manifest(self, processing_id: str) -> dict[str, Any]:
        """Build a JSON-compatible manifest for one burst-processing run.

        Parameters
        ----------
        processing_id : str
            Processing-run identifier.

        Returns
        -------
        dict
            Archive manifest with raw inputs, settings, products, and edges.
        """
        run = self.get_processing_run_full(processing_id)
        if run is None:
            raise KeyError(f"processing run not found: {processing_id}")
        experiment = _row_to_dict(self.get_experiment(str(run["experiment_id"])))
        products = [
            self._decode_processed_data_row(row)
            for row in self.get_processed_data_products(processing_id=processing_id)
        ]
        raw_data = [self._decode_raw_data_row(row) for row in run["input_raw_data"]]
        edges = [
            self._decode_provenance_edge_row(row)
            for row in self.get_provenance_edges(processing_id=processing_id)
        ]
        return {
            "schema": "fdb4chembio.burst_processing_manifest.v1",
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
        """Register an archive manifest as a processed-data product.

        Parameters
        ----------
        processing_id : str
            Processing run represented by the manifest.
        manifest : dict, optional
            Manifest content. It is generated when omitted.
        output_path : str, optional
            Optional path where the manifest was written.

        Returns
        -------
        str
            Processed-data identifier for the manifest record.
        """
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
            other_id = product["processed_data_id"]
            if other_id == product_id:
                continue
            self.add_provenance_edge(
                "processed_data",
                other_id,
                "processed_data",
                product_id,
                "included_in",
                processing_id=processing_id,
                settings_hash=run["settings_hash"] if run else None,
                software_version=run["software_version"] if run else None,
                checksum_snapshot={
                    "source": product["checksum"],
                    "archive_manifest": hashlib.sha256(data_json.encode("utf-8")).hexdigest()
                    if data_json
                    else None,
                },
            )
        return product_id

    @staticmethod
    def default_runtime_environment() -> dict[str, Any]:
        """Return a compact runtime environment summary.

        Returns
        -------
        dict
            Python and platform details suitable for JSON-RPC.
        """
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        }

    @staticmethod
    def _decode_raw_data_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        """Decode JSON fields in a raw-data row.

        Parameters
        ----------
        row : sqlite3.Row or dict
            Raw-data record.

        Returns
        -------
        dict
            JSON-compatible row.
        """
        data = dict(row)
        data["header_metadata"] = _json_loads(data.pop("header_metadata_json", None))
        data["detector_mapping"] = _json_loads(data.pop("detector_mapping_json", None))
        return data

    @staticmethod
    def _decode_processed_data_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        """Decode JSON fields in a processed-data row.

        Parameters
        ----------
        row : sqlite3.Row or dict
            Processed-data record.

        Returns
        -------
        dict
            JSON-compatible row.
        """
        data = dict(row)
        data["product_summary"] = _json_loads(data.pop("product_summary_json", None))
        data["metadata"] = _json_loads(data.pop("metadata_json", None))
        if data.get("data_json"):
            try:
                data["data"] = _json_loads(data["data_json"])
            except json.JSONDecodeError:
                data["data"] = data["data_json"]
        return data

    @staticmethod
    def _decode_provenance_edge_row(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        """Decode JSON fields in a provenance edge row.

        Parameters
        ----------
        row : sqlite3.Row or dict
            Provenance edge record.

        Returns
        -------
        dict
            JSON-compatible row.
        """
        data = dict(row)
        data["checksum_snapshot"] = _json_loads(data.pop("checksum_snapshot_json", None))
        data["metadata"] = _json_loads(data.pop("metadata_json", None))
        return data

    def add_sample_condition(
        self,
        condition_id: str,
        ph: Optional[float] = None,
        temperature: Optional[float] = None,
        ionic_strength: Optional[float] = None,
        buffer_composition: Optional[str] = None,
        details: Optional[str] = None,
    ):
        """Add or replace a typed sample condition."""
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO flr_sample_condition
                   (condition_id, ph, temperature, ionic_strength, buffer_composition, details)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (condition_id, ph, temperature, ionic_strength, buffer_composition, details),
            )

    def get_sample_probe_mappings(
        self,
        sample_id: Optional[str] = None,
        probe_id: Optional[int] = None,
    ) -> List[sqlite3.Row]:
        """Return explicit sample-probe mappings."""
        query = (
            "SELECT sp.*, p.chromophore_name, p.category AS probe_category, "
            "pp.entity_id, pp.asym_id, pp.residue_number, pp.residue_name, "
            "pp.description AS position_description "
            "FROM flr_sample_probe AS sp "
            "LEFT JOIN probes AS p ON p.probe_id = sp.probe_id "
            "LEFT JOIN flr_poly_probe_position AS pp "
            "ON pp.id = sp.poly_probe_position_id WHERE 1=1"
        )
        params = []
        if sample_id is not None:
            query += " AND sp.sample_id = ?"
            params.append(sample_id)
        if probe_id is not None:
            query += " AND sp.probe_id = ?"
            params.append(probe_id)
        query += " ORDER BY sp.sample_id, sp.sample_probe_id"
        return self.conn.execute(query, params).fetchall()

    def add_sample_probe(
        self,
        sample_id: str,
        probe_id: int,
        poly_probe_position_id: Optional[int] = None,
        fluorophore_type: str = "unspecified",
        description: str = "",
        sample_probe_id: Optional[int] = None,
    ) -> int:
        """Add or replace an explicit sample-probe mapping."""
        if sample_probe_id is None:
            if poly_probe_position_id is None:
                row = self.conn.execute(
                    "SELECT sample_probe_id FROM flr_sample_probe WHERE sample_id=? AND probe_id=? AND poly_probe_position_id IS NULL",
                    (sample_id, probe_id),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT sample_probe_id FROM flr_sample_probe WHERE sample_id=? AND probe_id=? AND poly_probe_position_id=?",
                    (sample_id, probe_id, poly_probe_position_id),
                ).fetchone()
            sample_probe_id = int(row["sample_probe_id"]) if row else None
        with self.conn:
            if sample_probe_id is None:
                self.conn.execute(
                    """INSERT INTO flr_sample_probe
                       (sample_id, probe_id, poly_probe_position_id, fluorophore_type, description)
                       VALUES (?, ?, ?, ?, ?)""",
                    (sample_id, probe_id, poly_probe_position_id, fluorophore_type, description),
                )
                if poly_probe_position_id is None:
                    row = self.conn.execute(
                        "SELECT sample_probe_id FROM flr_sample_probe WHERE sample_id=? AND probe_id=? AND poly_probe_position_id IS NULL",
                        (sample_id, probe_id),
                    ).fetchone()
                else:
                    row = self.conn.execute(
                        "SELECT sample_probe_id FROM flr_sample_probe WHERE sample_id=? AND probe_id=? AND poly_probe_position_id=?",
                        (sample_id, probe_id, poly_probe_position_id),
                    ).fetchone()
                return int(row["sample_probe_id"])
            self.conn.execute(
                """INSERT OR REPLACE INTO flr_sample_probe
                   (sample_probe_id, sample_id, probe_id, poly_probe_position_id, fluorophore_type, description)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    sample_probe_id,
                    sample_id,
                    probe_id,
                    poly_probe_position_id,
                    fluorophore_type,
                    description,
                ),
            )
            return sample_probe_id

    def clear_sample_probes(self, sample_id: str):
        """Delete all explicit probe mappings for a sample."""
        with self.conn:
            self.conn.execute("DELETE FROM flr_sample_probe WHERE sample_id = ?", (sample_id,))

    def delete_sample_probe(self, sample_probe_id: int):
        """Delete one explicit sample-probe mapping."""
        with self.conn:
            self.conn.execute(
                "DELETE FROM flr_sample_probe WHERE sample_probe_id = ?", (sample_probe_id,)
            )

    def add_entity_assembly(self, assembly_id: str, description: str = "", details: str = ""):
        """Add or replace an entity assembly."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_entity_assembly (assembly_id, description, details) VALUES (?, ?, ?)",
                (assembly_id, description, details),
            )

    def get_entity_assemblies(self) -> List[sqlite3.Row]:
        """Return entity assemblies."""
        return self.conn.execute(
            "SELECT * FROM flr_entity_assembly ORDER BY assembly_id"
        ).fetchall()

    def update_entity_assembly(self, assembly_id: str, description: str = "", details: str = ""):
        """Update an entity assembly."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO flr_entity_assembly (assembly_id, description, details) VALUES (?, ?, ?)",
                (assembly_id, description, details),
            )

    def delete_entity_assembly(self, assembly_id: str):
        """Delete an entity assembly."""
        with self.conn:
            self.conn.execute("DELETE FROM flr_sample WHERE entity_assembly_id = ?", (assembly_id,))
            self.conn.execute(
                "DELETE FROM flr_entity_assembly WHERE assembly_id = ?", (assembly_id,)
            )

    def update_analysis_record(self, analysis_id: str, **kwargs):
        """Update flr_fret_analysis metadata columns.

        Parameters
        ----------
        analysis_id : str
            Analysis identifier.
        **kwargs
            Column/value pairs.
        """
        if not kwargs:
            return
        cols = [f"{k} = ?" for k in kwargs]
        with self.conn:
            if "sample_id" in kwargs and kwargs["sample_id"]:
                self.conn.execute(
                    "INSERT OR IGNORE INTO flr_sample (sample_id) VALUES (?)",
                    (kwargs["sample_id"],),
                )
            self.conn.execute(
                f"INSERT OR IGNORE INTO flr_fret_analysis (analysis_id) VALUES (?)",
                (analysis_id,),
            )
            self.conn.execute(
                f"UPDATE flr_fret_analysis SET {', '.join(cols)} WHERE analysis_id = ?",
                (*kwargs.values(), analysis_id),
            )

    def add_probe_image(
        self, probe_id: int, data: bytes, image_format: str = "png", image_name: str = ""
    ):
        """Add or replace a probe image.

        Parameters
        ----------
        probe_id : int
            Probe identifier.
        data : bytes
            Binary image data.
        image_format : str
            Image format (e.g. 'png').
        image_name : str
            Optional image name.
        """
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO images (probe_id, image_name, image_data, image_format) VALUES (?, ?, ?, ?)",
                (probe_id, image_name, data, image_format),
            )

    def get_images(self, probe_id: int) -> List[tuple]:
        """Get all images for a probe.

        Parameters
        ----------
        probe_id : int
            Probe identifier.

        Returns
        -------
        list of tuple
            (image_name, image_data, image_format) entries.
        """
        return self.conn.execute(
            "SELECT image_name, image_data, image_format FROM images WHERE probe_id = ?",
            (probe_id,),
        ).fetchall()

    def add_entity(self, entity_id: str, **kwargs):
        """Add or replace an entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier.
        **kwargs
            Additional entity fields.
        """
        cols = ["entity_id"]
        vals = [entity_id]
        for k, v in kwargs.items():
            cols.append(k)
            vals.append(v)
        placeholders = ", ".join(["?"] * len(cols))
        with self.conn:
            self.conn.execute(
                f"INSERT OR REPLACE INTO entities ({', '.join(cols)}) VALUES ({placeholders})",
                tuple(vals),
            )

    def get_entities(self) -> List[sqlite3.Row]:
        """Get all entities.

        Returns
        -------
        list of sqlite3.Row
        """
        return self.conn.execute("SELECT * FROM entities").fetchall()

    def get_entity_by_id(self, entity_id: str) -> Optional[sqlite3.Row]:
        """Get an entity by its ID.

        Parameters
        ----------
        entity_id : str
            Entity identifier.

        Returns
        -------
        sqlite3.Row or None
        """
        return self.conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()

    def update_entity(self, entity_id: str, **kwargs):
        """Update entity fields.

        Parameters
        ----------
        entity_id : str
            Entity identifier.
        **kwargs
            Field name/value pairs to update.
        """
        if not kwargs:
            return
        cols = [f"{k} = ?" for k in kwargs.keys()]
        with self.conn:
            self.conn.execute(
                f"UPDATE entities SET {', '.join(cols)} WHERE entity_id = ?",
                (*kwargs.values(), entity_id),
            )

    def delete_entity(self, entity_id: str):
        """Delete an entity and all related data.

        Parameters
        ----------
        entity_id : str
            Entity identifier.
        """
        with self.conn:
            self.conn.execute("DELETE FROM entity_poly_seq WHERE entity_id = ?", (entity_id,))
            self.conn.execute(
                "DELETE FROM flr_poly_probe_position WHERE entity_id = ?", (entity_id,)
            )
            self.conn.execute("DELETE FROM entities WHERE entity_id = ?", (entity_id,))

    def set_sequence(self, entity_id: str, monomers: List[str]):
        """Set the sequence for an entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier.
        monomers : list of str
            List of monomer (residue) identifiers.
        """
        with self.conn:
            self.conn.execute("DELETE FROM entity_poly_seq WHERE entity_id = ?", (entity_id,))
            for i, mon in enumerate(monomers, 1):
                self.conn.execute(
                    "INSERT INTO entity_poly_seq (entity_id, num, mon_id) VALUES (?, ?, ?)",
                    (entity_id, i, mon),
                )

    def get_sequence(self, entity_id: str) -> List[sqlite3.Row]:
        """Get the sequence for an entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier.

        Returns
        -------
        list of sqlite3.Row
        """
        return self.conn.execute(
            "SELECT * FROM entity_poly_seq WHERE entity_id = ? ORDER BY num", (entity_id,)
        ).fetchall()

    def add_poly_probe_position(
        self,
        probe_id: int,
        entity_id: str,
        residue_number: int,
        asym_id: str = "A",
        residue_name: str = None,
        description: str = None,
    ):
        """Record a probe position on a polymer entity.

        Parameters
        ----------
        probe_id : int
            Probe identifier.
        entity_id : str
            Entity identifier.
        residue_number : int
            Residue number.
        asym_id : str
            Asymmetric unit identifier.
        residue_name : str, optional
            Residue name.
        description : str, optional
            Additional description.
        """
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO flr_poly_probe_position (probe_id, entity_id, asym_id, residue_number, residue_name, description)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (probe_id, entity_id, asym_id, residue_number, residue_name, description),
            )

    def get_poly_probe_positions(
        self, probe_id: int = None, entity_id: str = None
    ) -> List[sqlite3.Row]:
        """Get probe positions, optionally filtered.

        Parameters
        ----------
        probe_id : int, optional
            Filter by probe.
        entity_id : str, optional
            Filter by entity.

        Returns
        -------
        list of sqlite3.Row
        """
        query = "SELECT * FROM flr_poly_probe_position WHERE 1=1"
        params = []
        if probe_id is not None:
            query += " AND probe_id = ?"
            params.append(probe_id)
        if entity_id is not None:
            query += " AND entity_id = ?"
            params.append(entity_id)
        return self.conn.execute(query, params).fetchall()

    def delete_poly_probe_position(self, id: int):
        """Delete a probe position record.

        Parameters
        ----------
        id : int
            Position record identifier.
        """
        with self.conn:
            self.conn.execute("DELETE FROM flr_poly_probe_position WHERE id = ?", (id,))

    def search_probes(self, query: str) -> List[sqlite3.Row]:
        """Search probes by name, category, or description.

        Parameters
        ----------
        query : str
            Search string.

        Returns
        -------
        list of sqlite3.Row
        """
        q = f"%{query}%"
        return self.conn.execute(
            "SELECT * FROM probes WHERE chromophore_name LIKE ? OR category LIKE ? OR description LIKE ?",
            (q, q, q),
        ).fetchall()

    def get_standardized_items(self, include_uncurated: bool = True) -> List[Dict[str, Any]]:
        """Get all probes with standardized optical properties.

        Parameters
        ----------
        include_uncurated : bool
            If True, include uncurated probes.

        Returns
        -------
        list of dict
        """
        probes = self.get_probes(curated_only=not include_uncurated)
        result = []
        for p in probes:
            d = dict(p)
            std = self.get_standardized_optical_properties(d["probe_id"])
            d.update(std)
            result.append(d)
        return result

    def get_probe_types_dict(self) -> Dict[int, str]:
        """Get mapping from type_id to display_name.

        Returns
        -------
        dict of int to str
        """
        return {
            r["type_id"]: r["display_name"]
            for r in self.conn.execute("SELECT type_id, display_name FROM probe_types").fetchall()
        }

    def get_standardized_optical_properties(self, probe_id: int) -> Dict[str, Any]:
        """Get canonical optical properties for a probe.

        Parameters
        ----------
        probe_id : int
            Probe identifier.

        Returns
        -------
        dict of str to any
            Keys include abs_max, em_max, qy, ext_coeff, lifetime.
        """
        raw = self.get_optical_properties(probe_id)
        raw_low = {k.lower(): v for k, v in raw.items()}
        std = {}
        for key, aliases in self._PROP_ALIASES.items():
            for alias in aliases:
                if alias.lower() in raw_low:
                    val = raw_low[alias.lower()]
                    try:
                        std[key] = float(val)
                    except (ValueError, TypeError):
                        std[key] = val
                    break
            else:
                std[key] = None
        return std

    def get_probe_full(self, probe_id: int) -> Optional[Dict[str, Any]]:
        """Get complete probe info including optical properties and spectrum flags.

        Parameters
        ----------
        probe_id : int
            Probe identifier.

        Returns
        -------
        dict or None
        """
        probe = self.get_probe_by_id(probe_id)
        if not probe:
            return None
        d = dict(probe)
        d.update(self.get_standardized_optical_properties(probe_id))
        d["has_abs"] = (
            1
            if (
                self.get_spectrum(probe_id, "absorption")
                or self.get_spectrum(probe_id, "excitation")
            )
            else 0
        )
        d["has_em"] = 1 if self.get_spectrum(probe_id, "emission") else 0
        return d

    def validate_probe(self, probe_id: int) -> List[str]:
        """Validate probe data quality.

        Parameters
        ----------
        probe_id : int
            Probe identifier.

        Returns
        -------
        list of str
            List of validation error messages (empty if valid).
        """
        errors = []
        props = self.get_standardized_optical_properties(probe_id)
        if props.get("qy") is not None and not (0 <= props["qy"] <= 1.0):
            errors.append(f"QY {props['qy']} out of range [0, 1]")
        for k in ["abs_max", "em_max"]:
            if props.get(k) is not None and not (200 <= props[k] <= 1000):
                errors.append(f"{k} {props[k]} out of range [200, 1000] nm")
        if not self.get_spectrum(probe_id, "absorption") and not self.get_spectrum(
            probe_id, "excitation"
        ):
            errors.append("Missing absorption/excitation spectrum")
        return errors

    def completeness_score(self, probe_id: int) -> float:
        """Compute a completeness score for a probe.

        Parameters
        ----------
        probe_id : int
            Probe identifier.

        Returns
        -------
        float
            Score between 0 and 1.
        """
        props = self.get_standardized_optical_properties(probe_id)
        found = sum(1 for k in ["abs_max", "em_max", "qy", "ext_coeff"] if props.get(k) is not None)
        has_abs = (
            1
            if (
                self.get_spectrum(probe_id, "absorption")
                or self.get_spectrum(probe_id, "excitation")
            )
            else 0
        )
        has_em = 1 if self.get_spectrum(probe_id, "emission") else 0
        return (found + has_abs + has_em) / 6.0

    def export_flr_cif(
        self,
        path: Union[str, Path, "io.TextIOBase"],
        analysis_id: Optional[str] = None,
        include_extension: bool = True,
    ):
        """Export stored FLR data as a flrCIF-compatible mmCIF file.

        Small data such as spectra and optical properties are embedded as
        ChiSurf extension rows. Large photon streams are exported only as
        external-file references.

        Parameters
        ----------
        path : str, Path, or TextIOBase
            Output mmCIF path or a writable text stream.
        analysis_id : str, optional
            Analysis identifier to export. Defaults to the first stored FLR
            analysis or ``analysis_1``.
        include_extension : bool
            If True, include ChiSurf extension categories for metadata,
            spectra, optical properties, and photon streams.

        Returns
        -------
        Path or str
            Output path, or the generated CIF text if *path* is a stream.
        """
        import io
        import ihm.format

        is_stream = isinstance(path, io.TextIOBase)

        if analysis_id is None:
            row = self.conn.execute(
                "SELECT analysis_id FROM flr_fret_analysis ORDER BY analysis_id LIMIT 1"
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

        sample_row = _row_dict(
            self.conn.execute(
                "SELECT * FROM flr_sample WHERE sample_id = ?", (sample_id,)
            ).fetchone()
        )
        probes = [
            dict(row)
            for row in self.conn.execute("SELECT * FROM probes ORDER BY probe_id").fetchall()
        ]
        positions = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM flr_poly_probe_position ORDER BY id"
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
                        "probe_id": first_probe.get("probe_id")
                        if first_probe is not None
                        else None,
                        "poly_probe_position_id": first_position.get("id")
                        if first_position is not None
                        else None,
                        "chromophore_name": first_probe.get("chromophore_name")
                        if first_probe is not None
                        else None,
                        "fluorophore_type": "unspecified",
                        "description": "ChiSurf legacy sample probe mapping",
                    }
                ]
        distances = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM flr_fret_distance_restraint WHERE analysis_id = ? ORDER BY id",
                (analysis_id,),
            ).fetchall()
        ]
        forster = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM flr_fret_forster_radius ORDER BY id"
            ).fetchall()
        ]
        metadata = self.get_analysis_metadata(analysis_id)
        streams = [dict(row) for row in self.get_photon_streams(analysis_id)]
        external_files = [
            dict(row)
            for row in self.conn.execute("SELECT * FROM ihm_external_files ORDER BY id").fetchall()
        ]
        spectra = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM spectra ORDER BY probe_id, spectrum_type"
            ).fetchall()
        ]
        properties = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM optical_properties ORDER BY probe_id, property_name"
            ).fetchall()
        ]
        analysis_data = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM analysis_data WHERE analysis_id = ? ORDER BY data_type, data_name, id",
                (analysis_id,),
            ).fetchall()
        ]

        def _cif_value(value):
            if value is None:
                return None
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            return value

        def _array_to_text(blob, dtype=np.float64):
            if not blob:
                return ""
            return " ".join(f"{float(v):.8g}" for v in np.frombuffer(blob, dtype=dtype))

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
                ],
            ) as loop:
                for probe in probes:
                    loop.write(
                        **{
                            k: _cif_value(probe.get(k))
                            for k in [
                                "probe_id",
                                "chromophore_name",
                                "reactive_probe_flag",
                                "reactive_probe_name",
                                "probe_origin",
                                "probe_link_type",
                            ]
                        }
                    )

            with writer.loop(
                "_flr_poly_probe_position",
                [
                    "id",
                    "entity_id",
                    "entity_description",
                    "asym_id",
                    "seq_id",
                    "comp_id",
                    "atom_id",
                    "mutation_flag",
                    "modification_flag",
                    "auth_name",
                ],
            ) as loop:
                for pos in positions:
                    loop.write(
                        id=pos.get("id"),
                        entity_id=pos.get("entity_id"),
                        entity_description=pos.get("entity_id"),
                        asym_id=pos.get("asym_id"),
                        seq_id=pos.get("seq_id") or pos.get("residue_number"),
                        comp_id=pos.get("comp_id") or pos.get("residue_name"),
                        atom_id=pos.get("atom_id"),
                        mutation_flag=pos.get("mutation_flag") or "no",
                        modification_flag=pos.get("modification_flag") or "no",
                        auth_name=pos.get("auth_name") or pos.get("description"),
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
                for sample_probe in sample_probes:
                    loop.write(
                        sample_probe_id=sample_probe.get("sample_probe_id"),
                        sample_id=sample_probe.get("sample_id") or sample_id,
                        probe_id=sample_probe.get("probe_id"),
                        poly_probe_position_id=sample_probe.get("poly_probe_position_id"),
                        fluorophore_type=sample_probe.get("fluorophore_type") or "unspecified",
                        description=sample_probe.get("description"),
                    )

            with writer.loop(
                "_flr_fret_forster_radius",
                [
                    "id",
                    "donor_probe_id",
                    "acceptor_probe_id",
                    "forster_radius",
                    "reduced_forster_radius",
                ],
            ) as loop:
                for row in forster:
                    loop.write(
                        id=row.get("id"),
                        donor_probe_id=row.get("donor_probe_id"),
                        acceptor_probe_id=row.get("acceptor_probe_id"),
                        forster_radius=row.get("forster_radius"),
                        reduced_forster_radius=row.get("reduced_forster_radius")
                        or row.get("forster_radius"),
                    )

            condition_id = sample_row.get("sample_condition_id") or f"condition_{analysis_id}"
            condition = _row_dict(
                self.conn.execute(
                    "SELECT * FROM flr_sample_condition WHERE condition_id = ?", (condition_id,)
                ).fetchone()
            )
            condition.setdefault("condition_id", condition_id)
            with writer.loop("_flr_sample_condition", ["id", "details"]) as loop:
                details = (
                    " ".join(
                        f"{key}={condition.get(key)}"
                        for key in [
                            "ph",
                            "temperature",
                            "ionic_strength",
                            "buffer_composition",
                            "details",
                        ]
                        if condition.get(key) is not None
                    )
                    or None
                )
                loop.write(id=condition.get("condition_id"), details=details or None)

            with writer.loop(
                "_flr_sample",
                [
                    "id",
                    "sample_description",
                    "sample_details",
                    "num_of_probes",
                    "solvent_phase",
                    "sample_condition_id",
                    "entity_assembly_id",
                ],
            ) as loop:
                loop.write(
                    id=sample_id,
                    sample_description=sample_row.get("description")
                    or analysis.get("sample_id")
                    or sample_id,
                    sample_details=sample_row.get("details") or analysis.get("details"),
                    num_of_probes=sample_row.get("num_of_probes") or len(sample_probes) or None,
                    solvent_phase=sample_row.get("solvent_phase")
                    or metadata.get("solvent_phase", "liquid"),
                    sample_condition_id=condition_id,
                    entity_assembly_id=sample_row.get("entity_assembly_id"),
                )

            with writer.loop(
                "_flr_fret_analysis",
                [
                    "id",
                    "experiment_id",
                    "type",
                    "sample_probe_id_1",
                    "sample_probe_id_2",
                    "forster_radius_id",
                    "dataset_list_id",
                    "external_file_id",
                    "software_id",
                ],
            ) as loop:
                loop.write(
                    id=analysis_id,
                    experiment_id=analysis.get("experiment_id"),
                    type=analysis.get("type") or analysis.get("method") or "intensity-based",
                    sample_probe_id_1=analysis.get("sample_probe_id_1"),
                    sample_probe_id_2=analysis.get("sample_probe_id_2"),
                    forster_radius_id=analysis.get("forster_radius_id"),
                    dataset_list_id=1,
                    external_file_id=analysis.get("external_file_id"),
                    software_id=None,
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

    def export_flr_cif_to_text(
        self, analysis_id: Optional[str] = None, include_extension: bool = True
    ) -> str:
        """Export FLR data to a text string."""
        import io

        buffer = io.StringIO()
        self.export_flr_cif(buffer, analysis_id=analysis_id, include_extension=include_extension)
        return buffer.getvalue()
