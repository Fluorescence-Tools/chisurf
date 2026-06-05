import os
import sqlite3
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from . import schema
from .models import (
    Probe, ProbeType, Entity, SequenceResidue, 
    PolyProbePosition, SampleCondition, OpticalProperty, Spectrum
)

logger = logging.getLogger(__name__)

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
        "abs_max": ["abs_max", "absorption maximum", "λabs", "excitation max", "ex_max", "abs_peak", "λex"],
        "em_max": ["em_max", "emission maximum", "λfl", "emission max", "em_max", "em_peak", "λem"],
        "qy": ["qy", "fluorescence quantum yield", "ηfl", "quantum yield", "phi_acceptor", "phi", "qy_d", "phi_d"],
        "lifetime": ["lifetime", "fluorescence lifetime", "τfl", "tau", "tau_d", "tau_0"],
        "ext_coeff": ["ext_coeff", "molar extinction coefficient", "εmax", "extinction coefficient", "epsilon", "molar_ec"],
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
            plugin_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            db_path = plugin_dir / "sample_management.db"
        
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._ensure_connection()
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
        rows = [dict(r) for r in self.conn.execute("SELECT id, probe_id, property_name FROM optical_properties").fetchall()]
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
                            (row["probe_id"], new_name)).fetchone()
                        if exists:
                            self.conn.execute("DELETE FROM optical_properties WHERE id = ?", (row["id"],))
                        else:
                            self.conn.execute("UPDATE optical_properties SET property_name = ? WHERE id = ?", (new_name, row["id"]))

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
            self.conn.execute("INSERT OR IGNORE INTO probe_types (type_name, display_name) VALUES (?, ?)", (type_name, display_name))
        row = self.conn.execute("SELECT type_id FROM probe_types WHERE type_name = ?", (type_name,)).fetchone()
        return row["type_id"]

    def get_probe_types(self) -> List[sqlite3.Row]:
        """Return all probe types.

        Returns
        -------
        list of sqlite3.Row
        """
        return self.conn.execute("SELECT type_id, type_name, display_name FROM probe_types").fetchall()

    def add_probe(self, chromophore_name: str, type_id: int, description: str = "", category: str = "other", **kwargs) -> int:
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
            self.conn.execute(f"INSERT OR IGNORE INTO probes ({', '.join(cols)}) VALUES ({placeholders})", tuple(vals))
        row = self.conn.execute("SELECT probe_id FROM probes WHERE chromophore_name = ? AND type_id = ?", (chromophore_name, type_id)).fetchone()
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
        if not kwargs: return
        for k, v in kwargs.items():
            self._validate_enum(k, v)
        cols = [f"{k} = ?" for k in kwargs.keys()]
        with self.conn:
            self.conn.execute(f"UPDATE probes SET {', '.join(cols)} WHERE probe_id = ?", (*kwargs.values(), probe_id))

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

    def add_optical_property(self, probe_id: int, name: str, value: str, unit: Optional[str] = None):
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
                (probe_id, name, str(value), unit)
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
        rows = self.conn.execute("SELECT property_name, property_value FROM optical_properties WHERE probe_id = ?", (probe_id,)).fetchall()
        return {r["property_name"]: r["property_value"] for r in rows}

    def add_spectrum(self, probe_id: int, spec_type: str, wavelengths: np.ndarray, values: np.ndarray):
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
        """
        w_blob = wavelengths.astype(np.float64).tobytes()
        v_blob = values.astype(np.float64).tobytes()
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO spectra (probe_id, spectrum_type, wavelengths, intensity_values) VALUES (?, ?, ?, ?)",
                (probe_id, spec_type, w_blob, v_blob)
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
        row = self.conn.execute("SELECT wavelengths, intensity_values FROM spectra WHERE probe_id = ? AND spectrum_type = ?", (probe_id, spec_type)).fetchone()
        if row:
            w = np.frombuffer(row["wavelengths"], dtype=np.float64)
            v = np.frombuffer(row["intensity_values"], dtype=np.float64)
            return w, v
        return None

    def add_probe_image(self, probe_id: int, data: bytes, image_format: str = "png", image_name: str = ""):
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
                (probe_id, image_name, data, image_format)
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
        return self.conn.execute("SELECT image_name, image_data, image_format FROM images WHERE probe_id = ?", (probe_id,)).fetchall()

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
            self.conn.execute(f"INSERT OR REPLACE INTO entities ({', '.join(cols)}) VALUES ({placeholders})", tuple(vals))

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
        return self.conn.execute("SELECT * FROM entities WHERE entity_id = ?", (entity_id,)).fetchone()

    def update_entity(self, entity_id: str, **kwargs):
        """Update entity fields.

        Parameters
        ----------
        entity_id : str
            Entity identifier.
        **kwargs
            Field name/value pairs to update.
        """
        if not kwargs: return
        cols = [f"{k} = ?" for k in kwargs.keys()]
        with self.conn:
            self.conn.execute(f"UPDATE entities SET {', '.join(cols)} WHERE entity_id = ?", (*kwargs.values(), entity_id))

    def delete_entity(self, entity_id: str):
        """Delete an entity and all related data.

        Parameters
        ----------
        entity_id : str
            Entity identifier.
        """
        with self.conn:
            self.conn.execute("DELETE FROM entity_poly_seq WHERE entity_id = ?", (entity_id,))
            self.conn.execute("DELETE FROM flr_poly_probe_position WHERE entity_id = ?", (entity_id,))
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
                self.conn.execute("INSERT INTO entity_poly_seq (entity_id, num, mon_id) VALUES (?, ?, ?)", (entity_id, i, mon))

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
        return self.conn.execute("SELECT * FROM entity_poly_seq WHERE entity_id = ? ORDER BY num", (entity_id,)).fetchall()

    def add_poly_probe_position(self, probe_id: int, entity_id: str, residue_number: int, 
                               asym_id: str = 'A', residue_name: str = None, description: str = None):
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
            self.conn.execute("""
                INSERT OR REPLACE INTO flr_poly_probe_position (probe_id, entity_id, asym_id, residue_number, residue_name, description)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (probe_id, entity_id, asym_id, residue_number, residue_name, description))

    def get_poly_probe_positions(self, probe_id: int = None, entity_id: str = None) -> List[sqlite3.Row]:
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
            "SELECT * FROM probes WHERE chromophore_name LIKE ? OR category LIKE ? OR description LIKE ?", (q, q, q)).fetchall()

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
        return {r["type_id"]: r["display_name"] for r in self.conn.execute("SELECT type_id, display_name FROM probe_types").fetchall()}

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
        if not probe: return None
        d = dict(probe)
        d.update(self.get_standardized_optical_properties(probe_id))
        d["has_abs"] = 1 if (self.get_spectrum(probe_id, "absorption") or self.get_spectrum(probe_id, "excitation")) else 0
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
        if not self.get_spectrum(probe_id, "absorption") and not self.get_spectrum(probe_id, "excitation"):
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
        has_abs = 1 if (self.get_spectrum(probe_id, "absorption") or self.get_spectrum(probe_id, "excitation")) else 0
        has_em = 1 if self.get_spectrum(probe_id, "emission") else 0
        return (found + has_abs + has_em) / 6.0
