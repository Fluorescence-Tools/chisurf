"""MFDB-backed fluorophore database adapter for the development plugin."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from chisurf.core.mfdb.repository import MFDatabase, _utc_now

DEFAULT_DATABASE_PATH = Path(__file__).with_name("spectra.db")


# ---------------------------------------------------------------------------
# Canonical optical-component taxonomy — the SINGLE source of truth shared by
# every scraper. Each granular ``kind`` maps to:
#   (canonical category, default spectrum_type, human label)
# The canonical ``category`` drives the mfdb-admin "Spectra" radio tabs; the
# granular ``kind`` is preserved as a ``component_kind`` optical property so the
# finer classification (e.g. "bandpass" vs "longpass") is never lost.
#
# Scrapers MUST register components through ``register_component`` with one of
# these kinds, guaranteeing that filters, dichroics, detectors and light sources
# land in their own category rather than all collapsing into "other"/fluorophore.
# ---------------------------------------------------------------------------
COMPONENT_KINDS: dict[str, tuple[str, str, str]] = {
    # --- fluorophores -------------------------------------------------------
    "fluorescent_protein": ("protein", "absorption", "Fluorescent Protein"),
    "organic_dye": ("organic_dye", "absorption", "Organic Dye"),
    "quantum_dot": ("quantum_dot", "absorption", "Quantum Dot"),
    "nanoparticle": ("nanoparticle", "absorption", "Nanoparticle"),
    "fluorochrome": ("organic_dye", "excitation", "Fluorochrome"),
    # --- filters ------------------------------------------------------------
    "filter": ("filter", "transmission", "Filter"),
    "bandpass": ("filter", "transmission", "Bandpass Filter"),
    "longpass": ("filter", "transmission", "Longpass Filter"),
    "shortpass": ("filter", "transmission", "Shortpass Filter"),
    "notch": ("filter", "transmission", "Notch Filter"),
    "nd": ("filter", "transmission", "Neutral-Density Filter"),
    "laserline": ("filter", "transmission", "Laser-Line Filter"),
    "excitation_filter": ("filter", "transmission", "Excitation Filter"),
    "emission_filter": ("filter", "transmission", "Emission Filter"),
    "colorglass": ("filter", "transmission", "Color-Glass Filter"),
    "astronomy": ("filter", "transmission", "Astronomy Filter"),
    "machine_vision": ("filter", "transmission", "Machine-Vision Filter"),
    "tristimulus": ("filter", "transmission", "Tristimulus Filter"),
    # --- dichroics / beamsplitters / mirrors --------------------------------
    "dichroic": ("dichroic", "transmission", "Dichroic Beamsplitter"),
    "beamsplitter": ("dichroic", "transmission", "Beamsplitter"),
    "mirror": ("dichroic", "reflectance", "Mirror"),
    "hot_cold_mirror": ("dichroic", "reflectance", "Hot/Cold Mirror"),
    # --- detectors ----------------------------------------------------------
    "detector": ("detector", "responsivity", "Detector"),
    "apd": ("detector", "responsivity", "APD Detector"),
    "pmt": ("detector", "quantum_efficiency", "PMT Detector"),
    "camera": ("detector", "quantum_efficiency", "Camera Sensor"),
    # --- light sources ------------------------------------------------------
    "light_source": ("light_source", "emission", "Light Source"),
    "laser": ("light_source", "emission", "Laser"),
    "led": ("light_source", "emission", "LED"),
    "lamp": ("light_source", "emission", "Lamp"),
    # --- misc ---------------------------------------------------------------
    "polarizer": ("other", "transmission", "Polarizer"),
    "other": ("other", "transmission", "Optical Component"),
}


def resolve_component_kind(kind: str | None) -> tuple[str, str, str]:
    """Return ``(category, default_spectrum_type, label)`` for a component kind.

    Falls back to the ``other`` entry for unknown kinds so a scraper can never
    crash the import on an unexpected type — it just lands in the "other" tab.
    """
    return COMPONENT_KINDS.get((kind or "other").lower(), COMPONENT_KINDS["other"])


class FluorophoreDatabase(MFDatabase):
    """Legacy fluorophore database facade backed by canonical MFDB tables."""

    _CATEGORY_ALIASES = {
        "organic dyes": "organic_dye",
        "organic_dyes": "organic_dye",
        "organic dye": "organic_dye",
        "organic_dye": "organic_dye",
        "fluorescent proteins": "protein",
        "fluorescent_proteins": "protein",
        "protein": "protein",
        "proteins": "protein",
        "nanoparticle": "nanoparticle",
        "nanoparticles": "nanoparticle",
        "quantum dot": "quantum_dot",
        "quantum dots": "quantum_dot",
        "quantum_dot": "quantum_dot",
        "filters": "filter",
        "filter": "filter",
        "detectors": "detector",
        "detector": "detector",
        "dichroics": "dichroic",
        "dichroic": "dichroic",
        "lenses": "other",
        "lens": "other",
        "light sources": "light_source",
        "lightsources": "light_source",
        "lightsource": "light_source",
        "other": "other",
    }

    def __init__(
        self,
        db_path: str | os.PathLike[str] | None = None,
        readonly: bool = False,
        **kwargs: Any,
    ) -> None:
        """Open a fluorophore database stored with the MFDB schema.

        Parameters
        ----------
        db_path : str or os.PathLike, optional
            SQLite database path. Defaults to the plugin-local ``spectra.db``.
        readonly : bool, default=False
            Open the database in read-only mode when supported by MFDB.
        **kwargs
            Additional keyword arguments forwarded to :class:`MFDatabase`.

        """
        super().__init__(db_path or DEFAULT_DATABASE_PATH, readonly=readonly, **kwargs)

    def __enter__(self) -> FluorophoreDatabase:
        """Open the database for a legacy ``with db`` block without closing it."""
        if self._conn is None:
            self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Commit or roll back a legacy block while keeping the singleton usable."""
        if self._conn is None:
            return
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()

    def add_probe(self, *args: Any, **kwargs: Any) -> int:
        """Add or update a probe using the legacy plugin call shapes."""
        name = kwargs.pop("chromophore_name", None) or kwargs.pop("name", None)
        type_id = kwargs.pop("type_id", None) or kwargs.pop("probe_type_id", None)
        description = kwargs.pop("description", None)

        if args:
            name = name or args[0]
        if len(args) >= 2 and type_id is None:
            type_id = args[1]
        if len(args) >= 3 and description is None:
            description = args[2]

        if not name:
            raise ValueError("Probe name is required")

        category = self._normalize_enum("category", kwargs.pop("category", "other"))
        probe_origin = self._normalize_enum(
            "probe_origin",
            kwargs.pop("probe_origin", "extrinsic"),
        )
        probe_link_type = self._normalize_enum(
            "probe_link_type",
            kwargs.pop("probe_link_type", "covalent"),
        )
        fluorophore_type = self._normalize_enum(
            "fluorophore_type",
            kwargs.pop("fluorophore_type", "unspecified"),
        )
        reactive_probe_flag = self._normalize_enum(
            "reactive_probe_flag",
            kwargs.pop("reactive_probe_flag", "no"),
        )
        is_curated = int(bool(kwargs.pop("is_curated", 0)))
        quality_flag = int(bool(kwargs.pop("quality_flag", 1)))
        # Provenance — recorded so the GUI can show where each probe came from.
        source = kwargs.pop("source", None)
        source_ref = kwargs.pop("source_ref", None)
        retrieved_at = kwargs.pop("retrieved_at", None)
        kwargs.pop("details", None)

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported probe field(s): {unknown}")

        row = self.conn.execute(
            """SELECT probe_id FROM probes
               WHERE chromophore_name = ?
                 AND COALESCE(type_id, -1) = COALESCE(?, -1)
                 AND deleted_at IS NULL""",
            (str(name), type_id),
        ).fetchone()
        now = _utc_now()
        with self.conn:
            if row is not None:
                # COALESCE keeps an existing provenance value when this call does
                # not provide one (re-runs that only refresh spectra/properties).
                self.conn.execute(
                    """UPDATE probes
                       SET description = ?, category = ?, probe_origin = ?,
                           probe_link_type = ?, fluorophore_type = ?,
                           reactive_probe_flag = ?, is_curated = ?,
                           quality_flag = ?,
                           source = COALESCE(?, source),
                           source_ref = COALESCE(?, source_ref),
                           retrieved_at = COALESCE(?, retrieved_at),
                           updated_at = ?
                       WHERE probe_id = ?""",
                    (
                        description or "",
                        category,
                        probe_origin,
                        probe_link_type,
                        fluorophore_type,
                        reactive_probe_flag,
                        is_curated,
                        quality_flag,
                        source,
                        source_ref,
                        retrieved_at,
                        now,
                        row["probe_id"],
                    ),
                )
                return int(row["probe_id"])

            cursor = self.conn.execute(
                """INSERT INTO probes
                   (chromophore_name, type_id, description, category,
                    probe_origin, probe_link_type, fluorophore_type,
                    reactive_probe_flag, is_curated, quality_flag,
                    source, source_ref, retrieved_at, created_at,
                    updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(name),
                    type_id,
                    description or "",
                    category,
                    probe_origin,
                    probe_link_type,
                    fluorophore_type,
                    reactive_probe_flag,
                    is_curated,
                    quality_flag,
                    source,
                    source_ref,
                    retrieved_at,
                    now,
                    now,
                    None,
                ),
            )
            return int(cursor.lastrowid)

    def register_component(
        self,
        *,
        name: str,
        source: str,
        kind: str = "other",
        source_ref: str | None = None,
        type_name: str | None = None,
        type_display: str | None = None,
        description: str = "",
        properties: dict[str, Any] | None = None,
        spectra: Any = None,
        retrieved_at: str | None = None,
    ) -> int:
        """Register one optical component the same way for every scraper.

        This is the canonical ingestion entry point. It guarantees that a
        fluorophore, filter, dichroic, detector or light source is stored with a
        consistent **category** (from the shared :data:`COMPONENT_KINDS`
        taxonomy), **provenance** (``source``/``source_ref``/``retrieved_at``),
        the granular **component_kind**, canonical optical **properties**, and
        spectra recorded under a uniform spectrum-type vocabulary.

        Parameters
        ----------
        name
            Component name / model / part number (``chromophore_name``).
        source
            Provenance source slug, e.g. ``"fpbase"``, ``"chroma"``,
            ``"thorlabs"``, ``"3doptix"``, ``"atto"``, ``"photochemcad"``.
        kind
            Granular component kind — a key of :data:`COMPONENT_KINDS`
            (e.g. ``"bandpass"``, ``"dichroic"``, ``"apd"``,
            ``"fluorescent_protein"``). Resolves to the canonical category.
        source_ref
            Stable per-record reference (slug, accession, URL, part number).
        type_name, type_display
            Probe-type identifiers; default to ``"{source}_{kind}"`` and a
            human label derived from ``source`` + the kind's label.
        description
            Free-text description.
        properties
            Mapping of optical-property name → value. Names are canonicalized
            (so ``"Cut-On Wavelength (nm)"`` and ``"cut_on"`` collapse to the
            same key). ``None``/empty values are skipped.
        spectra
            Either a mapping ``{spectrum_type: (wavelengths, intensities)}`` or a
            single ``(wavelengths, intensities)`` tuple stored under the kind's
            default spectrum type. ``None`` entries are skipped.
        retrieved_at
            Acquisition timestamp; defaults to now.

        Returns
        -------
        int
            The probe id.
        """
        category, default_spectrum, label = resolve_component_kind(kind)
        type_name = type_name or f"{source}_{kind}"
        type_display = type_display or f"{source.replace('_', ' ').title()} {label}"
        retrieved_at = retrieved_at or _utc_now()

        type_id = self.add_probe_type(type_name, type_display)
        probe_id = self.add_probe(
            chromophore_name=name,
            type_id=type_id,
            description=description or "",
            category=category,
            source=source,
            source_ref=source_ref,
            retrieved_at=retrieved_at,
        )

        # Preserve the granular kind and a human-readable origin so neither the
        # fine classification nor the catalogue label is lost on the probe row.
        self.add_optical_property(probe_id, "component_kind", kind)
        self.add_optical_property(probe_id, "Origin", type_display)

        for prop_name, prop_value in (properties or {}).items():
            if prop_value is None or str(prop_value).strip() == "":
                continue
            self.add_optical_property(probe_id, prop_name, prop_value)

        for spectrum_type, payload in self._iter_spectra(spectra, default_spectrum):
            wavelengths, intensities = payload
            self.add_spectrum(probe_id, spectrum_type, wavelengths, intensities)

        return probe_id

    @staticmethod
    def _iter_spectra(spectra: Any, default_spectrum: str):
        """Normalize the ``spectra`` argument into ``(spectrum_type, (x, y))`` pairs."""
        if spectra is None:
            return
        # Single (wavelengths, intensities) tuple → use the kind's default type.
        if isinstance(spectra, tuple) and len(spectra) == 2:
            x, y = spectra
            if x is not None and y is not None and len(x) and len(y):
                yield default_spectrum, (x, y)
            return
        if isinstance(spectra, dict):
            for stype, payload in spectra.items():
                if payload is None:
                    continue
                x, y = payload
                if x is None or y is None or not len(x) or not len(y):
                    continue
                yield (stype or default_spectrum), (x, y)
            return
        raise TypeError(
            "spectra must be a {type: (x, y)} mapping or a single (x, y) tuple"
        )

    def add_probe_type(
        self,
        name: str,
        description: str | None = None,
        details: str | None = None,
    ) -> int:
        """Add or return a probe type from the MFDB probe-type table."""
        del details
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO probe_types (type_name, display_name) VALUES (?, ?)",
                (name, description or name),
            )
            if description:
                self.conn.execute(
                    "UPDATE probe_types SET display_name = ? WHERE type_name = ?",
                    (description, name),
                )
            row = self.conn.execute(
                "SELECT type_id FROM probe_types WHERE type_name = ?",
                (name,),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to add probe type: {name}")
        return int(row["type_id"])

    def update_probe(self, probe_id: int, **kwargs: Any) -> None:
        """Update mutable probe fields accepted by the legacy plugin."""
        aliases = {"name": "chromophore_name", "probe_type_id": "type_id"}
        allowed = {
            "chromophore_name",
            "description",
            "category",
            "is_curated",
            "quality_flag",
            "type_id",
            "probe_origin",
            "probe_link_type",
            "fluorophore_type",
            "reactive_probe_flag",
        }
        updates: dict[str, Any] = {}
        for key, value in kwargs.items():
            column = aliases.get(key, key)
            if column not in allowed:
                raise ValueError(f"Unsupported probe column: {key}")
            if column in self._VALID_ENUMS:
                value = self._normalize_enum(column, value)
            if column == "category":
                value = self._normalize_enum("category", value)
            if column in {"is_curated", "quality_flag"}:
                value = int(bool(value))
            updates[column] = value
        if not updates:
            return

        updates["updated_at"] = _utc_now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.conn:
            self.conn.execute(
                f"UPDATE probes SET {assignments} WHERE probe_id = ?",
                [*updates.values(), probe_id],
            )

    def get_probe(self, probe_id: int) -> dict[str, Any] | None:
        """Return one probe as a dictionary."""
        row = self.conn.execute(
            "SELECT * FROM probes WHERE probe_id = ? AND deleted_at IS NULL",
            (probe_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_probe_by_id(self, probe_id: int) -> dict[str, Any] | None:
        """Return one probe by integer identifier."""
        return self.get_probe(probe_id)

    def get_probe_types(self) -> list[dict[str, Any]]:
        """Return probe types with legacy ``id`` and MFDB ``type_id`` keys."""
        rows = self.conn.execute("SELECT * FROM probe_types ORDER BY type_id").fetchall()
        return [self._probe_type_dict(row) for row in rows]

    def get_probe_types_dict(self) -> dict[int, str]:
        """Return a mapping from probe type id to display name."""
        return {int(row["type_id"]): row["display_name"] for row in self.get_probe_types()}

    def add_optical_property(
        self,
        probe_id: int,
        property_type: str,
        value: Any,
        unit: str | None = None,
        method: str | None = None,
        condition_json: str | None = None,
        details: str | None = None,
    ) -> None:
        """Store an optical property, normalizing common legacy names."""
        property_name = self._canonical_property_name(property_type)
        extra = details or ""
        if method:
            extra = f"{extra} method={method}".strip()
        if condition_json:
            extra = f"{extra} condition={condition_json}".strip()
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                """INSERT OR REPLACE INTO optical_properties
                   (probe_id, property_name, property_value, unit, details,
                    created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    probe_id,
                    property_name,
                    None if value is None else str(value),
                    unit,
                    extra or None,
                    now,
                    now,
                    None,
                ),
            )

    def get_optical_properties(self, probe_id: int | None = None) -> dict[str, str]:
        """Return optical properties as a legacy ``name -> value`` dictionary."""
        if probe_id is None:
            rows = self.conn.execute(
                "SELECT property_name, property_value FROM optical_properties WHERE deleted_at IS NULL"
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT property_name, property_value
                   FROM optical_properties
                   WHERE probe_id = ? AND deleted_at IS NULL""",
                (probe_id,),
            ).fetchall()
        return {row["property_name"]: row["property_value"] for row in rows}

    def clear_optical_properties(self, probe_id: int) -> None:
        """Remove all optical properties for a probe."""
        with self.conn:
            self.conn.execute("DELETE FROM optical_properties WHERE probe_id = ?", (probe_id,))

    def _normalize_optical_property_keys(self) -> None:
        """Rewrite stored optical property aliases to canonical MFDB keys."""
        rows = self.conn.execute(
            "SELECT id, probe_id, property_name, property_value FROM optical_properties WHERE deleted_at IS NULL"
        ).fetchall()
        with self.conn:
            for row in rows:
                canonical = self._canonical_property_name(row["property_name"])
                if canonical == row["property_name"]:
                    continue
                existing = self.conn.execute(
                    """SELECT id FROM optical_properties
                       WHERE probe_id = ?
                         AND property_name = ?
                         AND deleted_at IS NULL""",
                    (row["probe_id"], canonical),
                ).fetchone()
                if existing is None:
                    self.conn.execute(
                        "UPDATE optical_properties SET property_name = ?, updated_at = ? WHERE id = ?",
                        (canonical, _utc_now(), row["id"]),
                    )
                else:
                    self.conn.execute("DELETE FROM optical_properties WHERE id = ?", (row["id"],))

    def add_spectrum(
        self,
        probe_id: int,
        spectrum_type: str,
        wavelengths: Any,
        intensity_values: Any,
        wavelength_unit: str = "nm",
        intensity_unit: str = "normalized",
        details: str | None = None,
    ) -> None:
        """Store a spectrum as NumPy ``float64`` blobs in MFDB."""
        x = np.asarray(wavelengths, dtype=np.float64)
        y = np.asarray(intensity_values, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("wavelengths and intensity_values must have the same shape")
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                """INSERT OR REPLACE INTO spectra
                   (probe_id, spectrum_type, wavelengths, intensity_values,
                    wavelength_unit, intensity_unit, details, created_at,
                    updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    probe_id,
                    spectrum_type,
                    x.tobytes(),
                    y.tobytes(),
                    wavelength_unit,
                    intensity_unit,
                    details,
                    now,
                    now,
                    None,
                ),
            )

    def get_spectrum(
        self,
        probe_id: int,
        spectrum_type: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | dict[str, Any] | None:
        """Return a legacy spectrum tuple or a raw spectrum row."""
        if spectrum_type is None:
            row = self.conn.execute("SELECT * FROM spectra WHERE id = ?", (probe_id,)).fetchone()
            return dict(row) if row is not None else None
        row = self.conn.execute(
            """SELECT wavelengths, intensity_values FROM spectra
               WHERE probe_id = ? AND spectrum_type = ? AND deleted_at IS NULL""",
            (probe_id, spectrum_type),
        ).fetchone()
        if row is None:
            return None
        return (
            np.frombuffer(row["wavelengths"], dtype=np.float64),
            np.frombuffer(row["intensity_values"], dtype=np.float64),
        )

    def get_standardized_items(self, include_uncurated: bool = False) -> list[dict[str, Any]]:
        """Return probes with canonical optical-property columns for the UI."""
        query = "SELECT * FROM probes WHERE deleted_at IS NULL"
        params: list[Any] = []
        if not include_uncurated:
            query += " AND is_curated = ?"
            params.append(1)
        query += " ORDER BY chromophore_name"
        rows = self.conn.execute(query, params).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            props = self.get_optical_properties(int(row["probe_id"]))
            for key in ("abs_max", "em_max", "qy", "lifetime", "ext_coeff"):
                item[key] = self._property_as_float(props.get(key))
            items.append(item)
        return items

    def search_probes(self, query: str) -> list[dict[str, Any]]:
        """Search probes by chromophore name."""
        rows = self.conn.execute(
            """SELECT * FROM probes
               WHERE chromophore_name LIKE ? AND deleted_at IS NULL
               ORDER BY chromophore_name""",
            (f"%{query}%",),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_probe_full(self, probe_id: int) -> dict[str, Any] | None:
        """Return a probe with canonical numeric properties and spectrum flags."""
        probe = self.get_probe(probe_id)
        if probe is None:
            return None
        props = self.get_optical_properties(probe_id)
        probe.update(props)
        for key in ("abs_max", "em_max", "qy", "lifetime", "ext_coeff"):
            if key in props:
                probe[key] = self._property_as_float(props[key])
        probe["has_abs"] = self.get_spectrum(probe_id, "absorption") is not None
        probe["has_em"] = self.get_spectrum(probe_id, "emission") is not None
        return probe

    def validate_probe(self, probe_id: int) -> list[str]:
        """Validate basic optical-property and spectrum consistency."""
        errors = []
        props = self.get_optical_properties(probe_id)
        qy = self._property_as_float(props.get("qy"))
        if qy is not None and not 0.0 <= qy <= 1.0:
            errors.append(f"QY {qy:g} out of range [0, 1]")
        for key in ("abs_max", "em_max"):
            value = self._property_as_float(props.get(key))
            if value is not None and not 200.0 <= value <= 1000.0:
                errors.append(f"{key} {value:.1f} out of range [200, 1000] nm")
        for spectrum_type in ("absorption", "emission"):
            spectrum = self.get_spectrum(probe_id, spectrum_type)
            if spectrum is None:
                continue
            wavelengths, values = spectrum
            if wavelengths.size == 0 or wavelengths.size != values.size:
                errors.append(f"{spectrum_type} spectrum has invalid data")
        return errors

    def add_probe_image(
        self,
        probe_id: int,
        data: bytes,
        fmt: str | None = None,
        image_name: str | None = None,
    ) -> None:
        """Store a probe structure image in the MFDB image table."""
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                """INSERT INTO images
                   (probe_id, image_name, image_data, image_format,
                    created_at, updated_at, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (probe_id, image_name, data, fmt, now, now, None),
            )

    def add_entity(
        self,
        entity_id: str,
        common_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Add or update a sample entity with the legacy call shape."""
        name = common_name or kwargs.pop("name", None) or entity_id
        entity_type = kwargs.pop("entity_type", kwargs.pop("type", "polymer"))
        details = kwargs.pop("details", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported entity field(s): {unknown}")
        with self.conn:
            now = _utc_now()
            self.conn.execute(
                """INSERT OR REPLACE INTO entities
                   (entity_id, type, description, formula_weight, src_method,
                    number_of_molecules, common_name, created_at, updated_at,
                    deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entity_id, entity_type, details, None, None, 1, name, now, now, None),
            )

    def get_sequence(self, entity_id: str) -> list[dict[str, Any]]:
        """Return polymer sequence rows for an entity."""
        rows = self.conn.execute(
            """SELECT * FROM entity_poly_seq
               WHERE entity_id = ? AND deleted_at IS NULL
               ORDER BY num""",
            (entity_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_poly_probe_positions(
        self,
        entity_id: str | None = None,
        probe_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return polymer probe-position rows."""
        query = "SELECT * FROM flr_poly_probe_position WHERE deleted_at IS NULL"
        params: list[Any] = []
        if entity_id is not None:
            query += " AND entity_id = ?"
            params.append(entity_id)
        if probe_id is not None:
            query += " AND probe_id = ?"
            params.append(probe_id)
        query += " ORDER BY id"
        return [dict(row) for row in self.conn.execute(query, params).fetchall()]

    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity and dependent legacy rows."""
        with self.conn:
            self.conn.execute("DELETE FROM flr_poly_probe_position WHERE entity_id = ?", (entity_id,))
            self.conn.execute("DELETE FROM entity_poly_seq WHERE entity_id = ?", (entity_id,))
            self.conn.execute("DELETE FROM entities WHERE entity_id = ?", (entity_id,))

    @classmethod
    def _normalize_enum(cls, field: str, value: Any) -> str:
        """Normalize and validate legacy enum values."""
        if value is None:
            return "other" if field == "category" else cls._VALID_ENUMS[field][0]
        text = str(value).strip()
        if field == "category":
            normalized = cls._CATEGORY_ALIASES.get(
                text.lower(),
                text.lower().replace("-", "_").replace(" ", "_"),
            )
        else:
            normalized = text.lower().replace("-", "_").replace(" ", "_")
        valid = cls._VALID_ENUMS[field]
        if normalized not in valid:
            raise ValueError(f"Invalid {field}: {value!r}")
        return normalized

    @classmethod
    def _canonical_property_name(cls, property_name: str) -> str:
        """Map common optical-property aliases to canonical keys."""
        key = str(property_name).strip()
        lower = key.lower()
        for canonical, aliases in cls._PROP_ALIASES.items():
            if lower in {alias.lower() for alias in aliases}:
                return canonical
        return key

    @staticmethod
    def _property_as_float(value: Any) -> float | None:
        """Convert an optical-property value to ``float`` when possible."""
        if value is None or value == "":
            return None
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _probe_type_dict(row: Any) -> dict[str, Any]:
        """Convert a probe-type row and add the legacy ``id`` alias."""
        data = dict(row)
        data["id"] = data["type_id"]
        return data
