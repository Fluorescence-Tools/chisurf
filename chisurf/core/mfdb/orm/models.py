"""Reflected SQLAlchemy mappings for the bounded MFDB slice.

The PRD-02/flrCIF field authority lives in the bundled ``.dic`` files and the
canonical SQLite schema, not in hand-written ORM model declarations. This module
builds lightweight SQLAlchemy classes from the canonical schema at import time
and exposes the historical class names used by the sample repository and tests.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import foreign, relationship

from chisurf.core.mfdb.schema import CREATE_TABLES_SQL

from .base import Base


_TABLE_CLASS_NAMES = {
    "mfdb_sample": "MfdbSampleIndex",
    "mfdb_experiment": "MfdbExperiment",
    "flr_sample": "FlrSample",
    "flr_sample_condition": "FlrSampleCondition",
    "flr_sample_probe": "FlrSampleProbe",
    "flr_poly_probe_position": "FlrPolyProbePosition",
    "entities": "Entity",
    "entity_poly_seq": "EntityPolySeq",
    "probes": "Probe",
    "probe_types": "ProbeType",
    "optical_properties": "OpticalProperty",
    "spectra": "Spectrum",
    "chem_descriptors": "ChemDescriptor",
    "flr_fret_forster_radius": "FlrFretForsterRadius",
    "flr_sample_key_value": "FlrSampleKeyValue",
    "flr_experiment": "FlrExperiment",
    "flr_experiment_type": "FlrExperimentType",
    "mfdb_vocabulary": "MfdbVocabulary",
}

_MAPPER_PRIMARY_KEYS = {
    "flr_sample_key_value": ("sample_id", "key"),
}


def _class_name_for_table(table_name: str) -> str:
    """Return the exported class name for a reflected table."""
    if table_name in _TABLE_CLASS_NAMES:
        return _TABLE_CLASS_NAMES[table_name]
    return "".join(part.capitalize() for part in table_name.split("_"))


def _build_schema_database(path: Path) -> None:
    """Create a temporary SQLite database from the canonical schema SQL."""
    with sqlite3.connect(str(path)) as conn:
        for sql in CREATE_TABLES_SQL:
            conn.execute(sql)
        conn.commit()


def _reflect_metadata() -> None:
    """Populate ``Base.metadata`` from the canonical MFDB schema."""
    engine = create_engine("sqlite:///:memory:", future=True)
    with engine.begin() as conn:
        for sql in CREATE_TABLES_SQL:
            conn.exec_driver_sql(sql)
    Base.metadata.reflect(bind=engine)
    for table in Base.metadata.tables.values():
        for column in table.primary_key.columns:
            column.nullable = False


def _new_model_class(class_name: str, table_name: str) -> type[Base]:
    """Create an unmapped model class for one reflected table."""
    return type(
        class_name,
        (Base,),
        {
            "__abstract__": True,
            "__module__": __name__,
            "__doc__": f"Reflected SQLAlchemy mapping for ``{table_name}``.",
        },
    )


def _fk(table_name: str, column_name: str):
    """Return a reflected column used in a relationship foreign-key list."""
    return Base.metadata.tables[table_name].c[column_name]


def _relationship_properties(classes: dict[str, type[Base]]) -> dict[str, dict[str, object]]:
    """Return relationship configuration for the bounded sample/probe graph."""
    return {
        "mfdb_sample": {
            "experiments": relationship(
                classes["mfdb_experiment"],
                back_populates="sample",
                cascade="all, delete-orphan",
            ),
        },
        "mfdb_experiment": {
            "sample": relationship(classes["mfdb_sample"], back_populates="experiments"),
        },
        "flr_sample": {
            "condition": relationship(
                classes["flr_sample_condition"],
                back_populates="samples",
                primaryjoin=(
                    _fk("flr_sample", "sample_condition_id")
                    == _fk("flr_sample_condition", "condition_id")
                ),
                foreign_keys=[_fk("flr_sample", "sample_condition_id")],
            ),
            "sample_probes": relationship(
                classes["flr_sample_probe"],
                back_populates="sample",
                cascade="all, delete-orphan",
            ),
            "fret_pairs": relationship(
                classes["flr_fret_forster_radius"],
                back_populates="sample",
                cascade="all, delete-orphan",
            ),
            "key_values": relationship(
                classes["flr_sample_key_value"],
                back_populates="sample",
                cascade="all, delete-orphan",
            ),
            "experiments": relationship(
                classes["flr_experiment"],
                back_populates="sample",
                cascade="all, delete-orphan",
            ),
        },
        "flr_sample_condition": {
            "samples": relationship(
                classes["flr_sample"],
                back_populates="condition",
                primaryjoin=(
                    _fk("flr_sample_condition", "condition_id")
                    == foreign(_fk("flr_sample", "sample_condition_id"))
                ),
                foreign_keys=[_fk("flr_sample", "sample_condition_id")],
            ),
        },
        "flr_sample_probe": {
            "sample": relationship(classes["flr_sample"], back_populates="sample_probes"),
            "probe": relationship(
                classes["probes"],
                back_populates="sample_probes",
                foreign_keys=[_fk("flr_sample_probe", "probe_id")],
            ),
            "position": relationship(
                classes["flr_poly_probe_position"],
                back_populates="sample_probes",
                foreign_keys=[_fk("flr_sample_probe", "poly_probe_position_id")],
            ),
        },
        "flr_poly_probe_position": {
            "probe": relationship(
                classes["probes"],
                back_populates="positions",
                foreign_keys=[_fk("flr_poly_probe_position", "probe_id")],
            ),
            "entity": relationship(
                classes["entities"],
                back_populates="probe_positions",
                foreign_keys=[_fk("flr_poly_probe_position", "entity_id")],
            ),
            "sample_probes": relationship(classes["flr_sample_probe"], back_populates="position"),
        },
        "entities": {
            "sequences": relationship(
                classes["entity_poly_seq"],
                back_populates="entity",
                cascade="all, delete-orphan",
            ),
            "probe_positions": relationship(
                classes["flr_poly_probe_position"],
                back_populates="entity",
            ),
        },
        "entity_poly_seq": {
            "entity": relationship(classes["entities"], back_populates="sequences"),
        },
        "probes": {
            "optical_properties": relationship(
                classes["optical_properties"],
                back_populates="probe",
                cascade="all, delete-orphan",
            ),
            "spectra": relationship(
                classes["spectra"],
                back_populates="probe",
                cascade="all, delete-orphan",
            ),
            "sample_probes": relationship(
                classes["flr_sample_probe"],
                back_populates="probe",
                foreign_keys=[_fk("flr_sample_probe", "probe_id")],
            ),
            "positions": relationship(
                classes["flr_poly_probe_position"],
                back_populates="probe",
                foreign_keys=[_fk("flr_poly_probe_position", "probe_id")],
            ),
            "chem_descriptor": relationship(
                classes["chem_descriptors"],
                foreign_keys=[_fk("probes", "chromophore_chem_descriptor_id")],
            ),
            "reactive_chem_descriptor": relationship(
                classes["chem_descriptors"],
                foreign_keys=[_fk("probes", "reactive_probe_chem_descriptor_id")],
            ),
            "probe_type": relationship(
                classes["probe_types"],
                foreign_keys=[_fk("probes", "type_id")],
            ),
        },
        "optical_properties": {
            "probe": relationship(classes["probes"], back_populates="optical_properties"),
        },
        "spectra": {
            "probe": relationship(classes["probes"], back_populates="spectra"),
        },
        "flr_fret_forster_radius": {
            "sample": relationship(classes["flr_sample"], back_populates="fret_pairs"),
            "donor_probe": relationship(
                classes["probes"],
                foreign_keys=[_fk("flr_fret_forster_radius", "donor_probe_id")],
            ),
            "acceptor_probe": relationship(
                classes["probes"],
                foreign_keys=[_fk("flr_fret_forster_radius", "acceptor_probe_id")],
            ),
        },
        "flr_sample_key_value": {
            "sample": relationship(classes["flr_sample"], back_populates="key_values"),
        },
        "flr_experiment": {
            "sample": relationship(classes["flr_sample"], back_populates="experiments"),
            "experiment_type": relationship(classes["flr_experiment_type"]),
        },
    }


def _install_reflected_mappings() -> dict[str, type[Base]]:
    """Reflect canonical tables and map lightweight ORM classes to them."""
    if not Base.metadata.tables:
        _reflect_metadata()

    classes: dict[str, type[Base]] = {}
    for table_name in sorted(_TABLE_CLASS_NAMES):
        if table_name not in Base.metadata.tables:
            continue
        class_name = _class_name_for_table(table_name)
        classes[table_name] = _new_model_class(class_name, table_name)

    relationship_config = _relationship_properties(classes)
    for table_name, cls in classes.items():
        table = Base.metadata.tables[table_name]
        mapper_kwargs = {"properties": relationship_config.get(table_name)}
        if table_name in _MAPPER_PRIMARY_KEYS:
            mapper_kwargs["primary_key"] = [
                table.c[column_name]
                for column_name in _MAPPER_PRIMARY_KEYS[table_name]
            ]
        Base.registry.map_imperatively(cls, table, **mapper_kwargs)
        cls.__abstract__ = False
        cls.__tablename__ = table_name

    return classes


_TABLE_CLASSES = _install_reflected_mappings()

for _table_name, _class in _TABLE_CLASSES.items():
    globals()[_class.__name__] = _class


def get_mapped_tables() -> list[type[Base]]:
    """Return reflected classes for the bounded MFDB sample/probe slice."""
    return [
        _TABLE_CLASSES[table_name]
        for table_name in _TABLE_CLASS_NAMES
        if table_name in _TABLE_CLASSES
    ]


def get_table_class_by_name(table_name: str) -> Optional[type[Base]]:
    """Return the reflected class for one table name."""
    return _TABLE_CLASSES.get(table_name)
