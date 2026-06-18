"""SQLAlchemy ORM models for the bounded MFDB sample/probe/FRET slice.

This module provides SQLAlchemy ORM mappings for the core MFDB tables that are
most heavily used by PRD-02a and PRD-02: sample, probe, FRET pair, vocabulary,
and dictionary metadata tables.

The mappings are designed to express the relationships between these tables
once, tested once, and reused by the existing repository API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class MfdbSampleIndex(Base):
    """ORM model for the mfdb_sample index table.

    This table provides a lightweight index for samples with metadata stored as JSON.
    """

    __tablename__ = "mfdb_sample"

    sample_id: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    sample_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    experiments: Mapped[List["MfdbExperiment"]] = relationship(
        back_populates="sample", cascade="all, delete-orphan"
    )


class MfdbExperiment(Base):
    """ORM model for the mfdb_experiment table."""

    __tablename__ = "mfdb_experiment"

    experiment_id: Mapped[str] = mapped_column(Text, primary_key=True)
    sample_id: Mapped[Optional[str]] = mapped_column(
        Text, ForeignKey("mfdb_sample.sample_id", ondelete="SET NULL")
    )
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    project_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    sample: Mapped[Optional[MfdbSampleIndex]] = relationship(
        back_populates="experiments"
    )


class FlrSample(Base):
    """ORM model for the flr_sample table.

    This is the canonical sample table for fluorescence data with detailed
    sample information and relationships to other entities.
    """

    __tablename__ = "flr_sample"

    sample_id: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    num_of_probes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    solvent_phase: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sample_condition_id: Mapped[Optional[str]] = mapped_column(
        Text, ForeignKey("flr_sample_condition.condition_id")
    )
    entity_assembly_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    project_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    measured_by_user_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    measured_by_device_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    measured_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sample_uuid: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    condition: Mapped[Optional["FlrSampleCondition"]] = relationship(
        back_populates="samples"
    )
    sample_probes: Mapped[List["FlrSampleProbe"]] = relationship(
        back_populates="sample", cascade="all, delete-orphan"
    )
    fret_pairs: Mapped[List["FlrFretForsterRadius"]] = relationship(
        back_populates="sample", cascade="all, delete-orphan"
    )
    key_values: Mapped[List["FlrSampleKeyValue"]] = relationship(
        back_populates="sample", cascade="all, delete-orphan"
    )
    experiments: Mapped[List["FlrExperiment"]] = relationship(
        back_populates="sample", cascade="all, delete-orphan"
    )


class FlrSampleCondition(Base):
    """ORM model for the flr_sample_condition table.

    Stores experimental conditions such as pH, temperature, ionic strength, etc.
    """

    __tablename__ = "flr_sample_condition"

    condition_id: Mapped[str] = mapped_column(Text, primary_key=True)
    ph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ionic_strength: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    buffer_composition: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    samples: Mapped[List[FlrSample]] = relationship(
        back_populates="condition"
    )


class FlrSampleProbe(Base):
    """ORM model for the flr_sample_probe table.

    Represents the association between samples and probes, including the
    specific position of each probe in the sample.
    """

    __tablename__ = "flr_sample_probe"

    sample_probe_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sample_id: Mapped[str] = mapped_column(
        Text, ForeignKey("flr_sample.sample_id", ondelete="CASCADE"), nullable=False
    )
    probe_id: Mapped[int] = mapped_column(
        ForeignKey("probes.probe_id"), nullable=False
    )
    poly_probe_position_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("flr_poly_probe_position.id"), nullable=True
    )
    fluorophore_type: Mapped[str] = mapped_column(
        Text, default="unspecified"
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    sample: Mapped[FlrSample] = relationship(back_populates="sample_probes")
    probe: Mapped[Probe] = relationship(foreign_keys=[probe_id])
    position: Mapped[Optional["FlrPolyProbePosition"]] = relationship(
        foreign_keys=[poly_probe_position_id]
    )

    # Unique constraint
    __table_args__ = (
        UniqueConstraint(
            "sample_id", "probe_id", "poly_probe_position_id",
            name="uq_flr_sample_probe_sample_probe_position"
        ),
    )


class FlrPolyProbePosition(Base):
    """ORM model for the flr_poly_probe_position table.

    Stores the position information for probes attached to polymeric entities,
    including residue-level and atom-level positioning.
    """

    __tablename__ = "flr_poly_probe_position"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    probe_id: Mapped[int] = mapped_column(
        ForeignKey("probes.probe_id"), nullable=False
    )
    entity_id: Mapped[str] = mapped_column(
        Text, ForeignKey("entities.entity_id"), nullable=False
    )
    asym_id: Mapped[str] = mapped_column(Text, default="A")
    residue_number: Mapped[int] = mapped_column(Integer, nullable=False)
    residue_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    atom_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mutation_flag: Mapped[str] = mapped_column(Text, default="no")
    modification_flag: Mapped[str] = mapped_column(Text, default="no")
    auth_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    probe: Mapped[Probe] = relationship(foreign_keys=[probe_id])
    entity: Mapped[Entity] = relationship(foreign_keys=[entity_id])
    sample_probes: Mapped[List[FlrSampleProbe]] = relationship(
        back_populates="position"
    )


class Entity(Base):
    """ORM model for the entities table.

    Represents biological entities (proteins, nucleic acids, etc.) that
    can have probes attached to them.
    """

    __tablename__ = "entities"

    entity_id: Mapped[str] = mapped_column(Text, primary_key=True)
    # Column name is 'type' in schema, but we map it to entity_type in Python
    # to avoid shadowing the builtin type() function
    type: Mapped[str] = mapped_column(Text, default="polymer")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    formula_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    src_method: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    number_of_molecules: Mapped[int] = mapped_column(Integer, default=1)
    common_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    sequences: Mapped[List["EntityPolySeq"]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )
    probe_positions: Mapped[List[FlrPolyProbePosition]] = relationship(
        back_populates="entity"
    )


class EntityPolySeq(Base):
    """ORM model for the entity_poly_seq table.

    Stores sequence information for entities, including the actual
    amino acid/nucleotide sequence.
    """

    __tablename__ = "entity_poly_seq"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_id: Mapped[str] = mapped_column(
        Text, ForeignKey("entities.entity_id", ondelete="CASCADE"), nullable=False
    )
    num: Mapped[int] = mapped_column(Integer, nullable=False)
    mon_id: Mapped[str] = mapped_column(Text, nullable=False)
    hetero: Mapped[str] = mapped_column(Text, default="n")
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    entity: Mapped[Entity] = relationship(back_populates="sequences")

    # Unique constraint
    __table_args__ = (
        UniqueConstraint("entity_id", "num", name="uq_entity_poly_seq_entity_num"),
    )


class Probe(Base):
    """ORM model for the probes table.

    Represents fluorescence probes (dyes) with their chemical properties
    and optical characteristics.
    """

    __tablename__ = "probes"

    probe_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chromophore_name: Mapped[str] = mapped_column(Text, nullable=False)
    reactive_probe_flag: Mapped[str] = mapped_column(Text, default="no")
    reactive_probe_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    probe_origin: Mapped[str] = mapped_column(Text, default="extrinsic")
    probe_link_type: Mapped[str] = mapped_column(Text, default="covalent")
    fluorophore_type: Mapped[str] = mapped_column(Text, default="unspecified")
    chromophore_chem_descriptor_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("chem_descriptors.id"), nullable=True
    )
    reactive_probe_chem_descriptor_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("chem_descriptors.id"), nullable=True
    )
    chromophore_center_atom: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(Text, default="other")
    is_curated: Mapped[int] = mapped_column(Integer, default=0)
    quality_flag: Mapped[int] = mapped_column(Integer, default=1)
    type_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("probe_types.type_id"), nullable=True
    )
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    optical_properties: Mapped[List["OpticalProperty"]] = relationship(
        back_populates="probe", cascade="all, delete-orphan"
    )
    spectra: Mapped[List["Spectrum"]] = relationship(
        back_populates="probe", cascade="all, delete-orphan"
    )
    sample_probes: Mapped[List[FlrSampleProbe]] = relationship(
        back_populates="probe", foreign_keys="FlrSampleProbe.probe_id"
    )
    positions: Mapped[List[FlrPolyProbePosition]] = relationship(
        back_populates="probe", foreign_keys="FlrPolyProbePosition.probe_id"
    )
    chem_descriptor: Mapped[Optional["ChemDescriptor"]] = relationship(
        foreign_keys=[chromophore_chem_descriptor_id]
    )
    reactive_chem_descriptor: Mapped[Optional["ChemDescriptor"]] = relationship(
        foreign_keys=[reactive_probe_chem_descriptor_id]
    )
    probe_type: Mapped[Optional["ProbeType"]] = relationship(
        foreign_keys=[type_id]
    )


class ProbeType(Base):
    """ORM model for the probe_types table."""

    __tablename__ = "probe_types"

    type_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type_name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class OpticalProperty(Base):
    """ORM model for the optical_properties table.

    Stores optical properties of probes such as absorption maxima,
    emission maxima, quantum yield, etc.
    """

    __tablename__ = "optical_properties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    probe_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("probes.probe_id", ondelete="CASCADE"), nullable=False
    )
    property_name: Mapped[str] = mapped_column(Text, nullable=False)
    property_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    unit: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    probe: Mapped[Probe] = relationship(back_populates="optical_properties")

    # Unique constraint
    __table_args__ = (
        UniqueConstraint(
            "probe_id", "property_name", name="uq_optical_properties_probe_property"
        ),
    )


class Spectrum(Base):
    """ORM model for the spectra table.

    Stores spectral data (absorption, emission) for probes as binary blobs.
    """

    __tablename__ = "spectra"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    probe_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("probes.probe_id", ondelete="CASCADE"), nullable=False
    )
    spectrum_type: Mapped[str] = mapped_column(Text, nullable=False)
    wavelengths: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary, nullable=True
    )
    intensity_values: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary, nullable=True
    )
    wavelength_unit: Mapped[str] = mapped_column(Text, default="nm")
    intensity_unit: Mapped[str] = mapped_column(Text, default="normalized")
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    probe: Mapped[Probe] = relationship(back_populates="spectra")

    # Unique constraint
    __table_args__ = (
        UniqueConstraint(
            "probe_id", "spectrum_type", name="uq_spectra_probe_type"
        ),
    )


class ChemDescriptor(Base):
    """ORM model for the chem_descriptors table.

    Stores chemical descriptors such as SMILES, InChI, etc. for probes.
    """

    __tablename__ = "chem_descriptors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    descriptor_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    descriptor: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    program: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    program_version: Mapped[Optional[str]] = mapped_column(Text, nullable=True)



class FlrFretForsterRadius(Base):
    """ORM model for the flr_fret_forster_radius table.

    Stores Förster radius (R₀) information for FRET pairs, including the
    donor/acceptor probes, R₀ value, κ² factor, refractive index, and
    overlap integral.

    This table is now sample-scoped (PRD-02 requirement) to prevent
    cross-sample leakage of FRET pair data.
    """

    __tablename__ = "flr_fret_forster_radius"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    forster_radius_id: Mapped[Optional[str]] = mapped_column(
        Text, unique=True, nullable=True
    )
    sample_id: Mapped[str] = mapped_column(
        Text, ForeignKey("flr_sample.sample_id", ondelete="CASCADE"), nullable=False
    )
    donor_probe_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("probes.probe_id"), nullable=False
    )
    acceptor_probe_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("probes.probe_id"), nullable=False
    )
    forster_radius: Mapped[float] = mapped_column(Float, nullable=False)
    reduced_forster_radius: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    kappa_squared: Mapped[float] = mapped_column(Float, default=0.666667)
    index_of_refraction: Mapped[float] = mapped_column(Float, default=1.4)
    overlap_integral: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    sample: Mapped[FlrSample] = relationship(back_populates="fret_pairs")
    donor_probe: Mapped[Probe] = relationship(
        foreign_keys=[donor_probe_id],
        backref="donor_fret_pairs"
    )
    acceptor_probe: Mapped[Probe] = relationship(
        foreign_keys=[acceptor_probe_id],
        backref="acceptor_fret_pairs"
    )

    # Unique constraint - scoped to sample to prevent cross-sample leaks
    __table_args__ = (
        UniqueConstraint(
            "sample_id", "donor_probe_id", "acceptor_probe_id",
            name="uq_flr_fret_forster_radius_sample_donor_acceptor"
        ),
    )


class FlrSampleKeyValue(Base):
    """ORM model for the flr_sample_key_value table.

    Stores key-value metadata pairs for samples, used for PDBx/flrCIF
    compliance and custom metadata.
    """

    __tablename__ = "flr_sample_key_value"

    sample_id: Mapped[str] = mapped_column(
        Text, ForeignKey("flr_sample.sample_id", ondelete="CASCADE"), nullable=False, primary_key=True
    )
    key: Mapped[str] = mapped_column(Text, nullable=False, primary_key=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    sample: Mapped[FlrSample] = relationship(back_populates="key_values")


class FlrExperiment(Base):
    """ORM model for the flr_experiment table."""

    __tablename__ = "flr_experiment"

    experiment_id: Mapped[str] = mapped_column(Text, primary_key=True)
    type_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("flr_experiment_type.type_id"), nullable=True
    )
    sample_id: Mapped[Optional[str]] = mapped_column(
        Text, ForeignKey("flr_sample.sample_id"), nullable=True
    )
    project_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    measured_by_user_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    measured_by_device_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ended_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    setup_definition_id: Mapped[Optional[str]] = mapped_column(
        Text, ForeignKey("fdb_setup_definition.setup_id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    sample: Mapped[Optional[FlrSample]] = relationship(back_populates="experiments")
    experiment_type: Mapped[Optional["FlrExperimentType"]] = relationship(
        foreign_keys=[type_id]
    )


class FlrExperimentType(Base):
    """ORM model for the flr_experiment_type table."""

    __tablename__ = "flr_experiment_type"

    type_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    category: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class MfdbVocabulary(Base):
    """ORM model for the mfdb_vocabulary table.

    Stores vocabulary entries for controlled fields, used for validation
    and autocomplete in the GUI.
    """

    __tablename__ = "mfdb_vocabulary"

    field_name: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text, primary_key=True)
    display_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_builtin: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    deleted_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)





def get_mapped_tables() -> List[type[Base]]:
    """Return a list of all mapped table classes.

    Returns
    -------
    List[type[Base]]
        List of all SQLAlchemy model classes defined in this module.
    """
    return [
        MfdbSampleIndex,
        MfdbExperiment,
        FlrSample,
        FlrSampleCondition,
        FlrSampleProbe,
        FlrPolyProbePosition,
        Entity,
        EntityPolySeq,
        Probe,
        ProbeType,
        OpticalProperty,
        Spectrum,
        ChemDescriptor,
        FlrFretForsterRadius,
        FlrSampleKeyValue,
        FlrExperiment,
        FlrExperimentType,
        MfdbVocabulary,
    ]


def get_table_class_by_name(table_name: str) -> Optional[type[Base]]:
    """Get the model class for a given table name.

    Parameters
    ----------
    table_name : str
        The name of the table.

    Returns
    -------
    type[Base] or None
        The model class for the table, or None if not found.
    """
    for table_class in get_mapped_tables():
        if hasattr(table_class, '__tablename__') and table_class.__tablename__ == table_name:
            return table_class
    return None