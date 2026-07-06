"""SQLAlchemy-backed sample repository adapter.

This module provides a SQLAlchemy-backed adapter for sample/probe/FRET operations,
serving as a canonical persistence path for the bounded MFDB slice. This adapter can
initially be called only from tests. Once verified, sample_manager.create_sample()
and get_sample_full_description() can delegate to it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from mfdb.dictionary_schema_map import build_dictionary_schema_map
from mfdb.external_refs import diff_sequences
from mfdb.models import (
    EntityDefinition,
    FretPairDefinition,
    MutationDefinition,
    ProbeDefinition,
    SampleDefinition,
)
from mfdb.pdbx_metadata import MmcifDictionary
from mfdb.repository import MFDatabase

from .base import session_scope
from .models import (
    Entity,
    EntityPolySeq,
    FlrFretForsterRadius,
    FlrPolyProbePosition,
    FlrSample,
    FlrSampleCondition,
    FlrSampleKeyValue,
    FlrSampleProbe,
    Probe,
    OpticalProperty,
    Spectrum,
    ChemDescriptor,
    StructRef,
    StructRefSeq,
    StructRefSeqDif,
)

logger = logging.getLogger(__name__)

_DICTIONARY = MmcifDictionary.load_bundled()
_SCHEMA_MAP = build_dictionary_schema_map(dictionary=_DICTIONARY)


def _column(dictionary_item: str) -> str:
    """Return the live DB column bound to a dictionary item."""
    mapped = _SCHEMA_MAP.map_dictionary_item(dictionary_item)
    if mapped is None:
        raise KeyError(f"No schema binding for dictionary item {dictionary_item}")
    return mapped.column_name


def _default(dictionary_item: str, fallback: str = "") -> str:
    """Return a dictionary default value with a fallback for unbound items."""
    value = _DICTIONARY.get_default(dictionary_item)
    if value in {"", ".", "?"}:
        return fallback
    return value


def _default_float(dictionary_item: str, fallback: float) -> float:
    """Return a dictionary float default value with a fallback."""
    value = _default(dictionary_item)
    if not value:
        return fallback
    return float(value)


def _value_or_default(value: str | None, dictionary_item: str, fallback: str = "") -> str:
    """Return an explicit value or the dictionary-defined default."""
    return value if value else _default(dictionary_item, fallback)


def _normalize_descriptor_value(value: str) -> str:
    """Normalize descriptor text for idempotent descriptor lookup."""
    return " ".join(str(value).strip().split())


def _get_or_create_descriptor(
    session: Session,
    descriptor_type: str,
    descriptor: str,
    *,
    program: str = "ChiSurf",
    program_version: str | None = None,
) -> int | None:
    """Return an existing chemical descriptor ID or insert one.

    Chemical descriptors are identity rows. The natural key is the normalized
    ``(descriptor_type, descriptor, program, program_version)`` tuple; repeated
    probe creation must reuse the same row instead of creating duplicates.
    """
    normalized_type = descriptor_type.strip().upper()
    normalized_descriptor = _normalize_descriptor_value(descriptor)
    if not normalized_type or not normalized_descriptor:
        return None

    existing = session.execute(
        select(ChemDescriptor).where(
            ChemDescriptor.descriptor_type == normalized_type,
            ChemDescriptor.descriptor == normalized_descriptor,
            ChemDescriptor.program == program,
            ChemDescriptor.program_version.is_(program_version)
            if program_version is None
            else ChemDescriptor.program_version == program_version,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing.id

    descriptor_row = ChemDescriptor(
        descriptor_type=normalized_type,
        descriptor=normalized_descriptor,
        program=program,
        program_version=program_version,
    )
    session.add(descriptor_row)
    session.flush()
    return descriptor_row.id


def _apply_probe_descriptors(
    session: Session,
    probe_row: Probe,
    definition: ProbeDefinition,
) -> None:
    """Attach existing or new chemical descriptor IDs to one probe row."""
    chromophore_descriptor_id = None
    if definition.chromophore_smiles:
        chromophore_descriptor_id = _get_or_create_descriptor(
            session,
            "SMILES",
            definition.chromophore_smiles,
        )
    elif definition.chromophore_inchi:
        chromophore_descriptor_id = _get_or_create_descriptor(
            session,
            "InChI",
            definition.chromophore_inchi,
        )
    if chromophore_descriptor_id is not None:
        probe_row.chromophore_chem_descriptor_id = chromophore_descriptor_id

    reactive_descriptor_id = None
    if definition.reactive_probe_smiles:
        reactive_descriptor_id = _get_or_create_descriptor(
            session,
            "SMILES",
            definition.reactive_probe_smiles,
        )
    if reactive_descriptor_id is not None:
        probe_row.reactive_probe_chem_descriptor_id = reactive_descriptor_id


def _persist_external_refs(session: Session, entity_def: EntityDefinition) -> None:
    """Persist UniProt/PDB cross-references and mutations for one entity (PRD-39).

    Writes ``struct_ref`` rows (UNP for ``uniprot_accession``, PDB for
    ``pdb_id``), a ``struct_ref_seq`` alignment for the UniProt reference, and a
    ``struct_ref_seq_dif`` row per :class:`MutationDefinition`. No-op when the
    entity carries no external references or mutations, so existing samples are
    unaffected.
    """
    entity_id = entity_def.name
    has_unp = bool(entity_def.uniprot_accession)
    mutations: list[MutationDefinition] = list(entity_def.mutations or [])
    if not mutations and entity_def.reference_sequence and entity_def.sequence:
        mutations = diff_sequences(
            entity_def.sequence,
            entity_def.reference_sequence,
        )
    if not has_unp and not entity_def.pdb_id and not mutations:
        return

    if entity_def.pdb_id:
        session.add(
            StructRef(
                ref_id=f"{entity_id}_PDB",
                entity_id=entity_id,
                db_name="PDB",
                pdbx_db_accession=entity_def.pdb_id,
                details=entity_def.pdb_chain_id or None,
            )
        )

    # The UniProt reference anchors the sequence alignment + mutation diffs.
    unp_ref_id = f"{entity_id}_UNP"
    session.add(
        StructRef(
            ref_id=unp_ref_id,
            entity_id=entity_id,
            db_name="UNP",
            pdbx_db_accession=entity_def.uniprot_accession,
            pdbx_seq_one_letter_code=entity_def.reference_sequence or None,
            organism=entity_def.organism or None,
        )
    )

    construct_len = len(entity_def.sequence or "") or None
    ref_len = len(entity_def.reference_sequence) if entity_def.reference_sequence else construct_len
    align_id = f"{entity_id}_UNP_aln"
    session.add(
        StructRefSeq(
            align_id=align_id,
            ref_id=unp_ref_id,
            seq_align_beg=1 if construct_len else None,
            seq_align_end=construct_len,
            db_align_beg=1 if ref_len else None,
            db_align_end=ref_len,
            pdbx_db_accession=entity_def.uniprot_accession,
        )
    )

    for ordinal, mut in enumerate(mutations, start=1):
        session.add(
            StructRefSeqDif(
                align_id=align_id,
                seq_num=mut.seq_id,
                mon_id=mut.mut_comp_id or None,
                db_mon_id=mut.wt_comp_id or None,
                details=(mut.kind or "engineered_mutation")
                .replace("_", " ")
                .upper(),
                pdbx_seq_db_name="UNP",
                pdbx_seq_db_accession_code=entity_def.uniprot_accession,
                pdbx_ordinal=ordinal,
            )
        )


def _attach_external_refs(
    session: Session, entity_id: str, entity_data: dict[str, Any]
) -> None:
    """Populate ``external_refs`` and ``mutations`` on an entity dict (PRD-39)."""
    refs = (
        session.execute(
            select(StructRef).where(
                StructRef.entity_id == entity_id,
                StructRef.deleted_at.is_(None),
            )
        )
        .scalars()
        .all()
    )
    external_refs = [
        {
            "db_name": r.db_name,
            "accession": r.pdbx_db_accession,
            "db_code": r.db_code,
            "organism": r.organism,
        }
        for r in refs
    ]
    entity_data["external_refs"] = external_refs

    mutations: list[dict[str, Any]] = []
    align_ids = []
    if refs:
        align_ids = [
            a
            for (a,) in session.execute(
                select(StructRefSeq.align_id).where(
                    StructRefSeq.ref_id.in_([r.ref_id for r in refs]),
                    StructRefSeq.deleted_at.is_(None),
                )
            ).all()
        ]
    if align_ids:
        difs = (
            session.execute(
                select(StructRefSeqDif).where(
                    StructRefSeqDif.align_id.in_(align_ids),
                    StructRefSeqDif.deleted_at.is_(None),
                )
            )
            .scalars()
            .all()
        )
        mutations = [
            {
                "seq_id": d.seq_num,
                "mut_comp_id": d.mon_id,
                "wt_comp_id": d.db_mon_id,
                "details": d.details,
            }
            for d in sorted(difs, key=lambda d: (d.pdbx_ordinal or 0, d.seq_num or 0))
        ]
    entity_data["mutations"] = mutations


def create_sample_graph(
    db: MFDatabase,
    definition: SampleDefinition,
    *,
    sample_id: str | None = None,
    display_name: str | None = None,
    sample_type: str | None = None,
    metadata_json: str | None = None,
) -> str:
    """Persist a full sample graph and return sample_id.

    This function creates a complete sample with all its associated data
    (entities, probes, positions, FRET pairs, conditions, metadata) in a
    single transaction.

    Parameters
    ----------
    db : MFDatabase
        The MFDatabase instance to use for persistence.
    definition : SampleDefinition
        The sample definition to persist.

    Returns
    -------
    str
        The sample_id of the created sample.

    Notes
    -----
    This function uses the SQLAlchemy ORM to persist the sample graph,
    ensuring that all relationships are properly maintained. It operates
    within a transaction scope to ensure atomicity.
    """
    db_path = db.db_path

    with session_scope(db_path) as session:
        return _create_sample_graph_in_session(
            session,
            definition,
            sample_id=sample_id,
            display_name=display_name,
            sample_type=sample_type,
            metadata_json=metadata_json,
        )


def _create_sample_graph_in_session(
    session: Session,
    definition: SampleDefinition,
    *,
    sample_id: str | None = None,
    display_name: str | None = None,
    sample_type: str | None = None,
    metadata_json: str | None = None,
) -> str:
    """Create sample graph within a SQLAlchemy session.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    definition : SampleDefinition
        The sample definition to persist.

    Returns
    -------
    str
        The sample_id of the created sample.
    """
    sample_id = sample_id or definition.name
    display_name = display_name or definition.name

    # Create the main sample record
    # flr_sample.description is the primary human-readable identifier (display name).
    # flr_sample.details holds an optional longer description.
    sample = FlrSample(
        sample_id=sample_id,
        description=display_name,
        details=definition.description or None,
        sample_type=sample_type,
        num_of_probes=len(definition.probes) if definition.probes else None,
        solvent_phase=definition.solvent_phase,
        project_id=None,
        sample_uuid=f"{sample_id}_uuid",  # Will be replaced with proper UUID generation
    )
    session.add(sample)
    session.flush()

    # Create entities and their sequences
    entity_map = {}  # entity_id -> Entity
    for entity_def in definition.entities:
        entity = Entity(
            entity_id=entity_def.name,
            type=entity_def.entity_type,
            description=entity_def.details,
            common_name=entity_def.name,
        )
        session.add(entity)
        entity_map[entity_def.name] = entity

        # Create sequence if provided
        if entity_def.sequence:
            for i, residue in enumerate(entity_def.sequence):
                seq_record = EntityPolySeq(
                    entity_id=entity_def.name,
                    num=i + 1,
                    mon_id=str(residue),
                )
                session.add(seq_record)

    session.flush()

    # Create external references (struct_ref family) and mutations (PRD-39)
    for entity_def in definition.entities:
        _persist_external_refs(session, entity_def)

    session.flush()

    # Create probes
    probe_map = {}  # probe name -> Probe
    for probe_def in definition.probes:
        # Check if probe already exists
        existing_probe = session.execute(
            select(Probe).where(Probe.chromophore_name == probe_def.name)
        ).scalar_one_or_none()

        if existing_probe:
            probe = existing_probe
        else:
            probe = Probe(
                chromophore_name=probe_def.name,
                **{
                    _column("_flr_probe_list.probe_origin"): _value_or_default(
                        probe_def.probe_origin,
                        "_flr_probe_list.probe_origin",
                    ),
                    _column("_flr_probe_list.probe_link_type"): _value_or_default(
                        probe_def.probe_link_type,
                        "_flr_probe_list.probe_link_type",
                    ),
                    "fluorophore_type": _default(
                        "_flr_sample_probe_details.fluorophore_type"
                    ),
                    _column("_flr_probe_list.reactive_probe_flag"): _value_or_default(
                        probe_def.reactive_probe_flag,
                        "_flr_probe_list.reactive_probe_flag",
                    ),
                },
                reactive_probe_name=probe_def.reactive_probe_name or None,
                chromophore_center_atom=probe_def.chromophore_center_atom,
                description=None,
                category="other",
            )
            session.add(probe)
            session.flush()
        _apply_probe_descriptors(session, probe, probe_def)

        probe_map[probe_def.name] = probe
        _add_probe_optical_data(session, probe, probe_def)

    # Create positions for each probe
    position_map = {}  # (probe_id, entity_id, residue_info) -> FlrPolyProbePosition
    for probe_def in definition.probes:
        if probe_def.entity_index is not None and probe_def.entity_index < len(definition.entities):
            entity_def = definition.entities[probe_def.entity_index]
            entity_id = entity_def.name

            position = FlrPolyProbePosition(
                probe_id=probe_map[probe_def.name].probe_id,
                entity_id=entity_id,
                **{
                    _column("_flr_poly_probe_position.asym_id"): _value_or_default(
                        probe_def.asym_id,
                        "_flr_poly_probe_position.asym_id",
                    ),
                    _column("_flr_poly_probe_position.seq_id"): probe_def.seq_id or 1,
                    _column("_flr_poly_probe_position.comp_id"): probe_def.comp_id,
                    _column("_flr_poly_probe_position.atom_id"): probe_def.atom_id,
                    _column("_flr_poly_probe_position.mutation_flag"): _value_or_default(
                        probe_def.mutation_flag,
                        "_flr_poly_probe_position.mutation_flag",
                    ),
                    _column("_flr_poly_probe_position.modification_flag"): _value_or_default(
                        probe_def.modification_flag,
                        "_flr_poly_probe_position.modification_flag",
                    ),
                    _column("_flr_poly_probe_position.auth_name"): probe_def.auth_name,
                },
                description=None,
            )
            session.add(position)
            session.flush()

            position_map[(probe_def.name, entity_id)] = position

    # Create sample_probes (association between samples and probes)
    fluorophore_types = _derive_fluorophore_types(definition)
    for i, probe_def in enumerate(definition.probes):
        probe = probe_map[probe_def.name]
        position = None

        # Find the position for this probe if it exists
        if probe_def.entity_index is not None and probe_def.entity_index < len(definition.entities):
            entity_def = definition.entities[probe_def.entity_index]
            position = position_map.get((probe_def.name, entity_def.name))

        sample_probe = FlrSampleProbe(
            sample_id=sample_id,
            probe_id=probe.probe_id,
            poly_probe_position_id=position.id if position else None,
            fluorophore_type=(
                fluorophore_types[i]
                if i < len(fluorophore_types)
                else _default("_flr_sample_probe_details.fluorophore_type")
            ),
            description=None,
        )
        session.add(sample_probe)

    # Create FRET pairs
    for fret_pair in definition.fret_pairs:
        # FretPairDefinition uses probe_1_index and probe_2_index, not probe objects
        donor_probe_def = definition.probes[fret_pair.probe_1_index] if fret_pair.probe_1_index < len(definition.probes) else None
        acceptor_probe_def = definition.probes[fret_pair.probe_2_index] if fret_pair.probe_2_index < len(definition.probes) else None

        donor_probe = probe_map.get(donor_probe_def.name) if donor_probe_def else None
        acceptor_probe = probe_map.get(acceptor_probe_def.name) if acceptor_probe_def else None

        if donor_probe and acceptor_probe:
            fret_record = FlrFretForsterRadius(
                forster_radius_id=f"{sample_id}_forster_{fret_pair.probe_1_index}_{fret_pair.probe_2_index}",
                sample_id=sample_id,
                donor_probe_id=donor_probe.probe_id,
                acceptor_probe_id=acceptor_probe.probe_id,
                forster_radius=(
                    fret_pair.forster_radius_nm
                    or _default_float("_flr_fret_forster_radius.forster_radius", 5.0)
                ),
                reduced_forster_radius=fret_pair.reduced_forster_radius_nm,
                kappa_squared=(
                    fret_pair.kappa_squared
                    if fret_pair.kappa_squared is not None
                    else _default_float("_flr_fret_forster_radius.kappa_squared", 0.666667)
                ),
                index_of_refraction=(
                    fret_pair.refractive_index
                    if fret_pair.refractive_index is not None
                    else _default_float("_flr_fret_forster_radius.index_of_refraction", 1.4)
                ),
                overlap_integral=fret_pair.overlap_integral,
                details=None,
            )
            session.add(fret_record)

    # Create condition from SampleDefinition fields
    # SampleDefinition has individual condition fields, not a condition dict
    has_condition = any([
        definition.ph is not None,
        definition.temperature_k is not None,
        definition.salt_concentration_m is not None,
        definition.buffer_description is not None,
        definition.solvent_phase is not None,
    ])

    if has_condition:
        condition_id = f"{sample_id}_condition"
        condition = FlrSampleCondition(
            condition_id=condition_id,
            ph=definition.ph,
            temperature=definition.temperature_k,
            ionic_strength=definition.salt_concentration_m,
            buffer_composition=definition.buffer_description,
            details=None,
        )
        session.add(condition)
        sample.sample_condition_id = condition_id

    # Create key-value metadata from extra dict
    for key, value in definition.extra.items():
        kv = FlrSampleKeyValue(
            sample_id=sample_id,
            key=key,
            value=str(value),
            details=None,
        )
        session.add(kv)

    # Update sample with condition reference
    if has_condition:
        sample.sample_condition_id = condition_id

    session.flush()

    return sample_id


def get_sample_graph(db: MFDatabase, sample_id: str) -> dict[str, Any] | None:
    """Return sample, entities, probes, positions, condition, and FRET pairs.

    This function retrieves a complete sample graph from the database using
    SQLAlchemy ORM, returning all associated data in a nested dictionary.

    Parameters
    ----------
    db : MFDatabase
        The MFDatabase instance to use for retrieval.
    sample_id : str
        The sample_id to retrieve.

    Returns
    -------
    Dict[str, Any] or None
        A dictionary containing the full sample graph, or None if not found.
        The dictionary includes keys: 'sample', 'entities', 'probes', 'positions',
        'condition', 'fret_pairs', and 'key_values'.
    """
    db_path = db.db_path

    with session_scope(db_path) as session:
        return _get_sample_graph_in_session(session, sample_id)


def _get_sample_graph_in_session(
    session: Session, sample_id: str
) -> dict[str, Any] | None:
    """Get sample graph within a SQLAlchemy session.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    sample_id : str
        The sample_id to retrieve.

    Returns
    -------
    Dict[str, Any] or None
        The full sample graph as a dictionary, or None if not found.
    """
    # Get the main sample
    sample = session.get(FlrSample, sample_id)
    if not sample:
        return None

    result = {
        "sample": {
            "sample_id": sample.sample_id,
            "description": sample.description,
            "details": sample.details,
            "num_of_probes": sample.num_of_probes,
            "solvent_phase": sample.solvent_phase,
            "project_id": sample.project_id,
            "sample_uuid": sample.sample_uuid,
            "created_at": sample.created_at,
            "updated_at": sample.updated_at,
        },
        "entities": [],
        "probes": [],
        "positions": [],
        "condition": None,
        "fret_pairs": [],
        "key_values": [],
    }

    # Get entities for this sample (via sample_probes -> positions -> entities)
    entities_seen = set()
    for sample_probe in sample.sample_probes:
        if sample_probe.position and sample_probe.position.entity:
            entity = sample_probe.position.entity
            if entity.entity_id not in entities_seen:
                entities_seen.add(entity.entity_id)
                entity_data = {
                    "entity_id": entity.entity_id,
                    "type": entity.type,
                    "description": entity.description,
                    "common_name": entity.common_name,
                    "formula_weight": entity.formula_weight,
                    "src_method": entity.src_method,
                    "number_of_molecules": entity.number_of_molecules,
                    "sequences": [
                        {"num": seq.num, "mon_id": seq.mon_id, "hetero": seq.hetero}
                        for seq in entity.sequences
                    ],
                }
                _attach_external_refs(session, entity.entity_id, entity_data)
                result["entities"].append(entity_data)

    # Get probes for this sample
    for sample_probe in sample.sample_probes:
        if sample_probe.probe:
            probe_data = {
                "probe_id": sample_probe.probe.probe_id,
                "name": sample_probe.probe.chromophore_name,
                "fluorophore_type": sample_probe.fluorophore_type,
                "probe_origin": sample_probe.probe.probe_origin,
                "probe_link_type": sample_probe.probe.probe_link_type,
                "reactive_probe_flag": sample_probe.probe.reactive_probe_flag,
                "chromophore_center_atom": sample_probe.probe.chromophore_center_atom,
                "description": sample_probe.description,
                "optical_properties": [
                    {
                        "property_name": prop.property_name,
                        "property_value": prop.property_value,
                        "unit": prop.unit,
                    }
                    for prop in sample_probe.probe.optical_properties
                ],
                "spectra": [
                    {
                        "spectrum_type": spec.spectrum_type,
                        "wavelength_unit": spec.wavelength_unit,
                        "intensity_unit": spec.intensity_unit,
                    }
                    for spec in sample_probe.probe.spectra
                ],
            }

            # Add position information
            if sample_probe.position:
                probe_data["position"] = {
                    "asym_id": sample_probe.position.asym_id,
                    "residue_number": sample_probe.position.residue_number,
                    "residue_name": sample_probe.position.residue_name,
                    "atom_id": sample_probe.position.atom_id,
                    "mutation_flag": sample_probe.position.mutation_flag,
                    "modification_flag": sample_probe.position.modification_flag,
                    "auth_name": sample_probe.position.auth_name,
                    "entity_id": sample_probe.position.entity_id,
                }

            result["probes"].append(probe_data)

    # Get condition
    if sample.condition:
        result["condition"] = {
            "condition_id": sample.condition.condition_id,
            "ph": sample.condition.ph,
            "temperature": sample.condition.temperature,
            "ionic_strength": sample.condition.ionic_strength,
            "buffer_composition": sample.condition.buffer_composition,
            "details": sample.condition.details,
        }

    # Get FRET pairs
    for fret_pair in sample.fret_pairs:
        result["fret_pairs"].append({
            "forster_radius_id": fret_pair.forster_radius_id,
            "sample_id": fret_pair.sample_id,
            "donor_probe_id": fret_pair.donor_probe_id,
            "acceptor_probe_id": fret_pair.acceptor_probe_id,
            "donor_probe": fret_pair.donor_probe.chromophore_name if fret_pair.donor_probe else "",
            "acceptor_probe": fret_pair.acceptor_probe.chromophore_name if fret_pair.acceptor_probe else "",
            "forster_radius": fret_pair.forster_radius,
            "reduced_forster_radius": fret_pair.reduced_forster_radius,
            "kappa_squared": fret_pair.kappa_squared,
            "index_of_refraction": fret_pair.index_of_refraction,
            "overlap_integral": fret_pair.overlap_integral,
            "details": fret_pair.details,
        })

    # Get key-value metadata
    for kv in sample.key_values:
        result["key_values"].append({
            "key": kv.key,
            "value": kv.value,
            "details": kv.details,
        })

    return result


def upsert_probe(db: MFDatabase, probe: ProbeDefinition) -> int:
    """Persist probe identity plus chemical descriptors and return probe_id.

    This function inserts or updates a probe record along with its chemical
    descriptors (SMILES, InChI, etc.) and returns the probe_id.

    Parameters
    ----------
    db : MFDatabase
        The MFDatabase instance to use for persistence.
    probe : ProbeDefinition
        The probe definition to persist.

    Returns
    -------
    int
        The probe_id of the persisted probe.
    """
    db_path = db.db_path

    with session_scope(db_path) as session:
        return _upsert_probe_in_session(session, probe)


def _upsert_probe_in_session(session: Session, probe: ProbeDefinition) -> int:
    """Upsert probe within a SQLAlchemy session.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    probe : ProbeDefinition
        The probe definition to persist.

    Returns
    -------
    int
        The probe_id of the persisted probe.
    """
    # Check if probe already exists
    existing_probe = session.execute(
        select(Probe).where(Probe.chromophore_name == probe.name)
    ).scalar_one_or_none()

    if existing_probe:
        # Update existing probe
        existing_probe.chromophore_name = probe.name
        setattr(
            existing_probe,
            _column("_flr_probe_list.probe_origin"),
            _value_or_default(probe.probe_origin, "_flr_probe_list.probe_origin"),
        )
        setattr(
            existing_probe,
            _column("_flr_probe_list.probe_link_type"),
            _value_or_default(probe.probe_link_type, "_flr_probe_list.probe_link_type"),
        )
        existing_probe.fluorophore_type = getattr(
            probe,
            "fluorophore_type",
            _default("_flr_sample_probe_details.fluorophore_type"),
        )
        setattr(
            existing_probe,
            _column("_flr_probe_list.reactive_probe_flag"),
            _value_or_default(
                probe.reactive_probe_flag,
                "_flr_probe_list.reactive_probe_flag",
            ),
        )
        existing_probe.reactive_probe_name = probe.reactive_probe_name
        existing_probe.chromophore_center_atom = probe.chromophore_center_atom
        existing_probe.description = getattr(probe, "description", None)
        existing_probe.category = getattr(probe, "category", "other")
        _apply_probe_descriptors(session, existing_probe, probe)
        probe_id = existing_probe.probe_id
    else:
        # Create new probe
        new_probe = Probe(
            chromophore_name=probe.name,
            **{
                _column("_flr_probe_list.probe_origin"): _value_or_default(
                    probe.probe_origin,
                    "_flr_probe_list.probe_origin",
                ),
                _column("_flr_probe_list.probe_link_type"): _value_or_default(
                    probe.probe_link_type,
                    "_flr_probe_list.probe_link_type",
                ),
                _column("_flr_probe_list.reactive_probe_flag"): _value_or_default(
                    probe.reactive_probe_flag,
                    "_flr_probe_list.reactive_probe_flag",
                ),
            },
            fluorophore_type=getattr(
                probe,
                "fluorophore_type",
                _default("_flr_sample_probe_details.fluorophore_type"),
            ),
            reactive_probe_name=probe.reactive_probe_name,
            chromophore_center_atom=probe.chromophore_center_atom,
            description=getattr(probe, "description", None),
            category=getattr(probe, "category", "other"),
        )
        session.add(new_probe)
        session.flush()
        _apply_probe_descriptors(session, new_probe, probe)
        probe_id = new_probe.probe_id

    session.flush()
    return probe_id


def _add_probe_optical_data(
    session: Session, probe: Probe, definition: ProbeDefinition
) -> None:
    """Persist optical properties and spectra declared by a probe definition."""
    existing_properties = {
        prop.property_name
        for prop in probe.optical_properties
        if prop.deleted_at is None
    }
    properties = [
        ("absorption_wavelength", definition.absorption_wavelength_nm, "nm"),
        ("emission_wavelength", definition.emission_wavelength_nm, "nm"),
        ("quantum_yield", definition.quantum_yield, None),
        ("extinction_coefficient", definition.extinction_coefficient, "M-1cm-1"),
    ]
    for name, value, unit in properties:
        if value is None or name in existing_properties:
            continue
        session.add(
            OpticalProperty(
                probe_id=probe.probe_id,
                property_name=name,
                property_value=str(value),
                unit=unit,
            )
        )

    existing_spectra = {
        spectrum.spectrum_type
        for spectrum in probe.spectra
        if spectrum.deleted_at is None
    }
    spectra = [
        ("absorption", definition.absorption_spectrum),
        ("emission", definition.emission_spectrum),
    ]
    for spectrum_type, values in spectra:
        if not values or spectrum_type in existing_spectra:
            continue
        wavelengths, intensities = values
        session.add(
            Spectrum(
                probe_id=probe.probe_id,
                spectrum_type=spectrum_type,
                wavelengths=json.dumps(wavelengths).encode("utf-8"),
                intensity_values=json.dumps(intensities).encode("utf-8"),
                wavelength_unit="nm",
                intensity_unit="normalized",
            )
        )


def _derive_fluorophore_types(definition: SampleDefinition) -> list[str]:
    """Derive sample-probe fluorophore roles from FRET pair indices."""
    unspecified = _default("_flr_sample_probe_details.fluorophore_type")
    types = [unspecified] * len(definition.probes)
    for pair in definition.fret_pairs:
        if 0 <= pair.probe_1_index < len(types):
            types[pair.probe_1_index] = (
                unspecified
                if types[pair.probe_1_index] == "acceptor"
                else "donor"
            )
        if 0 <= pair.probe_2_index < len(types):
            types[pair.probe_2_index] = (
                unspecified
                if types[pair.probe_2_index] == "donor"
                else "acceptor"
            )
    return types
