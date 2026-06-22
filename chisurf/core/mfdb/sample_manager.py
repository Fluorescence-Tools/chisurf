"""Sample CRUD operations for MFDB.

This module provides high-level functions for creating, querying, and
linking samples. It wraps the low-level MFDatabase methods that touch
``flr_sample`` (the flrCIF-canonical sample table) and ``mfdb_edge``
so callers do not need to know the underlying table layout.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from chisurf.core.mfdb.models import (
    ENTITY_TYPES,
    COMMON_PROBE_NAMES,
    EntityDefinition,
    FretPairDefinition,
    ProbeDefinition,
    SampleDefinition,
    compute_forster_radius,
)
from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary
from chisurf.core.mfdb.repository import MFDatabase

logger = logging.getLogger(__name__)


def create_sample(db: MFDatabase, definition: SampleDefinition) -> str:
    """Create a sample from a ``SampleDefinition`` and return its sample ID.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    definition : SampleDefinition
        Sample definition to store. Supports both new list-based format
        (probes, fret_pairs) and legacy flat format (donor, acceptor, etc.).

    Returns
    -------
    str
        The canonical sample ID.

    Notes
    -----
    The operation is idempotent by sample name: if a sample with the same
    display name already exists, its existing sample ID is returned.

    This function populates multiple tables:
    - ``flr_sample``: canonical flrCIF sample table
    - ``entities``: entity information
    - ``entity_poly_seq``: entity sequence
    - ``probes``: probe/fluorophore records
    - ``flr_sample_condition``: sample conditions (pH, temperature, buffer)
    - ``entity_assembly``: entity assembly linking entity + probes
    - ``flr_poly_probe_position``: probe positions on entity
    - ``flr_sample_probe``: sample-probe mappings
    - ``optical_properties``: probe photophysical properties
    - ``spectra``: spectral data (if provided in ProbeDefinition)
    - ``flr_fret_forster_radius``: Förster radius (if FRET pairs provided)
    - ``flr_sample_key_value``: PDBx/flrCIF key-value metadata

    The fluorophore_type on flr_sample_probe is derived from FRET pair context:
    - A probe that appears as probe_1 in any pair gets type="donor"
    - A probe that appears as probe_2 in any pair gets type="acceptor"
    - A probe appearing in both roles (relay dye) gets type="unspecified"
    - A probe in no pairs gets type="unspecified"
    """
    if not definition.name or not definition.name.strip():
        raise ValueError("sample name is required")

    existing = db.conn.execute(
        "SELECT sample_id FROM flr_sample WHERE description = ? AND deleted_at IS NULL",
        (definition.name.strip(),),
    ).fetchone()
    if existing:
        return existing["sample_id"]

    sample_id = _unique_sample_id(db, _slugify(definition.name) or str(uuid.uuid4())[:8])

    from chisurf.core.mfdb.orm.sample_repository import create_sample_graph

    create_sample_graph(
        db,
        definition,
        sample_id=sample_id,
        display_name=definition.name.strip(),
        sample_type=_sample_type(definition),
        metadata_json=_json_dumps(_metadata_from_definition(definition)),
    )

    return sample_id


def get_sample(db: MFDatabase, sample_id: str) -> dict[str, Any] | None:
    """Return a sample by ID as a dictionary, or ``None`` if missing.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    sample_id : str
        Canonical sample ID.

    Returns
    -------
    dict or None
        Sample fields with metadata merged into the top level.

    """
    row = db.conn.execute(
        """SELECT sample_id, description, details, sample_type
           FROM flr_sample
           WHERE sample_id = ? AND deleted_at IS NULL""",
        (sample_id,),
    ).fetchone()
    if not row:
        return None
    return _sample_row_to_dict(row)


def list_samples(db: MFDatabase) -> list[dict[str, Any]]:
    """Return all active samples as dictionaries ordered by display name.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.

    Returns
    -------
    list of dict
        Sample dictionaries.

    """
    rows = db.conn.execute(
        """SELECT sample_id, description, details
           FROM flr_sample
           WHERE deleted_at IS NULL
           ORDER BY description COLLATE NOCASE"""
    ).fetchall()
    return [_sample_row_to_dict(row) for row in rows]


def find_sample_by_name(db: MFDatabase, name: str) -> str | None:
    """Return the sample ID for an exact display name, or ``None``.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    name : str
        Display name to match exactly.

    Returns
    -------
    str or None
        Matching sample ID.

    """
    row = db.conn.execute(
        """SELECT sample_id FROM flr_sample
           WHERE description = ? AND deleted_at IS NULL""",
        (name,),
    ).fetchone()
    return row["sample_id"] if row else None


def link_artifact_to_sample(db: MFDatabase, artifact_id: str, sample_id: str) -> None:
    """Create an idempotent ``measured_sample`` edge from an artifact to a sample.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    artifact_id : str
        Artifact ID to link.
    sample_id : str
        Sample ID to link to the artifact.

    """
    if not artifact_id or not sample_id:
        return
    existing = db.conn.execute(
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

    with db._transaction():
        db.add_edge(
            source_node_type="artifact",
            source_node_id=artifact_id,
            target_node_type="sample",
            target_node_id=sample_id,
            relationship_type="measured_sample",
        )


def get_sample_for_artifact(db: MFDatabase, artifact_id: str) -> str | None:
    """Return the sample ID linked to an artifact, or ``None``.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    artifact_id : str
        Artifact ID to look up.

    Returns
    -------
    str or None
        Linked sample ID.

    """
    row = db.conn.execute(
        """SELECT target_node_id AS sample_id FROM mfdb_edge
           WHERE source_node_type = 'artifact'
             AND source_node_id = ?
             AND target_node_type = 'sample'
             AND relationship_type = 'measured_sample'
             AND deleted_at IS NULL
           LIMIT 1""",
        (artifact_id,),
    ).fetchone()
    return row["sample_id"] if row else None


def get_artifacts_for_sample(db: MFDatabase, sample_id: str) -> list[str]:
    """Return all artifact IDs linked to a sample.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    sample_id : str
        Sample ID to query.

    Returns
    -------
    list of str
        Artifact IDs linked to the sample.

    """
    rows = db.conn.execute(
        """SELECT source_node_id AS artifact_id FROM mfdb_edge
           WHERE target_node_type = 'sample'
             AND target_node_id = ?
             AND relationship_type = 'measured_sample'
             AND deleted_at IS NULL
           ORDER BY source_node_id""",
        (sample_id,),
    ).fetchall()
    return [row["artifact_id"] for row in rows]


def get_sample_name(db: MFDatabase, sample_id: str) -> str:
    """Return a human-readable sample name for a sample ID.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    sample_id : str
        Sample ID to look up.

    Returns
    -------
    str
        Display name, or the sample ID if no sample row exists.

    """
    sample = get_sample(db, sample_id) or {}
    return sample.get("display_name") or sample.get("name") or sample_id


def _metadata_from_definition(definition: SampleDefinition) -> dict[str, Any]:
    """Convert a ``SampleDefinition`` into JSON-serializable metadata.

    Includes both new PRD-02 fields and legacy fields for backward compatibility.

    Parameters
    ----------
    definition : SampleDefinition
        Sample definition to convert.

    Returns
    -------
    dict
        Metadata dictionary with all sample attributes.

    """
    import dataclasses

    # Collect all probes
    all_probes = _collect_all_probes(definition)

    metadata = {
        # New structured fields (PRD-02)
        "entities": [dataclasses.asdict(e) for e in definition.entities],
        "probes": [
            {
                # Legacy fields for backward compatibility
                "name": p.name,
                "position": p.position,
                "position_label": p.position_label,
                "chain_id": p.chain_id,
                "residue_name": p.residue_name,
                # New flrCIF fields (R13-2)
                "entity_index": p.entity_index,
                "seq_id": p.seq_id,
                "comp_id": p.comp_id,
                "asym_id": p.asym_id,
                "atom_id": p.atom_id,
                "mutation_flag": p.mutation_flag,
                "modification_flag": p.modification_flag,
                "auth_name": p.auth_name,
                # Photophysical properties
                "absorption_wavelength_nm": p.absorption_wavelength_nm,
                "emission_wavelength_nm": p.emission_wavelength_nm,
                "quantum_yield": p.quantum_yield,
                "extinction_coefficient": p.extinction_coefficient,
                # Chemical descriptors
                "chromophore_smiles": p.chromophore_smiles,
                "chromophore_inchi": p.chromophore_inchi,
                "reactive_probe_smiles": p.reactive_probe_smiles,
                "reactive_probe_name": p.reactive_probe_name,
                "reactive_probe_flag": p.reactive_probe_flag,
                "probe_origin": p.probe_origin,
                "probe_link_type": p.probe_link_type,
                "chromophore_center_atom": p.chromophore_center_atom,
                "linker_smiles": p.linker_smiles,
                "ambiguous_stoichiometry": p.ambiguous_stoichiometry,
                "probe_stoichiometry": p.probe_stoichiometry,
            }
            for p in all_probes
        ],
        "fret_pairs": [
            {
                "probe_1_index": fp.probe_1_index,
                "probe_2_index": fp.probe_2_index,
                "forster_radius_nm": fp.forster_radius_nm,
                "reduced_forster_radius_nm": fp.reduced_forster_radius_nm,
                "kappa_squared": fp.kappa_squared,
                "refractive_index": fp.refractive_index,
                "overlap_integral": fp.overlap_integral,
            }
            for fp in definition.fret_pairs
        ],
        # Legacy flat fields for backward compatibility
        "entity_name": definition.entity_name,
        "entity_sequence": definition.entity_sequence,
        "entity_type": definition.entity_type,
        "buffer_description": definition.buffer_description,
        "ph": float(definition.ph) if definition.ph is not None else None,
        "temperature_k": float(definition.temperature_k) if definition.temperature_k is not None else None,
        "salt_concentration_m": float(definition.salt_concentration_m) if definition.salt_concentration_m is not None else None,
        "solvent_phase": definition.solvent_phase,
        "donor_probe_name": definition.donor_probe_name,
        "donor_position": int(definition.donor_position) if definition.donor_position is not None else None,
        "donor_position_label": definition.donor_position_label,
        "acceptor_probe_name": definition.acceptor_probe_name,
        "acceptor_position": int(definition.acceptor_position) if definition.acceptor_position is not None else None,
        "acceptor_position_label": definition.acceptor_position_label,
        **definition.extra,
    }
    return metadata


def _sample_row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a sample database row into a public dictionary.

    Parameters
    ----------
    row : sqlite3.Row
        Sample row.

    Returns
    -------
    dict
        Sample dictionary with metadata merged in.

    """
    data = dict(row)
    result: dict[str, Any] = {
        "sample_id": data.get("sample_id", ""),
        "name": data.get("description", ""),
        "display_name": data.get("description", ""),
        "sample_type": data.get("sample_type", ""),
    }
    return result


def _unique_sample_id(db: MFDatabase, base: str) -> str:
    """Return a sample ID based on ``base`` that does not already exist.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    base : str
        Candidate ID prefix.

    Returns
    -------
    str
        Unique sample ID.

    """
    candidate = base[:64] or str(uuid.uuid4())[:8]
    if not _sample_exists(db, candidate):
        return candidate
    for idx in range(2, 1000):
        suffixed = f"{candidate}_{idx}"
        if not _sample_exists(db, suffixed):
            return suffixed
    return f"{candidate}_{uuid.uuid4().hex[:8]}"


def _sample_exists(db: MFDatabase, sample_id: str) -> bool:
    """Return whether a sample ID already exists.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    sample_id : str
        Candidate sample ID.

    Returns
    -------
    bool
        ``True`` if the sample ID exists.

    """
    return db.conn.execute(
        "SELECT 1 FROM flr_sample WHERE sample_id = ?",
        (sample_id,),
    ).fetchone() is not None


def _collect_all_probes(definition: SampleDefinition) -> list[ProbeDefinition]:
    """Collect all probes from both new list format and legacy fields.

    Parameters
    ----------
    definition : SampleDefinition
        Sample definition.

    Returns
    -------
    list of ProbeDefinition
        All probes in order: new probes list first, then legacy donor/acceptor.
    """
    probes = list(definition.probes)

    # Add legacy structured fields if they exist
    if definition.donor is not None and definition.donor.name:
        # Check if donor is already in probes list
        if not any(p.name == definition.donor.name and p.position == definition.donor.position
                   for p in probes):
            probes.append(definition.donor)

    if definition.acceptor is not None and definition.acceptor.name:
        # Check if acceptor is already in probes list
        if not any(p.name == definition.acceptor.name and p.position == definition.acceptor.position
                   for p in probes):
            probes.append(definition.acceptor)

    # Add legacy flat fields if they exist and aren't already covered
    if definition.donor_probe_name and definition.donor_position is not None:
        if not any(p.name == definition.donor_probe_name for p in probes):
            probes.append(ProbeDefinition(
                name=definition.donor_probe_name,
                position=definition.donor_position,
                position_label=definition.donor_position_label,
            ))

    if definition.acceptor_probe_name and definition.acceptor_position is not None:
        if not any(p.name == definition.acceptor_probe_name for p in probes):
            probes.append(ProbeDefinition(
                name=definition.acceptor_probe_name,
                position=definition.acceptor_position,
                position_label=definition.acceptor_position_label,
            ))

    return probes


def _insert_all_probes(db: MFDatabase, probes: list[ProbeDefinition]) -> list[int]:
    """Insert all probe records and return their IDs.

    Uses repository.find_or_add_probe() for proper deduplication and audit fields.
    Persists all chemical fields from ProbeDefinition to canonical probe tables (R14-4).

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    probes : list of ProbeDefinition
        List of probes to insert.

    Returns
    -------
    list of int
        List of probe_ids in the same order as input probes.
    """
    probe_ids = []

    for probe in probes:
        if not probe.name:
            probe_ids.append(0)  # Invalid probe
            continue

        # Use repository method for proper deduplication and audit fields
        # Pass all chemical fields from ProbeDefinition
        try:
            probe_id = db.find_or_add_probe(
                name=probe.name,
                category="other",
                description=probe.name,  # Use name as description if no better option
                reactive_probe_flag=probe.reactive_probe_flag,
                reactive_probe_name=probe.reactive_probe_name or None,
                probe_origin=probe.probe_origin,
                probe_link_type=probe.probe_link_type,
                chromophore_center_atom=probe.chromophore_center_atom or None,
            )

            # TODO: R14-4 - Also persist SMILES/InChI to chem_descriptors table
            # For now, we persist the basic chemical fields that are columns in probes table

            probe_ids.append(probe_id)
        except Exception:
            # Fallback: try to find existing probe
            existing = db.conn.execute(
                "SELECT probe_id FROM probes WHERE chromophore_name = ? AND deleted_at IS NULL",
                (probe.name,),
            ).fetchone()
            if existing:
                probe_ids.append(existing["probe_id"])
            else:
                probe_ids.append(0)

    return probe_ids


def _insert_entity(db: MFDatabase, sample_id: str, definition: SampleDefinition) -> list[str]:
    """Insert entity metadata and return list of entity IDs.

    Supports both the new entities list (PRD-02) and legacy flat fields
    for backward compatibility.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    sample_id : str
        Sample ID used to derive a stable entity ID.
    definition : SampleDefinition
        Sample definition.

    Returns
    -------
    list of str
        List of entity IDs in the same order as definition.entities.
        If no entities and no legacy fields, returns empty list.

    """
    entity_ids = []

    # Use new entities list if available (PRD-02)
    if definition.entities:
        for i, entity in enumerate(definition.entities):
            entity_id = _slugify(entity.name) or f"{sample_id}_entity_{i}"
            db.add_entity(
                entity_id,
                name=entity.name,
                sequence=list(entity.sequence) if entity.sequence else None,
                entity_type=entity.entity_type or "polymer",
                details=entity.details or None,
            )
            entity_ids.append(entity_id)
    # Fall back to legacy flat fields for backward compatibility
    elif definition.entity_name:
        entity_id = _slugify(definition.entity_name) or f"{sample_id}_entity"
        db.add_entity(
            entity_id,
            name=definition.entity_name,
            sequence=list(definition.entity_sequence) if definition.entity_sequence else None,
            entity_type=definition.entity_type or "polymer",
            details=definition.description or None,
        )
        entity_ids.append(entity_id)

    return entity_ids


def _insert_all_probe_positions(
    db: MFDatabase, entity_ids: list[str], probes: list[ProbeDefinition], probe_ids: list[int]
) -> list[int]:
    """Insert probe positions for all probes with valid positions.

    Supports multi-entity samples where each probe references its entity
    via entity_index (PRD-02).

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    entity_ids : list of str
        List of entity IDs (in the same order as SampleDefinition.entities).
    probes : list of ProbeDefinition
        List of probes.
    probe_ids : list of int
        List of probe IDs corresponding to probes.

    Returns
    -------
    list of int
        List of position_ids in the same order as input probes.
        0 for probes without valid positions or invalid entity_index.
    """
    position_ids = []

    for i, (probe, probe_id) in enumerate(zip(probes, probe_ids)):
        if probe_id == 0:  # Invalid probe
            position_ids.append(0)
            continue

        # Get the entity_id for this probe based on entity_index
        # If entity_index is out of range or entity_ids is empty, skip
        if not entity_ids or probe.entity_index < 0 or probe.entity_index >= len(entity_ids):
            position_ids.append(0)
            continue

        entity_id = entity_ids[probe.entity_index]

        # Use new flrCIF position fields (PRD-02)
        # seq_id is the residue number
        if probe.seq_id is None:
            position_ids.append(0)  # No position
            continue

        # Check if position already exists for this probe and entity
        existing = db.conn.execute(
            """SELECT id FROM flr_poly_probe_position
               WHERE probe_id = ? AND entity_id = ? AND residue_number = ?
               AND asym_id = ? AND deleted_at IS NULL""",
            (probe_id, entity_id, probe.seq_id, probe.asym_id or "A"),
        ).fetchone()

        if existing:
            position_ids.append(existing["id"])
            continue

        # Insert new position with all flrCIF fields
        now = _utc_now(db)
        cursor = db.conn.execute(
            """INSERT INTO flr_poly_probe_position
               (probe_id, entity_id, residue_number, asym_id, residue_name,
                atom_id, mutation_flag, modification_flag, auth_name,
                description, created_at, updated_at, deleted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                probe_id,
                entity_id,
                probe.seq_id,
                probe.asym_id or "A",
                probe.comp_id or None,
                probe.atom_id or None,
                probe.mutation_flag or "no",
                probe.modification_flag or "no",
                probe.auth_name or None,
                probe.position_label or None,  # legacy description field
                now,
                now,
                None,
            ),
        )
        position_ids.append(cursor.lastrowid)

    return position_ids


def _derive_fluorophore_types(
    definition: SampleDefinition, num_probes: int
) -> list[str]:
    """Derive fluorophore_type for each probe based on FRET pair context.

    A probe that appears as donor in one pair and acceptor in another (relay dye)
    gets type="unspecified".

    Parameters
    ----------
    definition : SampleDefinition
        Sample definition.
    num_probes : int
        Total number of probes.

    Returns
    -------
    list of str
        fluorophore_type for each probe ("donor", "acceptor", or "unspecified").
    """
    # Initialize all as unspecified
    types = ["unspecified"] * num_probes

    # Process FRET pairs from new format
    for pair in definition.fret_pairs:
        if pair.probe_1_index < num_probes:
            # Check if already marked as acceptor (relay dye)
            if types[pair.probe_1_index] == "acceptor":
                types[pair.probe_1_index] = "unspecified"
            else:
                types[pair.probe_1_index] = "donor"
        if pair.probe_2_index < num_probes:
            # If already marked as donor (relay dye), set to unspecified
            if types[pair.probe_2_index] == "donor":
                types[pair.probe_2_index] = "unspecified"
            else:
                types[pair.probe_2_index] = "acceptor"

    # Legacy support: if we have exactly 2 probes and no explicit fret_pairs,
    # assume they form a single FRET pair
    if (num_probes == 2 and not definition.fret_pairs and
        not definition.probes and (definition.donor or definition.acceptor)):
        types[0] = "donor"
        types[1] = "acceptor"

    return types


def _get_probe_description(
    probe: ProbeDefinition | None, index: int, definition: SampleDefinition
) -> str | None:
    """Get a description for a probe.

    Parameters
    ----------
    probe : ProbeDefinition or None
        The probe.
    index : int
        Probe index.
    definition : SampleDefinition
        Sample definition.

    Returns
    -------
    str or None
        Description string.
    """
    if probe is not None:
        parts = []
        if probe.position is not None:
            parts.append(f"pos={probe.position}")
        if probe.position_label:
            parts.append(probe.position_label)
        if probe.chain_id:
            parts.append(f"chain={probe.chain_id}")
        if probe.residue_name:
            parts.append(probe.residue_name)
        return ", ".join(parts) if parts else None

    # Try legacy fields
    if index == 0 and definition.donor_probe_name:
        return definition.donor_probe_name
    if index == 1 and definition.acceptor_probe_name:
        return definition.acceptor_probe_name

    return None


def _insert_fret_pairs(
    db: MFDatabase,
    sample_id: str,
    definition: SampleDefinition,
    probe_ids: list[int]
) -> None:
    """Insert Förster radius records for all FRET pairs.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    sample_id : str
        Sample ID.
    definition : SampleDefinition
        Sample definition.
    probe_ids : list of int
        List of probe IDs.
    """
    # Process explicit FRET pairs from new format
    for i, pair in enumerate(definition.fret_pairs):
        if pair.probe_1_index >= len(probe_ids) or pair.probe_2_index >= len(probe_ids):
            logger.warning(f"FRET pair {i}: invalid probe indices")
            continue

        if probe_ids[pair.probe_1_index] == 0 or probe_ids[pair.probe_2_index] == 0:
            logger.warning(f"FRET pair {i}: invalid probe IDs")
            continue

        forster_id = f"{sample_id}_forster_{i}"

        # Try to compute Förster radius from spectra if not provided
        forster_radius = pair.forster_radius_nm
        overlap_integral = pair.overlap_integral

        if forster_radius is None and pair.probe_1_index < len(definition.probes):
            donor_probe = definition.probes[pair.probe_1_index]
            acceptor_probe = definition.probes[pair.probe_2_index]

            if (donor_probe.emission_spectrum and acceptor_probe.absorption_spectrum and
                donor_probe.quantum_yield is not None):
                try:
                    import numpy as np
                    forster_radius, overlap_integral = compute_forster_radius(
                        donor_probe.emission_spectrum,
                        acceptor_probe.absorption_spectrum,
                        donor_probe.quantum_yield,
                        kappa_squared=pair.kappa_squared,
                        refractive_index=pair.refractive_index,
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute Förster radius: {e}")

        db.add_fret_forster_radius(
            forster_id,
            sample_id,
            probe_ids[pair.probe_1_index],
            probe_ids[pair.probe_2_index],
            forster_radius or 0.0,
            kappa_squared=pair.kappa_squared,
            refractive_index=pair.refractive_index,
            details=f"Förster radius for {definition.name} pair {i}",
        )

    # Legacy support: if we have old-style forster_radius_nm but no fret_pairs
    # Note: This checks for the old legacy field which may not exist in new definitions
    old_forster = getattr(definition, 'forster_radius_nm', None)
    old_kappa = getattr(definition, 'kappa_squared', 2.0/3.0)
    old_refractive = getattr(definition, 'refractive_index', 1.4)

    if (old_forster is not None and not definition.fret_pairs and
        len(probe_ids) >= 2):
        forster_id = f"{sample_id}_forster_legacy"
        db.add_fret_forster_radius(
            forster_id,
            sample_id,
            probe_ids[0],
            probe_ids[1],
            old_forster,
            kappa_squared=old_kappa,
            refractive_index=old_refractive,
            details=f"Förster radius for {definition.name} (legacy)",
        )


def _insert_optical_properties(
    db: MFDatabase, probe_id: int, probe: ProbeDefinition | None
) -> None:
    """Insert optical properties for a probe.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    probe_id : int
        Probe ID.
    probe : ProbeDefinition or None
        Probe definition.
    """
    if probe is None or probe_id == 0:
        return

    now = _utc_now(db)

    # Map of property names to (value, unit)
    properties = [
        ("absorption_wavelength", probe.absorption_wavelength_nm, "nm"),
        ("emission_wavelength", probe.emission_wavelength_nm, "nm"),
        ("quantum_yield", probe.quantum_yield, None),
        ("extinction_coefficient", probe.extinction_coefficient, "M-1cm-1"),
    ]

    for prop_name, value, unit in properties:
        if value is None:
            continue

        # Check if property already exists
        existing = db.conn.execute(
            """SELECT 1 FROM optical_properties
               WHERE probe_id = ? AND property_name = ? AND deleted_at IS NULL""",
            (probe_id, prop_name),
        ).fetchone()

        if existing:
            continue

        db.conn.execute(
            """INSERT INTO optical_properties
               (probe_id, property_name, property_value, unit, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (probe_id, prop_name, str(value), unit, now, now),
        )


def _insert_spectra(db: MFDatabase, probe_id: int, probe: ProbeDefinition | None) -> None:
    """Insert spectral data for a probe.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    probe_id : int
        Probe ID.
    probe : ProbeDefinition or None
        Probe definition.
    """
    if probe is None or probe_id == 0:
        return

    now = _utc_now(db)

    if probe.absorption_spectrum:
        wls, ints = probe.absorption_spectrum
        # Check if spectrum already exists
        existing = db.conn.execute(
            """SELECT 1 FROM spectra
               WHERE probe_id = ? AND spectrum_type = 'absorption' AND deleted_at IS NULL""",
            (probe_id,),
        ).fetchone()

        if not existing:
            db.conn.execute(
                """INSERT INTO spectra
                   (probe_id, spectrum_type, wavelengths, intensity_values,
                    wavelength_unit, intensity_unit, created_at, updated_at)
                   VALUES (?, 'absorption', ?, ?, 'nm', 'normalized', ?, ?)""",
                (probe_id, json.dumps(wls), json.dumps(ints), now, now),
            )

    if probe.emission_spectrum:
        wls, ints = probe.emission_spectrum
        existing = db.conn.execute(
            """SELECT 1 FROM spectra
               WHERE probe_id = ? AND spectrum_type = 'emission' AND deleted_at IS NULL""",
            (probe_id,),
        ).fetchone()

        if not existing:
            db.conn.execute(
                """INSERT INTO spectra
                   (probe_id, spectrum_type, wavelengths, intensity_values,
                    wavelength_unit, intensity_unit, created_at, updated_at)
                   VALUES (?, 'emission', ?, ?, 'nm', 'normalized', ?, ?)""",
                (probe_id, json.dumps(wls), json.dumps(ints), now, now),
            )


def _insert_condition(db: MFDatabase, sample_id: str, definition: SampleDefinition) -> str | None:
    """Insert a sample condition when condition metadata is supplied.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    sample_id : str
        Sample ID used as condition ID.
    definition : SampleDefinition
        Sample definition.

    Returns
    -------
    str or None
        Condition ID if created, None otherwise.

    """
    has_condition = any(
        (
            definition.buffer_description,
            definition.ph is not None,
            definition.temperature_k is not None,
            definition.salt_concentration_m is not None,
        )
    )
    if not has_condition:
        return None

    condition_id = f"{sample_id}_condition"
    now = _utc_now(db)
    db.conn.execute(
        """INSERT OR REPLACE INTO flr_sample_condition
           (condition_id, ph, temperature, ionic_strength, buffer_composition,
            details, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            condition_id,
            definition.ph,
            definition.temperature_k,
            definition.salt_concentration_m,
            definition.buffer_description or None,
            definition.description or None,
            now,
            now,
        ),
    )
    return condition_id


def _sample_type(definition: SampleDefinition) -> str:
    """Return the canonical sample type for a definition.

    Parameters
    ----------
    definition : SampleDefinition
        Sample definition.

    Returns
    -------
    str
        Sample type string.

    """
    if definition.entity_type in {"protein", "dna", "rna"}:
        return definition.entity_type
    return "physical_sample"


def _slugify(name: str) -> str:
    """Convert a display name into a safe ID prefix.

    Parameters
    ----------
    name : str
        Display name to slugify.

    Returns
    -------
    str
        Safe ASCII ID prefix.

    """
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug[:64]



def _json_dumps(value: Any) -> str:
    """Serialize a JSON-compatible value to a compact string.

    Parameters
    ----------
    value : Any
        Value to serialize.

    Returns
    -------
    str
        JSON string.

    """
    return json.dumps(value, sort_keys=True)


def _json_loads(value: Any) -> dict[str, Any]:
    """Load a JSON object from a database field.

    Parameters
    ----------
    value : Any
        JSON string or ``None``.

    Returns
    -------
    dict
        Parsed object, or an empty dictionary.

    """
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _utc_now(db: MFDatabase) -> str:
    """Return the current UTC timestamp using SQLite's clock.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.

    Returns
    -------
    str
        Current UTC timestamp.

    """
    return db.conn.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]


def _insert_sample_key_values(db: MFDatabase, sample_id: str, definition: SampleDefinition) -> None:
    """Insert PDBx/flrCIF key-value metadata for a sample.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    sample_id : str
        Sample ID.
    definition : SampleDefinition
        Sample definition.
    """
    now = _utc_now(db)

    # Auto-populate standard key-values
    key_values = []

    # flr.solvent_phase
    if definition.solvent_phase:
        key_values.append(("flr.solvent_phase", definition.solvent_phase, None))

    # flr.num_of_probes
    all_probes = _collect_all_probes(definition)
    num_probes = len(all_probes)
    if num_probes > 0:
        key_values.append(("flr.num_of_probes", str(num_probes), None))

    # pdbx.entity_type
    if definition.entity_type:
        key_values.append(("pdbx.entity_type", definition.entity_type, None))

    # chisurf.sample_origin
    key_values.append(("chisurf.sample_origin", "user_created", None))

    # Insert all key-values
    for key, value, details in key_values:
        db.conn.execute(
            """INSERT OR REPLACE INTO flr_sample_key_value
               (sample_id, key, value, details, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (sample_id, key, value, details, now, now),
        )


def set_sample_metadata(db: MFDatabase, sample_id: str, key: str, value: str, details: str | None = None) -> None:
    """Set a PDBx/flrCIF key-value pair on a sample.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    sample_id : str
        Sample ID.
    key : str
        PDBx-style key (e.g. 'pdbx.sample_type', 'flr.solvent_phase').
    value : str
        Value to set.
    details : str, optional
        Additional details.
    """
    now = _utc_now(db)

    # Validate against dictionary if available
    try:
        dic = MmcifDictionary.load_bundled()
        # Extract category and attribute from key
        if "." in key:
            category, attribute = key.split(".", 1)
            err = dic.validate_value(f"_{category}.{attribute}", value)
            if err:
                logger.warning(f"Metadata validation warning: {err}")
    except Exception:
        pass  # Dictionary not available, skip validation

    db.conn.execute(
        """INSERT OR REPLACE INTO flr_sample_key_value
           (sample_id, key, value, details, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (sample_id, key, value, details, now, now),
    )


def get_sample_full_description(db: MFDatabase, sample_id: str) -> dict[str, Any] | None:
    """Return the complete sample description including entity, probes, positions,
    condition, and Förster radius.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    sample_id : str
        Sample ID to query.

    Returns
    -------
    dict or None
        Complete sample description as a nested dictionary, or None if not found.

    The returned dictionary has the structure:
    {
        "sample_id": "...",
        "display_name": "...",
        "description": "...",
        "entity": {
            "entity_id": "...",
            "type": "protein",
            "common_name": "...",
            "sequence": "...",
        },
        "entities": [  # For multi-entity samples (PRD-02)
            {
                "entity_id": "...",
                "type": "protein",
                "common_name": "...",
                "sequence": "...",
            },
            ...
        ],
        "condition": {
            "ph": 7.4,
            "temperature_k": 298.15,
            "ionic_strength": 0.15,
            "buffer_composition": "PBS",
        },
        "probes": [
            {
                "probe_name": "Cy3B",
                "fluorophore_type": "donor",
                "position": {"residue_number": 5, "chain_id": "A", "residue_name": "CYS"},
                "properties": {"absorption_wavelength": {"value": "559", "unit": "nm"}, ...},
            },
            ...
        ],
        "fret_pairs": [
            {
                "donor_probe": "Cy3B",
                "acceptor_probe": "ATTO647N",
                "forster_radius_nm": 5.1,
                "kappa_squared": 0.666667,
                "refractive_index": 1.4,
            },
            ...
        ],
        "key_values": [{"key": "pdbx.sample_type", "value": "protein"}, ...],
    }
    """
    try:
        from chisurf.core.mfdb.orm.sample_repository import get_sample_graph

        graph = get_sample_graph(db, sample_id)
        if graph is not None:
            return _full_description_from_sample_graph(db, sample_id, graph)
    except Exception as exc:
        logger.warning(
            "Falling back to raw SQL sample description for %s: %s",
            sample_id,
            exc,
        )

    # Get basic sample info
    sample = get_sample(db, sample_id)
    if sample is None:
        return None

    result = {
        "sample_id": sample_id,
        "display_name": sample.get("display_name", ""),
        "description": sample.get("description", ""),
        "sample_type": sample.get("sample_type", ""),
        "metadata": sample.get("metadata_json", {}),
    }

    # Get entity info (list for multi-entity samples)
    entities = _get_entity_info(db, sample_id)
    if entities:
        result["entities"] = entities
        # Backward compatibility: also provide first entity as "entity"
        result["entity"] = entities[0] if entities else None

    # Get condition info
    condition = _get_condition_info(db, sample_id)
    if condition:
        result["condition"] = condition

    # Get probe info with positions and properties
    probes = _get_probe_info(db, sample_id)
    result["probes"] = probes

    # Get FRET pairs
    fret_pairs = _get_fret_pairs_info(db, sample_id)
    result["fret_pairs"] = fret_pairs

    # Get key-values
    key_values = db.get_sample_key_values(sample_id)
    result["key_values"] = key_values

    return result


def _full_description_from_sample_graph(
    db: MFDatabase, sample_id: str, graph: dict[str, Any]
) -> dict[str, Any] | None:
    """Convert ORM sample graph output to the public full-description shape."""
    sample_index = get_sample(db, sample_id) or {}
    sample = graph.get("sample", {})
    if not sample:
        return None

    result: dict[str, Any] = {
        "sample_id": sample_id,
        "display_name": sample_index.get("display_name") or sample_id,
        "description": sample.get("description") or "",
        "sample_type": sample_index.get("sample_type") or "",
        "metadata": sample_index.get("metadata_json", {}),
    }

    entities = []
    for entity in graph.get("entities", []):
        sequence = "".join(
            seq.get("mon_id", "") for seq in entity.get("sequences", [])
        )
        entity_info = {
            "entity_id": entity.get("entity_id", ""),
            "type": entity.get("type", ""),
            "common_name": entity.get("common_name", ""),
            "description": entity.get("description", ""),
        }
        if sequence:
            entity_info["sequence"] = sequence
        entities.append(entity_info)
    if entities:
        result["entities"] = entities
        result["entity"] = entities[0]

    condition = graph.get("condition")
    if condition:
        result["condition"] = {
            "condition_id": condition.get("condition_id"),
            "ph": condition.get("ph"),
            "temperature_k": condition.get("temperature"),
            "ionic_strength": condition.get("ionic_strength"),
            "salt_concentration_m": condition.get("ionic_strength"),
            "buffer_composition": condition.get("buffer_composition"),
            "details": condition.get("details"),
        }

    probes = []
    for probe in graph.get("probes", []):
        probe_info: dict[str, Any] = {
            "probe_name": probe.get("name", ""),
            "fluorophore_type": probe.get("fluorophore_type", "unspecified"),
            "description": probe.get("description"),
        }
        position = probe.get("position")
        if position:
            probe_info["position"] = {
                "residue_number": position.get("residue_number"),
                "chain_id": position.get("asym_id"),
                "residue_name": position.get("residue_name"),
                "description": None,
                "entity_index": None,
                "seq_id": position.get("residue_number"),
                "comp_id": position.get("residue_name"),
                "asym_id": position.get("asym_id"),
                "atom_id": position.get("atom_id"),
                "mutation_flag": position.get("mutation_flag"),
                "modification_flag": position.get("modification_flag"),
                "auth_name": position.get("auth_name"),
                "entity_id": position.get("entity_id"),
            }
        if probe.get("optical_properties"):
            probe_info["properties"] = {
                prop.get("property_name"): {
                    "value": prop.get("property_value"),
                    "unit": prop.get("unit"),
                }
                for prop in probe.get("optical_properties", [])
                if prop.get("property_name")
            }
        probes.append(probe_info)
    result["probes"] = probes

    result["fret_pairs"] = [
        {
            "forster_radius_id": pair.get("forster_radius_id"),
            "sample_id": pair.get("sample_id"),
            "donor_probe": pair.get("donor_probe", ""),
            "acceptor_probe": pair.get("acceptor_probe", ""),
            "forster_radius_nm": pair.get("forster_radius"),
            "kappa_squared": pair.get("kappa_squared"),
            "refractive_index": pair.get("index_of_refraction"),
            "overlap_integral": pair.get("overlap_integral"),
            "details": pair.get("details"),
        }
        for pair in graph.get("fret_pairs", [])
    ]

    result["key_values"] = graph.get("key_values", [])
    return result


def _get_entity_info(db: MFDatabase, sample_id: str) -> list[dict[str, Any]]:
    """Get entity information for a sample.

    Retrieves entity information by tracing the relationship:
    sample -> flr_sample_probe -> flr_poly_probe_position -> entity_id -> entities

    For multi-entity samples (PRD-02), returns all entities associated with the sample.
    Falls back to pattern matching on entities table if no probe positions exist.

    Returns
    -------
    list of dict
        List of entity information dictionaries, each with keys:
        entity_id, type, common_name, description, sequence (if available).
    """
    entities = []

    # Get all probe positions for this sample to find all unique entity_ids
    probe_pos_rows = db.conn.execute(
        """SELECT DISTINCT ppp.entity_id
           FROM flr_sample_probe AS sp
           JOIN flr_poly_probe_position AS ppp ON sp.poly_probe_position_id = ppp.id
           WHERE sp.sample_id = ? AND sp.deleted_at IS NULL AND ppp.deleted_at IS NULL""",
        (sample_id,),
    ).fetchall()

    entity_ids = [dict(r)["entity_id"] for r in probe_pos_rows if dict(r).get("entity_id")]

    # If no entities found via probe positions, try fallback
    if not entity_ids:
        # Try pattern matching on entities table - entity_ids for this sample
        # are typically f"{sample_id}_entity" or f"{sample_id}_entity_{i}"
        # Use exact prefix match to avoid substring matching issues
        entity_rows = db.conn.execute(
            """SELECT entity_id
               FROM entities WHERE entity_id LIKE ? AND deleted_at IS NULL""",
            (f"{sample_id}_entity%",),
        ).fetchall()
        entity_ids = [dict(r)["entity_id"] for r in entity_rows if dict(r).get("entity_id")]

        # Also try the slugified entity name pattern
        if not entity_ids:
            entity_rows = db.conn.execute(
                """SELECT entity_id
                   FROM entities WHERE entity_id = ? AND deleted_at IS NULL""",
                (f"{sample_id}_entity",),
            ).fetchall()
            entity_ids = [dict(r)["entity_id"] for r in entity_rows if dict(r).get("entity_id")]

    # Get details for each entity
    for entity_id in entity_ids:
        # Get entity details
        entity_row = db.conn.execute(
            """SELECT type, common_name, description
               FROM entities WHERE entity_id = ? AND deleted_at IS NULL
               LIMIT 1""",
            (entity_id,),
        ).fetchone()

        if not entity_row:
            continue

        entity_info = {
            "entity_id": entity_id,
            "type": dict(entity_row)["type"],
            "common_name": dict(entity_row)["common_name"] or "",
            "description": dict(entity_row)["description"] or "",
        }

        # Get sequence
        seq_rows = db.conn.execute(
            """SELECT mon_id FROM entity_poly_seq
               WHERE entity_id = ? AND deleted_at IS NULL
               ORDER BY num""",
            (entity_id,),
        ).fetchall()

        if seq_rows:
            sequence = "".join(dict(r)["mon_id"] for r in seq_rows)
            entity_info["sequence"] = sequence

        entities.append(entity_info)

    return entities


def _get_condition_info(db: MFDatabase, sample_id: str) -> dict[str, Any] | None:
    """Get condition information for a sample."""
    # Condition IDs are constructed as f"{sample_id}_condition"
    condition_id = f"{sample_id}_condition"
    row = db.conn.execute(
        """SELECT * FROM flr_sample_condition
           WHERE condition_id = ? AND deleted_at IS NULL
           LIMIT 1""",
        (condition_id,),
    ).fetchone()

    if not row:
        return None

    row_dict = dict(row)
    return {
        "condition_id": row_dict.get("condition_id"),
        "ph": row_dict.get("ph"),
        "temperature_k": row_dict.get("temperature"),
        "ionic_strength": row_dict.get("ionic_strength"),
        "salt_concentration_m": row_dict.get("ionic_strength"),  # alias for ionic_strength
        "buffer_composition": row_dict.get("buffer_composition"),
        "details": row_dict.get("details"),
    }


def _get_probe_info(db: MFDatabase, sample_id: str) -> list[dict[str, Any]]:
    """Get probe information for a sample including positions and properties."""
    probe_mappings = db.get_sample_probe_mappings(sample_id)
    probes = []

    for mapping in probe_mappings:
        probe_info = {
            "probe_name": mapping.get("chromophore_name", ""),
            "fluorophore_type": mapping.get("fluorophore_type", "unspecified"),
            "description": mapping.get("description"),
        }

        # Get position info
        position_id = mapping.get("poly_probe_position_id")
        if position_id:
            pos_row = db.conn.execute(
                """SELECT * FROM flr_poly_probe_position
                   WHERE id = ? AND deleted_at IS NULL""",
                (position_id,),
            ).fetchone()
            if pos_row:
                pos_dict = dict(pos_row)
                probe_info["position"] = {
                    # Legacy fields (kept for backward compatibility)
                    "residue_number": pos_dict.get("residue_number"),
                    "chain_id": pos_dict.get("asym_id"),
                    "residue_name": pos_dict.get("residue_name"),
                    "description": pos_dict.get("description"),
                    # New flrCIF fields (PRD-02)
                    "entity_index": pos_dict.get("entity_index"),
                    "seq_id": pos_dict.get("seq_id"),
                    "comp_id": pos_dict.get("comp_id"),
                    "asym_id": pos_dict.get("asym_id"),
                    "atom_id": pos_dict.get("atom_id"),
                    "mutation_flag": pos_dict.get("mutation_flag"),
                    "modification_flag": pos_dict.get("modification_flag"),
                    "auth_name": pos_dict.get("auth_name"),
                }

        # Get optical properties
        props_rows = db.conn.execute(
            """SELECT property_name, property_value, unit FROM optical_properties
               WHERE probe_id = ? AND deleted_at IS NULL""",
            (mapping.get("probe_id"),),
        ).fetchall()

        if props_rows:
            probe_info["properties"] = {
                prop["property_name"]: {
                    "value": prop["property_value"],
                    "unit": prop["unit"],
                }
                for prop in [dict(r) for r in props_rows]
            }

        # Get spectra
        spectra_rows = db.conn.execute(
            """SELECT spectrum_type, wavelengths, intensity_values
               FROM spectra
               WHERE probe_id = ? AND deleted_at IS NULL""",
            (mapping.get("probe_id"),),
        ).fetchall()

        if spectra_rows:
            probe_info["spectra"] = {
                row["spectrum_type"]: {
                    "wavelengths": json.loads(row["wavelengths"]),
                    "intensity_values": json.loads(row["intensity_values"]),
                }
                for row in [dict(r) for r in spectra_rows]
            }

        probes.append(probe_info)

    return probes


def _get_fret_pairs_info(db: MFDatabase, sample_id: str) -> list[dict[str, Any]]:
    """Get FRET pair information for a sample."""
    # First, get all probe_ids for this sample
    probe_mappings = db.get_sample_probe_mappings(sample_id)
    sample_probe_ids = [m.get("probe_id") for m in probe_mappings if m.get("probe_id")]

    if not sample_probe_ids:
        return []

    # Query FRET pairs for this specific sample
    rows = db.conn.execute(
        """SELECT * FROM flr_fret_forster_radius
           WHERE sample_id = ? AND deleted_at IS NULL""",
        (sample_id,)
    ).fetchall()

    fret_pairs = []
    for row in rows:
        row_dict = dict(row)

        # Get probe names for donor and acceptor using correct column names
        donor_name = _get_probe_name_by_id(db, row_dict.get("donor_probe_id"))
        acceptor_name = _get_probe_name_by_id(db, row_dict.get("acceptor_probe_id"))

        fret_pairs.append({
            "id": row_dict.get("id"),
            "forster_radius_id": row_dict.get("forster_radius_id"),
            "donor_probe": donor_name,
            "acceptor_probe": acceptor_name,
            "forster_radius_nm": row_dict.get("forster_radius"),
            "kappa_squared": row_dict.get("kappa_squared"),
            "refractive_index": row_dict.get("index_of_refraction"),
            "overlap_integral": row_dict.get("overlap_integral"),
            "details": row_dict.get("details"),
        })

    return fret_pairs


def _get_probe_name_by_id(db: MFDatabase, probe_id: int | None) -> str:
    """Get probe name by probe_id."""
    if probe_id is None:
        return ""

    row = db.conn.execute(
        """SELECT chromophore_name FROM probes
           WHERE probe_id = ? AND deleted_at IS NULL""",
        (probe_id,),
    ).fetchone()

    if row:
        return dict(row)["chromophore_name"]
    return ""


def validate_sample_for_export(db: MFDatabase, sample_id: str) -> list[str]:
    """Check that a sample has enough data for valid flrCIF export.

    Parameters
    ----------
    db : MFDatabase
        Active MFDB connection.
    sample_id : str
        Sample ID to validate.

    Returns
    -------
    list of str
        List of warnings/missing fields. Empty list = export-ready.
    """
    warnings = []

    # Get full description
    desc = get_sample_full_description(db, sample_id)
    if desc is None:
        return [f"Sample {sample_id} not found"]

    # Check for entity
    if "entity" not in desc or not desc["entity"].get("entity_id"):
        warnings.append("Missing entity information")

    # Check for at least one probe with position
    probes_with_positions = [
        p for p in desc.get("probes", [])
        if p.get("position") and p.get("position").get("residue_number") is not None
    ]
    if len(probes_with_positions) < 1:
        warnings.append("Missing probe positions")

    # Check for sample-probe mappings with fluorophore_type
    # Only warn about unspecified probes that are NOT in any FRET pair
    # (relay dyes correctly have unspecified type)
    probes = desc.get("probes", [])
    fret_pairs = desc.get("fret_pairs", [])

    # Get all probe names that appear in FRET pairs
    probes_in_fret_pairs = set()
    for pair in fret_pairs:
        if pair.get("donor_probe"):
            probes_in_fret_pairs.add(pair["donor_probe"])
        if pair.get("acceptor_probe"):
            probes_in_fret_pairs.add(pair["acceptor_probe"])

    # Count unspecified probes that are not in any FRET pair
    unspecified_orphan_count = sum(
        1 for p in probes
        if p.get("fluorophore_type") == "unspecified" and p.get("probe_name") not in probes_in_fret_pairs
    )
    if unspecified_orphan_count > 0:
        warnings.append(f"{unspecified_orphan_count} probes have unspecified fluorophore_type and are not in any FRET pair")

    # Check for sample condition
    condition = desc.get("condition", {})
    if condition.get("ph") is None:
        warnings.append("Missing pH in sample condition")
    if condition.get("temperature_k") is None:
        warnings.append("Missing temperature in sample condition")

    # Recommended fields
    if not desc.get("entity", {}).get("sequence"):
        warnings.append("Recommended: entity sequence not provided")

    for probe in probes:
        if "properties" not in probe or not probe["properties"]:
            warnings.append(f"Recommended: optical properties missing for probe {probe.get('probe_name')}")
            break  # Only warn once

    if not desc.get("fret_pairs"):
        warnings.append("Recommended: no FRET pairs defined")

    return warnings


def suggest_pdbx_keys(prefix: str = "") -> list[tuple[str, str]]:
    """Return (key, description) pairs from PDBx dictionary matching prefix.

    Parameters
    ----------
    prefix : str, optional
        Filter keys by this prefix.

    Returns
    -------
    list of tuple
        (key, description) pairs.
    """
    from chisurf.core.mfdb.pdbx_metadata import suggest_pdbx_keys as _suggest_pdbx_keys
    return _suggest_pdbx_keys(prefix)
