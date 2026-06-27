from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import numpy as np

# ── Canonical Vocabulary Constants (sourced from the mmCIF dictionary) ──

def _load_dictionary_vocabulary() -> dict[str, tuple[str, ...]]:
    """Load all enumeration tuples from the bundled dictionary."""
    result: dict[str, tuple[str, ...]] = {}
    _name_map: dict[str, str] = {
        "artifact_kind": "_mfdb_artifact.artifact_kind",
        "data_format": "_mfdb_artifact.data_format",
        "operation_type": "_mfdb_operation.operation_type",
        "direction": "_mfdb_operation_artifact.direction",
        "relationship_type": "_mfdb_edge.relationship_type",
        "status": "_mfdb_operation.status",
        "validation_status": "_mfdb_operation.validation_status",
        "storage_mode": "_mfdb_object.storage_mode",
        "parameter_type": "_mfdb_parameter.parameter_type",
        "lifecycle_status": "_mfdb_branch.lifecycle_status",
    }
    try:
        from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary
        dic = MmcifDictionary.load_bundled()
        for key, full_name in _name_map.items():
            raw = dic.get_enumerations(full_name)
            result[key] = tuple(v for v in raw if v not in {"#", ".", "?"})
    except Exception:
        pass
    return result

_enums = _load_dictionary_vocabulary()

ARTIFACT_KINDS: tuple[str, ...] = _enums.get("artifact_kind", ())
DATA_FORMATS: tuple[str, ...] = _enums.get("data_format", ())
OPERATION_TYPES: tuple[str, ...] = _enums.get("operation_type", ())
DIRECTIONS: tuple[str, ...] = _enums.get("direction", ())
RELATIONSHIP_TYPES: tuple[str, ...] = _enums.get("relationship_type", ())
STATUS_VALUES: tuple[str, ...] = _enums.get("status", ())
VALIDATION_STATUS_VALUES: tuple[str, ...] = _enums.get("validation_status", ())
STORAGE_MODES: tuple[str, ...] = _enums.get("storage_mode", ())
PARAMETER_TYPES: tuple[str, ...] = _enums.get("parameter_type", ())
LIFECYCLE_STATUSES: tuple[str, ...] = _enums.get("lifecycle_status", ())

# ── Sample Vocabulary Constants (PDBx/mmCIF/flrCIF) ──────────────────────────

# These vocabulary constants are loaded from JSON data files to separate
# data from code. The actual data is stored in chisurf/core/mfdb/data/*.json files.

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_vocabulary(filename: str) -> tuple[str, ...]:
    """Load vocabulary from a JSON file in the data directory."""
    filepath = _DATA_DIR / f"{filename}.json"
    if not filepath.exists():
        # Fallback for testing or when data files are not present
        return ()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return ()
        return tuple(data)
    except (json.JSONDecodeError, OSError):
        return ()


# Load vocabulary constants from JSON files
ENTITY_TYPES: tuple[str, ...] = _load_vocabulary("entity_types") or (
    "protein", "dna", "rna", "polymer", "non-polymer", "water",
    "macromolecule", "oligosaccharide", "ligand", "solvent",
)
COMMON_PROBE_NAMES: tuple[str, ...] = _load_vocabulary("probe_names") or (
    # Fallback if JSON file not available
    "Alexa Fluor 488", "Alexa Fluor 532", "Alexa Fluor 594", "Alexa Fluor 647",
    "ATTO 488", "ATTO488", "ATTO 532", "ATTO532", "ATTO 647N", "ATTO647N",
    "Cy3", "Cy3B", "Cy5",
    "FITC", "TAMRA",
)
BUFFER_COMPONENTS: tuple[str, ...] = _load_vocabulary("buffer_components") or (
    "PBS", "Tris", "HEPES", "NaCl", "KCl", "MgCl2",
    "DTT", "EDTA", "Glycerol", "Tween-20",
)
SAMPLE_CONDITION_FIELDS: tuple[str, ...] = _load_vocabulary("sample_condition_fields") or (
    "ph", "temperature_k", "salt_concentration_m", "buffer_description",
)


def _load_default_spectra() -> dict[str, dict[str, Any]]:
    """Load default fluorophore spectra from JSON file.

    Returns a dictionary mapping probe names to their default photophysical
    properties and spectra. Falls back to empty dict if file not available.
    """
    filepath = _DATA_DIR / "default_fluorophore_spectra.json"
    if not filepath.exists():
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, OSError):
        return {}


# Default fluorophore spectra library (loaded from JSON)
DEFAULT_FLUOROPHORE_SPECTRA: dict[str, dict[str, Any]] = _load_default_spectra()


# ── Entity Definition ──────────────────────────────────────────────────────────


@dataclass
class EntityDefinition:
    """One biomolecular entity in the sample.

    A sample can involve multiple entities (e.g. a heterodimer of protein A
    and protein B, or a protein + DNA). Each entity may appear as one or
    more chains (asym_ids) in the entity assembly.

    Maps to the ``entities`` + ``entity_poly_seq`` tables.

    Parameters
    ----------
    name : str
        Entity common name, for example ``"T4 Lysozyme"``.
    entity_type : str
        Entity type, must be one of ``ENTITY_TYPES`` if provided.
        Valid values: protein, dna, rna, polymer, non-polymer, water,
        macromolecule, oligosaccharide, ligand, solvent.
    sequence : str
        Amino-acid or nucleotide sequence of the construct, **as measured**
        (i.e. including engineered mutations such as cysteine substitutions).
    details : str, optional
        Additional entity description.
    uniprot_accession : str, optional
        UniProt accession of the source record, e.g. ``"P00720"``. Maps to a
        ``struct_ref`` row with ``db_name="UNP"``. See PRD-39.
    pdb_id : str, optional
        PDB identifier of the corresponding structure, e.g. ``"2LZM"``. Maps to
        a ``struct_ref`` row with ``db_name="PDB"``.
    pdb_chain_id : str, optional
        Author chain identifier within the PDB entry.
    organism : str, optional
        Source organism (typically populated from UniProt).
    reference_sequence : str, optional
        Canonical / wild-type sequence from the reference DB. The diff between
        ``sequence`` and ``reference_sequence`` yields ``mutations``.
    mutations : list of MutationDefinition, optional
        Engineered mutations / sequence differences vs. ``reference_sequence``.
        Maps to ``struct_ref_seq_dif`` rows.
    """
    name: str = ""
    entity_type: str = ""
    sequence: str = ""
    details: str = ""
    # ── External references (PRD-39) ──
    uniprot_accession: Optional[str] = None
    pdb_id: Optional[str] = None
    pdb_chain_id: Optional[str] = None
    organism: Optional[str] = None
    reference_sequence: Optional[str] = None
    mutations: list["MutationDefinition"] = dataclasses.field(default_factory=list)


# ── Mutation Definition ──────────────────────────────────────────────────────


@dataclass
class MutationDefinition:
    """One difference between the construct and its reference DB sequence.

    Maps 1:1 to a ``struct_ref_seq_dif`` row. The common case in smFRET is a
    cysteine substitution introduced for dye labeling (e.g. ``S48C`` replaces
    SER with CYS at residue 48). See PRD-39.

    Parameters
    ----------
    seq_id : int
        Residue number in the construct sequence numbering.
    mut_comp_id : str
        Construct residue (3-letter code), e.g. ``"CYS"`` (``mon_id``).
    wt_comp_id : str, optional
        Reference (wild-type) residue, e.g. ``"SER"`` (``db_mon_id``).
    auth_name : str, optional
        Author label for the mutation, e.g. ``"S48C"``.
    kind : str, optional
        Difference type, written to ``struct_ref_seq_dif.details``. One of
        ``engineered_mutation`` (default), ``conflict``, ``insertion``,
        ``deletion``, ``variant``.
    rationale : str, optional
        Why the mutation was made, e.g. ``"cysteine labeling"``.
    """
    seq_id: int = 0
    mut_comp_id: str = ""
    wt_comp_id: str = ""
    auth_name: str = ""
    kind: str = "engineered_mutation"
    rationale: str = ""


# ── Probe Definition for photophysical properties ────────────────────────────


@dataclass
class ProbeDefinition:
    """One fluorescent probe attached to the sample.

    A probe is a physical chromophore at a specific position on the
    biomolecule. It has intrinsic photophysical properties (spectra,
    quantum yield, extinction coefficient) and chemical identity
    (SMILES/InChI).

    **No donor/acceptor label here.** Whether a probe acts as donor or
    acceptor is not an intrinsic property — it depends on which other
    probe it is paired with. Cy5 is an acceptor relative to Cy3B but
    a donor relative to Cy7. In homo-FRET, the same dye is both.

    A sample can have any number of probes. The ``spectra`` table stores
    full absorption and emission spectra as arrays. The FRET relationship
    between probes is determined by spectral overlap (emission of one
    overlapping absorption of another), which defines the overlap
    integral J(λ) and thus the Förster radius R₀.

    Parameters
    ----------
    name : str
        Probe/fluorophore name, for example ``"Cy3B"``.
        Must be in ``COMMON_PROBE_NAMES`` vocabulary if known.
    position : int, optional
        Residue or sequence position number.
    position_label : str, optional
        Human-readable position label, for example ``"C48"`` or ``"5-prime"``.
    chain_id : str, optional
        PDBx `_struct_asym.id` value. Default is ``"A"``.
    residue_name : str, optional
        PDBx `_entity_poly_seq.mon_id` value, for example ``"CYS"``.
    absorption_wavelength_nm : float, optional
        Absorption peak wavelength in nanometers.
    emission_wavelength_nm : float, optional
        Emission peak wavelength in nanometers.
    quantum_yield : float, optional
        Fluorescence quantum yield (0–1).
    extinction_coefficient : float, optional
        Molar extinction coefficient in M⁻¹cm⁻¹.
    absorption_spectrum : tuple of (list[float], list[float]), optional
        Absorption spectrum as (wavelengths_nm, intensities).
        Stored in ``spectra`` table.
    emission_spectrum : tuple of (list[float], list[float]), optional
        Emission spectrum as (wavelengths_nm, intensities).
        Stored in ``spectra`` table.
    chromophore_smiles : str, optional
        SMILES for the chromophore.
    chromophore_inchi : str, optional
        InChI for the chromophore.
    reactive_probe_smiles : str, optional
        SMILES for the reactive form (e.g. maleimide).
    reactive_probe_name : str, optional
        Name of reactive form, e.g. "Cy3B-maleimide".
    reactive_probe_flag : str, optional
        "yes" if reactive form differs from chromophore. Default "no".
    probe_origin : str, optional
        "extrinsic" or "intrinsic" (e.g. Trp). Default "extrinsic".
    probe_link_type : str, optional
        How probe attaches to biomolecule. Default "covalent".
    chromophore_center_atom : str, optional
        Atom name for AV simulation center.
    linker_smiles : str, optional
        SMILES for the full probe-linker conjugate.
    ambiguous_stoichiometry : str, optional
        "yes" if labeling stoichiometry is uncertain. Default "no".
    probe_stoichiometry : float, optional
        Average number of probes at this site.

    **Position model (flrCIF flr_poly_probe_position):** A probe position is
    defined by:
    - ``entity_index`` — which entity in ``SampleDefinition.entities`` (index into list)
    - ``seq_id`` — residue number within the entity sequence (mandatory)
    - ``comp_id`` — residue name (3-letter code, e.g. "CYS", "dT") (mandatory)
    - ``asym_id`` — chain/strand within the entity assembly (mandatory for multi-chain)
    - ``atom_id`` — specific attachment atom (e.g. "CB" for Cβ, "C5") (optional)
    - ``mutation_flag`` — "yes"/"no" — residue mutated for labeling? (mandatory)
    - ``modification_flag`` — "yes"/"no" — residue chemically modified? (mandatory)
    - ``auth_name`` — author-provided position name (e.g. "S131C") (optional)

    **Note:** Residue index alone is NOT sufficient — in multi-chain complexes
    (heterodimers, multi-subunit assemblies, protein-DNA complexes),
    the same residue number exists on different chains. The entity + chain + residue
    tuple uniquely identifies the position.

    **Legacy fields:** For backward compatibility, the old fields ``position``,
    ``position_label``, ``chain_id``, and ``residue_name`` are kept and mapped
    to the new fields when the new fields are not provided.

    Database mapping:
        | ProbeDefinition field | DB table | Column(s) |
        |------------------------|----------|-----------|
        | name | `probes` | `chromophore_name` |
        | absorption/emission scalars | `optical_properties` | property_name + value + unit |
        | absorption_spectrum | `spectra` | spectrum_type="absorption", wavelengths, intensity_values |
        | emission_spectrum | `spectra` | spectrum_type="emission", wavelengths, intensity_values |
        | chromophore_smiles/inchi | `ihm_chemical_component_descriptor` | smiles, inchi |
        | reactive_probe_smiles | `ihm_chemical_component_descriptor` | smiles |
        | linker_smiles | `ihm_chemical_component_descriptor` | smiles |
        | probe_origin, probe_link_type | `probes` | direct columns |
        | position | `flr_poly_probe_position` | residue_number (legacy) |
        | chain_id | `flr_poly_probe_position` | asym_id (legacy) |
        | residue_name | `flr_poly_probe_position` | residue_name (legacy) |
        | entity_index, seq_id, comp_id, asym_id, atom_id | `flr_poly_probe_position` | (new fields) |
        | mutation_flag, modification_flag, auth_name | `flr_poly_probe_position` | (new fields) |
    """

    name: str
    # ── Legacy position fields (kept for backward compatibility) ──
    position: Optional[int] = None
    position_label: str = ""
    chain_id: str = "A"
    residue_name: str = ""
    # ── New flrCIF-compatible position fields (PRD-02) ──
    entity_index: int = 0
    seq_id: Optional[int] = None
    comp_id: str = ""
    asym_id: str = ""
    atom_id: str = ""
    mutation_flag: str = "no"
    modification_flag: str = "no"
    auth_name: str = ""
    absorption_wavelength_nm: Optional[float] = None
    emission_wavelength_nm: Optional[float] = None
    quantum_yield: Optional[float] = None
    extinction_coefficient: Optional[float] = None
    absorption_spectrum: Optional[tuple[list[float], list[float]]] = None
    emission_spectrum: Optional[tuple[list[float], list[float]]] = None
    chromophore_smiles: str = ""
    chromophore_inchi: str = ""
    reactive_probe_smiles: str = ""
    reactive_probe_name: str = ""
    reactive_probe_flag: str = "no"
    probe_origin: str = ""
    probe_link_type: str = ""
    chromophore_center_atom: str = ""
    linker_smiles: str = ""
    ambiguous_stoichiometry: str = "no"
    probe_stoichiometry: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate probe physical constraints and auto-populate from defaults."""
        # Backward compatibility: map legacy position fields to new flrCIF fields
        # Only apply if new fields are not explicitly set
        if self.seq_id is None and self.position is not None:
            self.seq_id = self.position
        if not self.comp_id and self.residue_name:
            self.comp_id = self.residue_name
        if not self.asym_id and self.chain_id:
            self.asym_id = self.chain_id
        if not self.asym_id:
            self.asym_id = "A"
        if not self.auth_name and self.position_label:
            self.auth_name = self.position_label

        # Auto-populate from default spectra library if available
        if self.name in DEFAULT_FLUOROPHORE_SPECTRA:
            defaults = DEFAULT_FLUOROPHORE_SPECTRA[self.name]
            # Only set fields that are None - user-provided values take precedence
            if self.absorption_wavelength_nm is None and "absorption_wavelength_nm" in defaults:
                self.absorption_wavelength_nm = defaults["absorption_wavelength_nm"]
            if self.emission_wavelength_nm is None and "emission_wavelength_nm" in defaults:
                self.emission_wavelength_nm = defaults["emission_wavelength_nm"]
            if self.quantum_yield is None and "quantum_yield" in defaults:
                self.quantum_yield = defaults["quantum_yield"]
            if self.extinction_coefficient is None and "extinction_coefficient" in defaults:
                self.extinction_coefficient = defaults["extinction_coefficient"]
            if self.absorption_spectrum is None and "absorption_spectrum" in defaults:
                abs_data = defaults["absorption_spectrum"]
                if abs_data:
                    self.absorption_spectrum = (
                        abs_data["wavelengths_nm"],
                        abs_data["intensities"]
                    )
            if self.emission_spectrum is None and "emission_spectrum" in defaults:
                em_data = defaults["emission_spectrum"]
                if em_data:
                    self.emission_spectrum = (
                        em_data["wavelengths_nm"],
                        em_data["intensities"]
                    )
            if not self.chromophore_smiles and "chromophore_smiles" in defaults:
                self.chromophore_smiles = defaults["chromophore_smiles"]
            if not self.chromophore_inchi and "chromophore_inchi" in defaults:
                self.chromophore_inchi = defaults["chromophore_inchi"]
            if not self.probe_origin and "probe_origin" in defaults:
                self.probe_origin = defaults["probe_origin"]
            if not self.probe_link_type and "probe_link_type" in defaults:
                self.probe_link_type = defaults["probe_link_type"]

        # Apply defaults for fields that should never be empty
        if not self.probe_origin:
            self.probe_origin = "extrinsic"
        if not self.probe_link_type:
            self.probe_link_type = "covalent"

        # Validate probe physical constraints
        if self.quantum_yield is not None:
            if not (0.0 <= self.quantum_yield <= 1.0):
                raise ValueError(f"quantum_yield {self.quantum_yield} must be between 0.0 and 1.0")
        if self.extinction_coefficient is not None:
            if self.extinction_coefficient < 0:
                raise ValueError(f"extinction_coefficient {self.extinction_coefficient} must be >= 0")
        if self.probe_stoichiometry is not None:
            if self.probe_stoichiometry < 0:
                raise ValueError(f"probe_stoichiometry {self.probe_stoichiometry} must be >= 0")


def validate_vocabulary(value: str, valid_values: tuple[str, ...], label: str) -> None:
    """Validate that *value* is in *valid_values*, raising ``ValueError`` otherwise.

    Parameters
    ----------
    value : str
        The value to check.
    valid_values : tuple of str
        The set of accepted values.
    label : str
        Human-readable name for error messages (e.g. ``'artifact_kind'``).

    Raises
    ------
    ValueError
        If *value* is not in *valid_values*.
    """
    if value not in valid_values:
        raise ValueError(
            f"Invalid {label} {value!r}. "
            f"Must be one of {valid_values}"
        )


# ── Legacy Model Classes (unchanged) ──────────────────────────────────

@dataclass
class ProbeType:
    """Corresponds to flr_probe_type_list (simplified)"""

    type_id: int
    type_name: str
    display_name: str


@dataclass
class Probe:
    """Corresponds to flr_probe_list"""

    probe_id: int
    chromophore_name: str
    type_id: int
    category: str = "other"
    description: str = ""
    is_curated: bool = False
    quality_flag: bool = True
    probe_origin: str = "extrinsic"
    probe_link_type: str = "covalent"
    fluorophore_type: str = "unspecified"
    reactive_probe_flag: str = "no"
    reactive_probe_name: Optional[str] = None
    chromophore_chem_descriptor_id: Optional[int] = None
    reactive_probe_chem_descriptor_id: Optional[int] = None
    chromophore_center_atom: Optional[str] = None


@dataclass
class Entity:
    """Corresponds to mmCIF 'entity'"""

    entity_id: str
    type: str = "polymer"
    description: Optional[str] = None
    formula_weight: Optional[float] = None
    src_method: Optional[str] = None
    number_of_molecules: int = 1
    common_name: Optional[str] = None


@dataclass
class SequenceResidue:
    """Corresponds to mmCIF 'entity_poly_seq'"""

    entity_id: str
    num: int
    mon_id: str
    hetero: str = "n"


@dataclass
class PolyProbePosition:
    """Corresponds to mmCIF 'flr_poly_probe_position'"""

    probe_id: int
    entity_id: str
    residue_number: int
    asym_id: str = "A"
    residue_name: Optional[str] = None
    description: Optional[str] = None


@dataclass
class SampleCondition:
    """Corresponds to mmCIF 'flr_sample_condition'"""

    condition_id: str
    ph: Optional[float] = None
    temperature: Optional[float] = None
    ionic_strength: Optional[float] = None
    buffer_composition: Optional[str] = None
    details: Optional[str] = None


@dataclass
class SampleUser:
    """Laboratory user or operator associated with a sample."""

    user_id: str
    display_name: str
    email: Optional[str] = None
    affiliation: Optional[str] = None
    details: Optional[str] = None


@dataclass
class SampleDevice:
    """Measurement device associated with a sample."""

    device_id: str
    name: str
    device_type: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    location: Optional[str] = None
    owner: Optional[str] = None
    details: Optional[str] = None


@dataclass
class ExperimentType:
    """User-definable fluorescence experiment type."""

    type_id: Optional[int]
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    details: Optional[str] = None


@dataclass
class Experiment:
    """Fluorescence experiment linked to a sample and metadata."""

    experiment_id: str
    type_id: Optional[int] = None
    sample_id: Optional[str] = None
    project_id: Optional[str] = None
    measured_by_user_id: Optional[str] = None
    measured_by_device_id: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    status: Optional[str] = None
    details: Optional[str] = None


@dataclass
class ExperimentData:
    """Embedded or linked raw data for an experiment."""

    data_id: Optional[int]
    experiment_id: str
    data_type: str
    storage_mode: str
    file_path: Optional[str] = None
    url: Optional[str] = None
    folder_path: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    data_json: Optional[str] = None
    data_blob: Optional[bytes] = None
    details: Optional[str] = None


@dataclass
class SampleProbe:
    """Explicit mapping between a sample and a probe."""

    sample_probe_id: Optional[int]
    sample_id: str
    probe_id: int
    poly_probe_position_id: Optional[int] = None
    fluorophore_type: str = "unspecified"
    description: Optional[str] = None


@dataclass
class EntityAssembly:
    """Corresponds to mmCIF 'flr_entity_assembly'."""

    assembly_id: str
    description: Optional[str] = None
    details: Optional[str] = None


@dataclass
class OpticalProperty:
    """Generic optical property (absorption peak, QY, etc.)"""

    probe_id: int
    property_name: str
    property_value: str
    unit: Optional[str] = None


@dataclass
class Spectrum:
    """Spectral data container"""

    probe_id: int
    spectrum_type: str
    wavelengths: np.ndarray
    intensity_values: np.ndarray
    wavelength_unit: str = "nm"
    intensity_unit: str = "normalized"
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the spectrum to a dictionary.

        Returns
        -------
        dict
            Dictionary with probe_id, spectrum_type, wavelengths, and
            intensity_values as native Python types.
        """
        return {
            "probe_id": self.probe_id,
            "spectrum_type": self.spectrum_type,
            "wavelengths": self.wavelengths.tolist(),
            "intensity_values": self.intensity_values.tolist(),
            "wavelength_unit": self.wavelength_unit,
            "intensity_unit": self.intensity_unit,
            "details": self.details,
        }


@dataclass
class ExternalFile:
    """External file reference for large analysis data."""

    file_id: int
    reference_id: Optional[str] = None
    file_path: Optional[str] = None
    file_format: Optional[str] = None
    content_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    md5: Optional[str] = None
    uuid: Optional[str] = None
    details: Optional[str] = None


@dataclass
class PhotonStream:
    """Photon-stream metadata pointing to an external file."""

    stream_id: str
    analysis_id: Optional[str] = None
    external_file_id: Optional[int] = None
    detector_id: Optional[str] = None
    description: Optional[str] = None
    details: Optional[str] = None


@dataclass
class AnalysisMetadata:
    """User-provided analysis metadata."""

    analysis_id: str
    key: str
    value: str
    details: Optional[str] = None


# ── New MFDB Model Classes ────────────────────────────────────────────

@dataclass
class MfdbSample:
    """Sample identity and broad biological/chemical context (``flr_sample``)."""

    sample_id: str
    display_name: str
    sample_type: str
    metadata_json: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None


@dataclass
class FretPairDefinition:
    """One FRET pair between two probes on the sample.

    The pair defines which probe acts as donor (energy transfer source)
    and which as acceptor (energy transfer sink) **for this specific
    pair**. The same probe can appear as donor in one pair and acceptor
    in another (e.g. relay dye in 3-color FRET).

    The Förster radius R₀ is determined by:
        R₀⁶ = (9 ln10 κ² QD J(λ)) / (128 π⁵ n⁴ Nₐ)
    where J(λ) is the spectral overlap integral between the donor
    emission and acceptor absorption spectra. If full spectra are
    provided on the probes, J(λ) and R₀ can be computed automatically.

    For 3-color FRET (Cy3B → Cy5 → Cy7), create two pairs:
        FretPairDefinition(probe_1_index=0, probe_2_index=1, ...)  # Cy3B→Cy5
        FretPairDefinition(probe_1_index=1, probe_2_index=2, ...)  # Cy5→Cy7
    where indices refer to positions in SampleDefinition.probes.
    probe_1 is the energy transfer source (donor role in this pair).
    probe_2 is the energy transfer sink (acceptor role in this pair).

    Parameters
    ----------
    probe_1_index : int
        Index into SampleDefinition.probes (donor role).
    probe_2_index : int
        Index into SampleDefinition.probes (acceptor role).
    forster_radius_nm : float, optional
        Förster radius in nanometers. Can be computed from spectra.
    reduced_forster_radius_nm : float, optional
        Reduced Förster radius (R₀ adjusted for experimental conditions).
    kappa_squared : float, optional
        Orientation factor κ². Default is 2/3 ≈ 0.666667 (dynamic averaging).
    refractive_index : float, optional
        Refractive index of the medium. Default is 1.4 (water).
    overlap_integral : float, optional
        J(λ) spectral overlap integral, computed from spectra if available.
    """

    probe_1_index: int
    probe_2_index: int
    forster_radius_nm: Optional[float] = None
    reduced_forster_radius_nm: Optional[float] = None
    kappa_squared: float = 2.0 / 3.0
    refractive_index: float = 1.4
    overlap_integral: Optional[float] = None


@dataclass
class SampleDefinition:
    """Complete sample definition for creating or updating a sample.

    This dataclass represents a full flrCIF-compatible sample description,
    including entity, probes with photophysical properties, positions,
    conditions, and Förster radius.

    Parameters
    ----------
    name : str
        Human-readable sample name (required).
    description : str, optional
        Free-text sample description.
    entity_name : str, optional
        Entity common name, for example ``"T4 lysozyme"``.
    entity_sequence : str, optional
        Amino-acid or nucleotide sequence.
    entity_type : str, optional
        Entity type, must be one of ``ENTITY_TYPES`` if provided.
        Valid values: protein, dna, rna, polymer, non-polymer, water,
        macromolecule, oligosaccharide, ligand, solvent.
    probes : list of ProbeDefinition, optional
        List of all probes on the sample. Replaces flat donor/acceptor fields.
        Supports 1, 2, 3, 4+ probes for various FRET configurations.
    fret_pairs : list of FretPairDefinition, optional
        List of FRET pairs defining energy transfer relationships between probes.
        Supports 0, 1, 2+ FRET pairs.
    entities : list of EntityDefinition, optional
        List of biomolecular entities in the sample. Supports multi-entity
        samples (e.g., protein-DNA complexes, heterodimers). Each entity may
        appear as one or more chains (asym_ids) in the entity assembly.
        For backward compatibility: if entity_name is provided and entities
        is empty, auto-creates entities = [EntityDefinition(name=entity_name, ...)].
    buffer_description : str, optional
        Buffer description, for example ``"PBS pH 7.4"``.
    ph : float, optional
        Sample pH. Must be 0–14 if provided. None = unset, 0.0 = valid acidic.
    temperature_k : float, optional
        Temperature in Kelvin. Must be > 0 if provided.
    salt_concentration_m : float, optional
        Salt concentration in mol/L.
    solvent_phase : str, optional
        Physical phase of the solvent.
    extra : dict, optional
        Additional metadata stored as JSON.
    validate_vocabulary : bool, optional
        If ``True`` (default ``False``), validate entity_type and probe names
        against known vocabularies. Set to ``True`` for strict validation.

    Notes
    -----
    Legacy flat fields (donor_probe_name, donor_position, etc.) are kept for
    backward compatibility. When both legacy and new list fields are
    provided, the new list fields take precedence.

    For pH=0.0 (strongly acidic, valid) vs pH=None (unset): pH=0.0 is stored
    as 0.0, pH=None is stored as NULL in the database.

    The ``fluorophore_type`` on ``flr_sample_probe`` is derived from FRET pair
    context, not set directly on the probe. A probe that appears as probe_1
    (donor role) in any pair gets ``fluorophore_type="donor"``, and probe_2 gets
    ``"acceptor"``. A probe appearing in both roles (relay dye) gets
    ``"unspecified"``. A probe in no pairs gets ``"unspecified"``.

    Database mapping:
        Each ``ProbeDefinition`` → one row in ``flr_sample_probe``.
        Each ``FretPairDefinition`` → one row in ``flr_fret_forster_radius``.
    """

    name: str
    description: str = ""
    # New structured fields for multi-entity support (PRD-02)
    entities: list[EntityDefinition] = dataclasses.field(default_factory=list)
    # Legacy flat entity fields - kept for backward compatibility
    entity_name: str = ""
    entity_sequence: str = ""
    entity_type: str = ""
    # New structured fields (replacing flat donor/acceptor)
    probes: list[ProbeDefinition] = dataclasses.field(default_factory=list)
    fret_pairs: list[FretPairDefinition] = dataclasses.field(default_factory=list)
    # Legacy flat fields - kept for backward compatibility
    donor_probe_name: str = ""
    donor_position: Optional[int] = None
    donor_position_label: str = ""
    acceptor_probe_name: str = ""
    acceptor_position: Optional[int] = None
    acceptor_position_label: str = ""
    # Legacy structured fields (for backward compat, deprecated)
    donor: Optional[ProbeDefinition] = None
    acceptor: Optional[ProbeDefinition] = None
    buffer_description: str = ""
    ph: Optional[float] = None
    temperature_k: Optional[float] = None
    salt_concentration_m: Optional[float] = None
    solvent_phase: Optional[str] = None
    extra: dict = dataclasses.field(default_factory=dict)
    validate_vocabulary: bool = False

    def __post_init__(self) -> None:
        """Validate vocabulary fields and physical constraints if validation is enabled."""
        if not self.name or not self.name.strip():
            raise ValueError("sample name is required")

        # Backward compatibility: auto-create entities from legacy flat fields
        # if entities list is empty (PRD-02 Task 1.2)
        if not self.entities and self.entity_name:
            self.entities = [
                EntityDefinition(
                    name=self.entity_name,
                    entity_type=self.entity_type,
                    sequence=self.entity_sequence,
                )
            ]

        # Auto-create default entity for new-format samples (R13-1)
        # If probes are provided but entities is empty, create a single default entity
        # to avoid validation failure on entity_index=0
        if not self.entities and self.probes:
            self.entities = [
                EntityDefinition(
                    name=self.name or "sample_entity",
                    entity_type="",
                    sequence="",
                )
            ]

        # Validate physical constraints regardless of vocabulary validation
        self._validate_physical_constraints()

        # Validate probe indices in FRET pairs
        self._validate_fret_pairs()

        # Validate entity_index references for each probe (PRD-02 Task 6)
        self._validate_entity_indices()

        if self.validate_vocabulary:
            self._validate_vocabulary()

    def _validate_physical_constraints(self) -> None:
        """Validate physical constraints on numeric fields."""
        if self.ph is not None:
            if not (0.0 <= self.ph <= 14.0):
                raise ValueError(f"pH {self.ph} must be between 0.0 and 14.0")

        if self.temperature_k is not None:
            if self.temperature_k <= 0:
                raise ValueError(f"temperature_k {self.temperature_k} must be > 0")

        # Note: ProbeDefinition.__post_init__ is called automatically by dataclass
        # and validates quantum_yield and extinction_coefficient for each probe

    def _validate_fret_pairs(self) -> None:
        """Validate that FRET pair indices are valid.

        Per PRD-02: Validation should only count self.probes for FRET pair
        index validation. Legacy donor/acceptor fields are deprecated and
        should not be conflated with the new probes list.
        """
        num_probes = len(self.probes)

        for i, pair in enumerate(self.fret_pairs):
            if pair.probe_1_index < 0 or pair.probe_1_index >= num_probes:
                raise ValueError(
                    f"FRET pair {i}: probe_1_index {pair.probe_1_index} "
                    f"out of range (0-{num_probes-1})"
                )
            if pair.probe_2_index < 0 or pair.probe_2_index >= num_probes:
                raise ValueError(
                    f"FRET pair {i}: probe_2_index {pair.probe_2_index} "
                    f"out of range (0-{num_probes-1})"
                )
            if pair.kappa_squared is not None:
                if not (0.0 <= pair.kappa_squared <= 4.0):
                    raise ValueError(
                        f"FRET pair {i}: kappa_squared {pair.kappa_squared} "
                        f"must be between 0.0 and 4.0"
                    )

    def _validate_entity_indices(self) -> None:
        """Validate entity_index references for each probe (PRD-02 Task 6).

        Ensures that each probe's entity_index points to a valid entity
        in the entities list.
        """
        num_entities = len(self.entities)

        for i, probe in enumerate(self.probes):
            if probe.entity_index < 0 or probe.entity_index >= num_entities:
                raise ValueError(
                    f"probe[{i}].entity_index={probe.entity_index} "
                    f"exceeds entities list length {num_entities}"
                )

    def _validate_vocabulary(self) -> None:
        """Validate entity_type and probe names against known vocabularies.

        Uses both the static vocabulary constants (ENTITY_TYPES, COMMON_PROBE_NAMES)
        and the MmcifDictionary for PDBx/pdbihm/flrCIF validation.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Validate entity_type against ENTITY_TYPES vocabulary
        if self.entity_type:
            validate_vocabulary(
                self.entity_type,
                ENTITY_TYPES,
                "entity_type"
            )

            # Also validate against flrCIF/PDBx dictionary if available
            try:
                from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary
                dic = MmcifDictionary.load_bundled()
                # Validate against _flr_entity.type or _entity.type
                for category in ["flr_entity", "entity"]:
                    err = dic.validate_value(f"_{category}.type", self.entity_type)
                    if err is None:
                        # Validation passed
                        break
                    # If we get here, try the next category
                else:
                    # None of the categories validated successfully
                    logger.warning(
                        f"entity_type '{self.entity_type}' not found in PDBx/flrCIF "
                        f"dictionary. Consider using a standard value."
                    )
            except Exception:
                # Dictionary not available, skip this validation
                pass

        # Validate probe names from new probes list
        # Per PRD-02: unknown probe names should WARN (not reject), since custom
        # dyes are valid. Only entity_type should hard-reject.
        for i, probe in enumerate(self.probes):
            if probe.name:
                if probe.name not in COMMON_PROBE_NAMES:
                    logger.warning(
                        f"probe[{i}].name '{probe.name}' not in COMMON_PROBE_NAMES. "
                        f"Consider using a standard probe name."
                    )

                # Validate against flrCIF dictionary if available
                try:
                    from chisurf.core.mfdb.pdbx_metadata import MmcifDictionary
                    dic = MmcifDictionary.load_bundled()
                    # Try to validate against flr_poly_probe.chromophore_name
                    err = dic.validate_value(
                        "_flr_poly_probe.chromophore_name",
                        probe.name
                    )
                    if err:
                        logger.debug(
                            f"probe[{i}].name '{probe.name}' validation: {err}"
                        )
                except Exception:
                    pass

        # Validate probe names from legacy structured fields
        # Per PRD-02: unknown probe names should WARN only (custom dyes are valid)
        if self.donor is not None and self.donor.name:
            if self.donor.name not in COMMON_PROBE_NAMES:
                logger.warning(
                    f"donor.name '{self.donor.name}' not in COMMON_PROBE_NAMES. "
                    f"Consider using a standard probe name."
                )

        if self.acceptor is not None and self.acceptor.name:
            if self.acceptor.name not in COMMON_PROBE_NAMES:
                logger.warning(
                    f"acceptor.name '{self.acceptor.name}' not in COMMON_PROBE_NAMES. "
                    f"Consider using a standard probe name."
                )

        # Also validate legacy flat fields if provided
        if self.donor_probe_name and self.donor_probe_name not in COMMON_PROBE_NAMES:
            logger.warning(
                f"donor_probe_name '{self.donor_probe_name}' not in COMMON_PROBE_NAMES. "
                f"Consider using a standard probe name."
            )

        if self.acceptor_probe_name and self.acceptor_probe_name not in COMMON_PROBE_NAMES:
            logger.warning(
                f"acceptor_probe_name '{self.acceptor_probe_name}' not in COMMON_PROBE_NAMES. "
                f"Consider using a standard probe name."
            )

    def get_probe_names(self) -> list[str]:
        """Return all probe names from probes list, legacy structured fields, and flat fields."""
        names = []
        for probe in self.probes:
            names.append(probe.name)
        if self.donor is not None and self.donor.name:
            names.append(self.donor.name)
        if self.acceptor is not None and self.acceptor.name:
            names.append(self.acceptor.name)
        if self.donor_probe_name and self.donor_probe_name not in names:
            names.append(self.donor_probe_name)
        if self.acceptor_probe_name and self.acceptor_probe_name not in names:
            names.append(self.acceptor_probe_name)
        return names

    def get_all_probes(self) -> list[ProbeDefinition]:
        """Return all probes including those from legacy fields."""
        probes = list(self.probes)
        if self.donor is not None:
            probes.append(self.donor)
        if self.acceptor is not None:
            probes.append(self.acceptor)
        return probes

    def get_donor_probe_name(self) -> str:
        """Return donor probe name from either ProbeDefinition or legacy field.

        Deprecated: Use probes list and fret_pairs instead.
        """
        if self.donor is not None:
            return self.donor.name
        return self.donor_probe_name

    def get_acceptor_probe_name(self) -> str:
        """Return acceptor probe name from either ProbeDefinition or legacy field.

        Deprecated: Use probes list and fret_pairs instead.
        """
        if self.acceptor is not None:
            return self.acceptor.name
        return self.acceptor_probe_name

    def get_donor_position(self) -> Optional[int]:
        """Return donor position from either ProbeDefinition or legacy field.

        Deprecated: Use probes list instead.
        """
        if self.donor is not None:
            return self.donor.position
        return self.donor_position

    def get_acceptor_position(self) -> Optional[int]:
        """Return acceptor position from either ProbeDefinition or legacy field.

        Deprecated: Use probes list instead.
        """
        if self.acceptor is not None:
            return self.acceptor.position
        return self.acceptor_position


def compute_forster_radius(
    donor_emission: tuple[np.ndarray, np.ndarray],
    acceptor_absorption: tuple[np.ndarray, np.ndarray],
    donor_quantum_yield: float,
    kappa_squared: float = 2.0 / 3.0,
    refractive_index: float = 1.4,
) -> tuple[float, float]:
    """Compute Förster radius R₀ (nm) and spectral overlap integral J(λ) from spectra.

    The Förster radius is computed using the formula:
        R₀⁶ = (9 ln10 κ² QD J(λ)) / (128 π⁵ n⁴ Nₐ)

    where:
        - κ² is the orientation factor (kappa_squared)
        - QD is the donor quantum yield
        - J(λ) is the spectral overlap integral
        - n is the refractive index of the medium
        - Nₐ is Avogadro's number

    Parameters
    ----------
    donor_emission : tuple of (np.ndarray, np.ndarray)
        Donor emission spectrum as (wavelengths_nm, intensities).
        Intensities should be normalized or in consistent units.
    acceptor_absorption : tuple of (np.ndarray, np.ndarray)
        Acceptor absorption spectrum as (wavelengths_nm, intensities).
        Intensities should be in molar extinction coefficient units (M⁻¹cm⁻¹).
    donor_quantum_yield : float
        Donor fluorescence quantum yield (0–1).
    kappa_squared : float, optional
        Orientation factor κ². Default is 2/3 ≈ 0.666667 (dynamic averaging).
    refractive_index : float, optional
        Refractive index of the medium. Default is 1.4 (water).

    Returns
    -------
    tuple of (float, float)
        (forster_radius_nm, overlap_integral)

    Notes
    -----
    The spectral overlap integral J(λ) is computed as:
        J(λ) = ∫ F_donor(λ) ε_acceptor(λ) λ⁴ dλ / ∫ F_donor(λ) dλ

    where F_donor is the donor emission spectrum and ε_acceptor is the
    acceptor molar extinction coefficient spectrum.
    """
    import scipy.integrate

    # Constants
    N_A = 6.02214076e23  # Avogadro's number (mol⁻¹)
    ln_10 = np.log(10.0)

    # Unpack spectra
    donor_wl, donor_int = donor_emission
    acceptor_wl, acceptor_ext = acceptor_absorption

    # Convert to numpy arrays if they aren't already
    donor_wl = np.asarray(donor_wl, dtype=float)
    donor_int = np.asarray(donor_int, dtype=float)
    acceptor_wl = np.asarray(acceptor_wl, dtype=float)
    acceptor_ext = np.asarray(acceptor_ext, dtype=float)

    # Find overlapping wavelength range
    min_wl = max(np.min(donor_wl), np.min(acceptor_wl))
    max_wl = min(np.max(donor_wl), np.max(acceptor_wl))

    if min_wl >= max_wl:
        # No spectral overlap
        return 0.0, 0.0

    # Interpolate both spectra onto a common wavelength grid
    # Use a fine grid for accurate integration
    common_wl = np.linspace(min_wl, max_wl, 1000)

    # Interpolate donor emission
    from scipy.interpolate import interp1d
    donor_interp = interp1d(donor_wl, donor_int, kind='linear', bounds_error=False, fill_value=0.0)
    donor_int_common = donor_interp(common_wl)

    # Interpolate acceptor absorption
    acceptor_interp = interp1d(acceptor_wl, acceptor_ext, kind='linear', bounds_error=False, fill_value=0.0)
    acceptor_ext_common = acceptor_interp(common_wl)

    # Compute spectral overlap integral J(λ)
    # J(λ) = ∫ F_donor(λ) ε_acceptor(λ) λ⁻⁴ dλ / ∫ F_donor(λ) dλ
    # Note: The standard formula uses λ⁴ in numerator, but there are different conventions
    # We use: J = ∫ F_donor(λ) ε_acceptor(λ) λ⁴ dλ / ∫ F_donor(λ) dλ
    numerator_integral = scipy.integrate.trapezoid(
        donor_int_common * acceptor_ext_common * common_wl**4, common_wl
    )
    denominator_integral = scipy.integrate.trapezoid(donor_int_common, common_wl)

    if denominator_integral <= 0:
        return 0.0, 0.0

    overlap_integral = numerator_integral / denominator_integral

    # Compute Förster radius R₀ in nm
    # R₀⁶ = (9 ln10 κ² QD J) / (128 π⁵ n⁴ Nₐ) * 1e-51  (conversion factors for nm)
    # Simplified constant for units where λ is in nm and ε in M⁻¹cm⁻¹:
    # R₀⁶ (in nm⁶) = 8.785e-5 * κ² * QD * J / n⁴
    constant = 8.785e-5  # nm⁶ * M * cm⁻¹
    r6 = constant * kappa_squared * donor_quantum_yield * overlap_integral / (refractive_index**4)

    if r6 <= 0:
        return 0.0, float(overlap_integral)

    forster_radius_nm = r6 ** (1.0 / 6.0)

    return float(forster_radius_nm), float(overlap_integral)


@dataclass
class MfdbExperiment:
    """Groups measurements, analyses, and archived projects (``flr_experiment``)."""

    experiment_id: str
    display_name: str
    sample_id: Optional[str] = None
    project_id: Optional[str] = None
    status: str = "pending"
    metadata_json: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None


@dataclass
class MfdbVocabulary:
    """Extensible vocabulary entry (``mfdb_vocabulary``)."""

    field_name: str
    value: str
    display_name: str
    description: Optional[str] = None
    is_builtin: bool = True
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None


# ── Canonical MFDB Models ─────────────────────────────────────────────

@dataclass
class MfdbObject:
    """Content-addressed object in the deduplicated object store (``mfdb_object``)."""

    object_uuid: str
    content_md5: str
    storage_path: str
    original_filename: Optional[str] = None
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    refcount: int = 1
    metadata_json: Optional[str] = None
    created_at: Optional[str] = None
    created_by_user_uuid: Optional[str] = None


@dataclass
class MfdbArtifact:
    """A data object, file reference, folder reference, object snapshot,
    exported archive, figure, table, fit result, or manifest (``mfdb_artifact``)."""

    artifact_id: str
    artifact_kind: str
    storage_mode: str = "local_file"
    data_format: Optional[str] = None
    experiment_id: Optional[str] = None
    file_path: Optional[str] = None
    url: Optional[str] = None
    folder_path: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    checksum_algorithm: str = "sha256"
    row_count: Optional[int] = None
    validation_status: str = "unvalidated"
    validation_message: Optional[str] = None
    metadata_json: Optional[str] = None
    data_json: Optional[str] = None
    data_blob: Optional[bytes] = None
    object_uuid: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None


@dataclass
class MfdbOperation:
    """A measurement, import, processing, analysis, fitting, archive, or
    restore action (``mfdb_operation``)."""

    operation_id: str
    operation_type: str
    setup_id: Optional[str] = None
    setup_version: Optional[int] = None
    experiment_id: Optional[str] = None
    settings_json: Optional[str] = None
    settings_hash: Optional[str] = None
    operator_user_id: Optional[str] = None
    software_package: Optional[str] = None
    software_module: Optional[str] = None
    software_version: Optional[str] = None
    runtime_environment_json: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    status: str = "pending"
    error_message: Optional[str] = None
    traceback_summary: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None


@dataclass
class MfdbOperationArtifact:
    """Link table between operations and artifacts (inputs & outputs).
    Role is NOT NULL in the database (``mfdb_operation_artifact``)."""

    operation_id: str
    artifact_id: str
    direction: str  # 'input' or 'output'
    role: str = "generic"
    ordinal: int = 0
    checksum_snapshot: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None


@dataclass
class MfdbEdge:
    """General provenance relationship edge (``mfdb_edge``).
    Never stores ``input_to`` or ``produced`` — those belong in
    ``MfdbOperationArtifact``."""

    edge_id: Optional[int]
    source_node_type: str
    source_node_id: str
    target_node_type: str
    target_node_id: str
    relationship_type: str
    operation_id: Optional[str] = None
    settings_hash: Optional[str] = None
    timestamp: Optional[str] = None
    software_version: Optional[str] = None
    checksum_snapshot_json: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None


@dataclass
class MfdbParameter:
    """Semantically important fit or analysis parameter (``mfdb_parameter``)."""

    parameter_id: Optional[int]
    parameter_uuid: str
    operation_id: str
    name: str
    value: Optional[float] = None
    standard_error: Optional[float] = None
    confidence_interval_low: Optional[float] = None
    confidence_interval_high: Optional[float] = None
    initial_value: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    bounds_on: int = 0
    units: Optional[str] = None
    parameter_type: str = "free"
    expression: Optional[str] = None
    prior_json: Optional[str] = None
    mapping_json: Optional[str] = None
    metadata_json: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None


@dataclass
class MfdbSetup:
    """Instrument and configuration setup version snapshot (``mfdb_setup``)."""

    setup_id: str
    name: str
    version: int = 1
    instrument_id: Optional[str] = None
    description: Optional[str] = None
    configuration_json: Optional[str] = None
    detectors_json: Optional[str] = None
    timing_calibration_json: Optional[str] = None
    irf_definition_json: Optional[str] = None
    dark_count_json: Optional[str] = None
    timing_resolution_json: Optional[str] = None
    burst_defaults_json: Optional[str] = None
    fcs_calibration_json: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None


@dataclass
class MfdbAuditLog:
    """Change log entry for operations on database (``mfdb_audit_log``)."""

    log_id: Optional[int]
    action: str
    target_type: str
    target_id: str
    timestamp: Optional[str] = None
    operator_user_id: Optional[str] = None
    details_json: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    deleted_at: Optional[str] = None


# ── Backward-Compatibility Aliases ────────────────────────────────────

FdbArtifact = MfdbArtifact
FdbOperation = MfdbOperation
FdbOperationArtifact = MfdbOperationArtifact
FdbEdge = MfdbEdge
FdbParameter = MfdbParameter
FdbSetup = MfdbSetup
FdbAuditLog = MfdbAuditLog

# Old constant names
ARTIFACT_TYPES = ARTIFACT_KINDS
