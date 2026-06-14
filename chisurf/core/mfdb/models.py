from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

# ── Canonical Vocabulary Constants ─────────────────────────────────────

ARTIFACT_KINDS: tuple[str, ...] = (
    "raw_measurement", "processed_data", "analysis_result",
    "fit_result", "parameter_table", "selection_mask",
    "project_snapshot", "archive_manifest", "archive_file",
    "visualization", "external_reference", "chinet_session", "chinet_node",
    # Legacy values (accepted for backward compat)
    "raw_data", "bur", "ptu", "spc", "bh", "fcs",
    "tcspc", "decay", "irf", "pda", "model_curve", "residual",
    "plot_export", "table_export", "project_archive", "external_file",
    "photon_hdf5", "burst_table", "spectra", "hdf5", "zip",
    "json_summary", "mti_summary", "fcs_correlation", "irf_curve",
    "tcspc_decay", "anisotropy_curve", "pda_histogram", "fit_results",
    "derived_product", "gmm_summary", "clustering_labels",
)

DATA_FORMATS: tuple[str, ...] = (
    "ptu", "spc", "bh", "tttr", "photon_hdf5", "bur",
    "hdf5", "zip", "json", "csv", "tsv", "png", "svg",
    "sqlite", "directory", "unknown",
)

OPERATION_TYPES: tuple[str, ...] = (
    "measurement_import", "validation", "burst_selection",
    "filtering", "fcs_correlation", "microtime_histogram",
    "tcspc_fitting", "model_fitting", "ndxplorer_selection",
    "ndxplorer_clustering", "project_snapshot",
    "project_restore", "archive_export",
    # Legacy values
    "import", "burst_filtering", "gmm_fitting", "analysis",
    "fitting", "project_archive", "local_fit", "global_fit",
    "project", "analysis_run",
)

DIRECTIONS: tuple[str, ...] = ("input", "output")

RELATIONSHIP_TYPES: tuple[str, ...] = (
    "included_in", "contains", "derived_from", "supersedes",
    "uses_external_reference", "parameter_depends_on", "parameter_of", "linked_to",
    "project_contains", "grouped_in",
)

STATUS_VALUES: tuple[str, ...] = (
    "pending", "running", "succeeded", "failed", "cancelled",
    "success", "converged",
)

VALIDATION_STATUS_VALUES: tuple[str, ...] = (
    "unvalidated", "valid", "invalid", "warning",
)

STORAGE_MODES: tuple[str, ...] = (
    "local_file", "local_directory", "url", "managed_archive",
    "embedded_json", "embedded_blob",
    "local", "remote", "embedded", "folder",
)

PARAMETER_TYPES: tuple[str, ...] = (
    "free", "fixed", "linked", "shared", "local", "global", "calibrated",
)

LIFECYCLE_STATUSES: tuple[str, ...] = (
    "active", "archived", "superseded", "deleted",
)


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
