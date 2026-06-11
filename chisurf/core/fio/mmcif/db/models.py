from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


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
