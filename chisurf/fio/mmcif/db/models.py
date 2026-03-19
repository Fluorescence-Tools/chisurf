from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
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
    type: str = 'polymer'
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
    hetero: str = 'n'

@dataclass
class PolyProbePosition:
    """Corresponds to mmCIF 'flr_poly_probe_position'"""
    probe_id: int
    entity_id: str
    residue_number: int
    asym_id: str = 'A'
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "spectrum_type": self.spectrum_type,
            "wavelengths": self.wavelengths.tolist(),
            "intensity_values": self.intensity_values.tolist()
        }
