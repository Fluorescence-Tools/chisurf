"""
Maps lightpath simulator graph state to mmCIF/flrCIF structures.

mmCIF category mapping:
  InstrumentConfig   → _flr_instrument
  LaserLine          → _flr_inst_setting.excitation_wavelength
  DetectorSetting    → _flr_inst_setting.detector_*
  FilterSetting      → _flr_inst_setting.filter_*
  FluorophoreSetting → _flr_probe_list  (partial)
  InstrumentSetting  → _flr_inst_setting (composite)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from .simulator import NodeState

@dataclass
class InstrumentConfig:
    """→ _flr_instrument"""
    id: str = "instrument_1"
    details: str = "ChiSurf Lightpath Simulator"

@dataclass
class LaserLine:
    """→ _flr_inst_setting (excitation source)"""
    wavelength_nm: float
    power_relative: float = 1.0

@dataclass
class FilterSetting:
    """→ _flr_inst_setting (filter/dichroic/splitter)"""
    name: str
    spectrum_db_id: Optional[int]   # references FluorophoreDatabase item
    role: str   # 'excitation_filter' | 'emission_filter' | 'dichroic' | 'splitter'

@dataclass
class DetectorSetting:
    """→ _flr_inst_setting (detector channel)"""
    name: str
    qe_spectrum_db_id: Optional[int]

@dataclass
class FluorophoreSetting:
    """→ _flr_probe_list (partial)"""
    name: str
    db_id: Optional[int]
    quantum_yield: float
    extinction_coefficient: float

@dataclass
class InstrumentSetting:
    """→ _flr_inst_setting (top-level composite)"""
    id: str
    instrument: InstrumentConfig
    lasers: List[LaserLine]         = field(default_factory=list)
    filters: List[FilterSetting]    = field(default_factory=list)
    detectors: List[DetectorSetting]= field(default_factory=list)
    fluorophores: List[FluorophoreSetting] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def build_instrument_setting(node_states: Dict[str, NodeState], db) -> InstrumentSetting:
    """
    Parse propagated NodeState dict into an InstrumentSetting.
    """
    lasers: List[LaserLine] = []
    filters: List[FilterSetting] = []
    detectors: List[DetectorSetting] = []
    fluorophores: List[FluorophoreSetting] = []

    for ns in node_states.values():
        if ns.node_type == "light_source":
            mode = ns.config.get("source_mode", "manual")
            if mode == "manual":
                for part in ns.config.get("manual_lines", "").split(","):
                    part = part.strip()
                    if not part: continue
                    pieces = part.split(":")
                    try:
                        wl   = float(pieces[0])
                        pwr  = float(pieces[1]) if len(pieces) > 1 else 1.0
                        lasers.append(LaserLine(wl, pwr))
                    except ValueError:
                        pass
            else:
                sid = ns.config.get("spectrum_id")
                if sid:
                    # In DB mode, we might not have a single wavelength
                    lasers.append(LaserLine(wavelength_nm=0.0, power_relative=1.0))

        elif ns.node_type in ("filter", "splitter"):
            role = "dichroic" if ns.node_type == "splitter" else "filter"
            filters.append(FilterSetting(
                name=ns.config.get("filter_name", ns.node_type),
                spectrum_db_id=ns.config.get("spectrum_id"),
                role=role
            ))

        elif ns.node_type == "detector":
            detectors.append(DetectorSetting(
                name=ns.config.get("detector_name", "Detector"),
                qe_spectrum_db_id=ns.config.get("spectrum_id")
            ))

        elif ns.node_type == "sample":
            dye_props = ns.config.get("dye_properties", {})
            for sid in ns.config.get("spectrum_ids", []):
                with db:
                    props = db.get_standardized_optical_properties(sid)
                    item  = db.get_probe_by_id(sid)
                    name  = item["chromophore_name"] if item else f"Dye_{sid}"
                
                sid_str = str(sid)
                qy = float(dye_props.get(sid_str, {}).get("qy", props.get("qy") or 1.0))
                ec = float(dye_props.get(sid_str, {}).get("ec", props.get("ext_coeff") or 1.0))
                    
                fluorophores.append(FluorophoreSetting(
                    name=name,
                    db_id=sid,
                    quantum_yield=qy,
                    extinction_coefficient=ec
                ))

    return InstrumentSetting(
        id="inst_setting_1",
        instrument=InstrumentConfig(),
        lasers=lasers,
        filters=filters,
        detectors=detectors,
        fluorophores=fluorophores
    )
