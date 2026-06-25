"""Headless optical path simulator — no Qt dependency."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from chisurf.gui.widgets.node_editor.graph import GraphDef

from .crosstalk import WAVELENGTHS, propagate_node

logger = logging.getLogger(__name__)

@dataclass
class NodeState:
    id: str
    node_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    input_spectra: Dict = field(default_factory=dict)
    output_spectra: Dict = field(default_factory=dict)
    node_char: Optional[Any] = None

class OpticalPathSimulator:
    """
    Runs signal propagation on a plain graph dict (no Qt).

    Usage (headless):
        sim = OpticalPathSimulator(db)
        sim.load_from_dict(graph_dict)   # same format as scene.to_dict()
        results = sim.propagate()
        setting = sim.to_instrument_setting()
    """
    def __init__(self, db):
        self.db = db
        self._states: Dict[str, NodeState] = {}
        self._graph: Optional[GraphDef] = None

    def load_from_dict(self, scene_dict: dict) -> None:
        """Load from a plain graph dict (serialised NodeScene.to_dict())."""
        self._graph = GraphDef.from_dict(scene_dict)
        self._states = {
            n.id: NodeState(id=n.id, node_type=n.node_type, config=n.config.copy())
            for n in self._graph.nodes
        }

    def propagate(self) -> Dict[str, NodeState]:
        """Run multi-pass propagation. Returns dict of NodeState by node_id."""
        if not self._graph:
            return {}

        # Build adjacency for propagation
        nodes_dict = {n.id: n for n in self._graph.nodes}
        out_adj = {n_id: {} for n_id in nodes_dict}
        
        for edge in self._graph.edges:
            src, tgt = edge.source, edge.target
            if src not in nodes_dict or tgt not in nodes_dict: continue
            
            # Port mapping - in the JSON format, port indices are absolute
            # We need to know if it's an output or input port
            src_node = nodes_dict[src]
            tgt_node = nodes_dict[tgt]
            
            # Logic from simulator_widget.py
            # port_abs >= len(inputs) means it is an output
            u_is_out = edge.source_port >= len(src_node.inputs)
            v_is_out = edge.target_port >= len(tgt_node.inputs)
            
            if u_is_out and not v_is_out:
                s, t, s_p_abs, t_p_abs = src, tgt, edge.source_port, edge.target_port
            elif v_is_out and not u_is_out:
                s, t, s_p_abs, t_p_abs = tgt, src, edge.target_port, edge.source_port
            else: continue
            
            s_node = nodes_dict[s]
            t_node = nodes_dict[t]
            
            s_out_idx = s_p_abs - len(s_node.inputs)
            s_p = s_node.outputs[s_out_idx]
            s_p_name = s_p.name if hasattr(s_p, 'name') else str(s_p)
            
            t_p = t_node.inputs[t_p_abs]
            t_p_name = t_p.name if hasattr(t_p, 'name') else str(t_p)
            
            out_adj[s].setdefault(s_p_name, []).append((t, t_p_name))

        # 1. Clear spectral states
        for ns in self._states.values():
            ns.input_spectra = {}
            ns.output_spectra = {}
            ns.node_char = None
            ns.config["_last_signals"] = {}

        # 2. Multi-pass propagation
        to_process = [n_id for n_id, ns in self._states.items() if ns.node_type == "light_source"]
        if not to_process: to_process = list(self._states.keys())
        
        # Limit iterations to prevent infinite loops in cyclic graphs
        max_iters = len(self._states) * 2
        iters = 0
        
        while to_process and iters < max_iters:
            n_id = to_process.pop(0)
            if n_id not in self._states: continue
            ns = self._states[n_id]
            
            out_specs, node_char = propagate_node(ns.node_type, ns.config, ns.input_spectra, self.db)
            ns.node_char = node_char
            ns.output_spectra = out_specs
            
            for p_name, port_dict in out_specs.items():
                for tgt_id, tgt_p_name in out_adj.get(n_id, {}).get(p_name, []):
                    if tgt_id not in self._states: continue
                    tgt_ns = self._states[tgt_id]
                    tgt_inputs = tgt_ns.input_spectra
                    tgt_in = tgt_inputs.setdefault(tgt_p_name, {})
                    
                    for src_id, spec in port_dict.items():
                        if isinstance(spec, np.ndarray):
                            if src_id not in tgt_in or not isinstance(tgt_in[src_id], np.ndarray):
                                tgt_in[src_id] = np.zeros_like(WAVELENGTHS)
                            tgt_in[src_id] += spec
                        else:
                            if isinstance(spec, dict) and isinstance(tgt_in.get(src_id), dict):
                                tgt_in[src_id].update(spec)
                            else:
                                tgt_in[src_id] = spec
                    
                    if tgt_id not in to_process: to_process.append(tgt_id)
            iters += 1
            
        return self._states

    def get_detector_signals(self) -> List[Dict]:
        """Return [{laser, detector, dye, intensity}] rows."""
        row_data = []
        for n_id, ns in self._states.items():
            if ns.node_type == "detector":
                det_name = ns.config.get("detector_name", f"Detector {n_id}")
                signals = ns.config.get("_last_signals", {})
                for src_key, val in signals.items():
                    if val <= 1e-12: continue
                    
                    if " (ex " in src_key:
                        parts = src_key.split(" (ex ")
                        dye = parts[0]
                        laser = parts[1].rstrip(")")
                    else:
                        dye = "None"
                        laser = src_key
                        
                    if " (QY:" in dye:
                        dye = dye.split(" (QY:")[0]
                        
                    row_data.append({
                        "laser": laser,
                        "detector": det_name,
                        "dye": dye,
                        "intensity": val
                    })
        row_data.sort(key=lambda x: (x["laser"], x["detector"], x["dye"]))
        return row_data

    @staticmethod
    def _matrix_from_records(
        records: List[Dict],
        row_key: str,
        column_key: str,
        value_key: str,
        value_label: str,
    ) -> Dict[str, Any]:
        """Build a JSON-safe matrix payload from row records."""
        row_labels = sorted({str(record[row_key]) for record in records})
        column_labels = sorted({str(record[column_key]) for record in records})
        values_by_key = {
            (str(record[row_key]), str(record[column_key])): float(record[value_key])
            for record in records
        }
        values = [
            [
                values_by_key.get((row_label, column_label), 0.0)
                for column_label in column_labels
            ]
            for row_label in row_labels
        ]
        return {
            "rows": row_labels,
            "columns": column_labels,
            "values": values,
            "records": records,
            "value": value_label,
        }

    def get_excitation_rows(self) -> List[Dict]:
        """Return excitation probabilities by laser and dye."""
        by_key: Dict[tuple[str, str], float] = {}
        for ns in self._states.values():
            if ns.node_type != "sample":
                continue
            for row in ns.config.get("_last_excitation", []):
                laser = str(row.get("laser", ""))
                dye = str(row.get("dye", ""))
                if not laser or not dye:
                    continue
                by_key[(laser, dye)] = by_key.get((laser, dye), 0.0) + float(
                    row.get("excitation", 0.0)
                )
        return [
            {"laser": laser, "dye": dye, "excitation": value}
            for (laser, dye), value in sorted(by_key.items())
        ]

    def get_crosstalk_matrices(self) -> Dict[str, Any]:
        """Return excitation, emission, and detected-intensity matrices."""
        excitation_records = self.get_excitation_rows()
        detector_records = self.get_detector_signals()
        excitation_by_key = {
            (row["laser"], row["dye"]): row["excitation"]
            for row in excitation_records
        }

        emission_accumulator: Dict[tuple[str, str], list[float]] = {}
        detected_records = []
        for row in detector_records:
            laser = str(row["laser"])
            dye = str(row["dye"])
            detector = str(row["detector"])
            intensity = float(row["intensity"])
            detected_records.append(
                {
                    "source": f"{laser} | {dye}",
                    "detector": detector,
                    "intensity": intensity,
                }
            )
            excitation = excitation_by_key.get((laser, dye), 0.0)
            if excitation > 0.0:
                emission_accumulator.setdefault((dye, detector), []).append(
                    intensity / excitation
                )

        emission_records = [
            {
                "dye": dye,
                "detector": detector,
                "probability": float(np.mean(values)) if values else 0.0,
            }
            for (dye, detector), values in sorted(emission_accumulator.items())
        ]

        return {
            "excitation": self._matrix_from_records(
                excitation_records,
                "laser",
                "dye",
                "excitation",
                "relative_excitation",
            ),
            "emission": self._matrix_from_records(
                emission_records,
                "dye",
                "detector",
                "probability",
                "relative_detection_per_excitation",
            ),
            "detected": self._matrix_from_records(
                detected_records,
                "source",
                "detector",
                "intensity",
                "detected_intensity",
            ),
        }

    def to_instrument_setting(self) -> Any:
        """Build mmCIF-ready InstrumentSetting from current propagated state."""
        from .mmcif_export import build_instrument_setting
        return build_instrument_setting(self._states, self.db)
