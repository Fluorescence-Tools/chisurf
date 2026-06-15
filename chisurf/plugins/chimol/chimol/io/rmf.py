from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, List, Dict, Any, Tuple
import numpy as np
import copy

try:
    import RMF
except ImportError:
    RMF = None

class RmfNotAvailableError(RuntimeError):
    pass

@dataclass
class RmfHierarchyNode:
    name: str
    rmf_index: int
    node_type: str  # e.g., 'STATE', 'CHAIN', 'RESIDUE', 'PARTICLE'
    children: List["RmfHierarchyNode"] = field(default_factory=list)
    parent: Optional["RmfHierarchyNode"] = field(default=None, repr=False)
    parent_index: Optional[int] = None
    
    # Structural metadata
    chain_id: Optional[str] = None
    res_num: Optional[int] = None
    res_type: Optional[str] = None
    copy_index: Optional[int] = None
    radius: Optional[float] = None
    
    # Indices in the flat coordinate array (for the viewer)
    atom_indices: List[int] = field(default_factory=list)

class _RmfHierarchyInfo:
    """Track structural information encountered through the RMF hierarchy."""
    def __init__(self):
        self.chain_id = None
        self.copy_index = None
        self.res_num = None
        self.res_type = None
        
    def handle_node(self, node: RMF.NodeConstHandle, loader: _RmfLoader) -> _RmfHierarchyInfo:
        rhi = self
        
        # Chain
        if loader.chainf.get_is(node):
            rhi = copy.copy(rhi)
            c = loader.chainf.get(node)
            rhi.chain_id = c.get_chain_id()
            
        # Copy
        if loader.copyf.get_is(node):
            rhi = copy.copy(rhi)
            rhi.copy_index = loader.copyf.get(node).get_copy_index()
            
        # Fragment
        if loader.fragmentf.get_is(node):
            rhi = copy.copy(rhi)
            f = loader.fragmentf.get(node)
            resinds = f.get_residue_indexes()
            if resinds:
                rhi.res_num = resinds[len(resinds) // 2]
            rhi.res_type = 'UNK'
            
        # Residue
        if loader.residuef.get_is(node):
            rhi = copy.copy(rhi)
            r = loader.residuef.get(node)
            rhi.res_num = r.get_residue_index()
            rhi.res_type = r.get_residue_type()
            
        return rhi

class _RmfLoader:
    def __init__(self):
        self.particle_nodes: List[RMF.NodeConstHandle] = []
        self.rmf_index_to_particle_idx: Dict[int, int] = {}
        
    def load(self, path: Path) -> Dict[str, Any]:
        if RMF is None:
            raise RmfNotAvailableError("RMF loading requires 'RMF' package.")
            
        r = RMF.open_rmf_file_read_only(str(path))
        
        # Initialize factories
        self.particlef = RMF.ParticleConstFactory(r)
        self.chainf = RMF.ChainConstFactory(r)
        self.fragmentf = RMF.FragmentConstFactory(r)
        self.residuef = RMF.ResidueConstFactory(r)
        self.copyf = RMF.CopyConstFactory(r)
        self.statef = RMF.StateConstFactory(r)
        self.bondf = RMF.BondConstFactory(r)
        self.represf = RMF.RepresentationConstFactory(r)
        self.atomf = RMF.AtomConstFactory(r)
        self.segmentf = RMF.SegmentConstFactory(r)
        self.coloredf = RMF.ColoredConstFactory(r)
        
        try:
            self.softwaref = RMF.SoftwareProvenanceConstFactory(r)
        except AttributeError:
            try:
                self.softwaref = RMF.SoftwareConstFactory(r)
            except AttributeError:
                self.softwaref = None

        r.set_current_frame(RMF.FrameID(0))
        
        # 1. First pass: Collect all particle nodes and build hierarchy
        rhi = _RmfHierarchyInfo()
        root_node = self._handle_node(r.get_root_node(), rhi)
        
        # 2. Extract coordinates for all frames
        num_frames = r.get_number_of_frames()
        num_particles = len(self.particle_nodes)
        
        if num_particles == 0:
            raise RuntimeError(f"No particles/coordinates found in {path}")
            
        rmf_frame_series: Dict[str, list[float]] = {}
        rmf_frame_metadata: Dict[str, list[Any]] = {}
        stat_keys = self._extract_stat_keys(r)
        for _key, name in stat_keys:
            rmf_frame_series[name] = []
            rmf_frame_metadata[name] = []
            
        frames_arr = np.zeros((num_frames, num_particles, 3), dtype=np.float32)
        coord_buffer = np.zeros((num_particles, 3), dtype=np.float64)
        
        for f in range(num_frames):
            r.set_current_frame(RMF.FrameID(f))
            try:
                RMF.get_all_global_coordinates(r, r.get_root_node(), coord_buffer)
            except Exception:
                for i, node in enumerate(self.particle_nodes):
                    coord_buffer[i] = self.particlef.get(node).get_coordinates()
            frames_arr[f] = coord_buffer.astype(np.float32)
            self._extract_stat_values(
                r.get_root_node(),
                stat_keys,
                rmf_frame_series,
                rmf_frame_metadata,
            )
            
        # 3. Extract radii
        radii_arr = np.zeros(num_particles, dtype=np.float32)
        for i, node in enumerate(self.particle_nodes):
            radii_arr[i] = self.particlef.get(node).get_radius()
            
        # 4. Extract metadata (restraints, states, PROVENANCE, BONDS, stat)
        restraints = []
        rmf_provenance = []
        states = []
        bond_pairs = []
        
        self._extract_metadata(r.get_root_node(), restraints, rmf_provenance, states, bond_pairs)
        
        return {
            "hierarchy": root_node,
            "frames": frames_arr,
            "radii": radii_arr,
            "states": states,
            "restraints": restraints,
            "rmf_provenance": rmf_provenance,
            "bond_pairs": np.array(bond_pairs, dtype=np.int32) if bond_pairs else None,
            "rmf_frame_series": {
                name: np.asarray(values, dtype=float)
                for name, values in rmf_frame_series.items()
            },
            "rmf_frame_metadata": rmf_frame_metadata,
        }
        
    def _handle_node(self, node: RMF.NodeConstHandle, parent_rhi: _RmfHierarchyInfo) -> RmfHierarchyNode:
        rhi = parent_rhi.handle_node(node, self)
        
        ntype = "NODE"
        if self.statef.get_is(node): ntype = "STATE"
        elif self.chainf.get_is(node): ntype = "CHAIN"
        elif self.residuef.get_is(node): ntype = "RESIDUE"
        elif self.atomf.get_is(node): ntype = "ATOM"
        elif self.particlef.get_is(node): ntype = "PARTICLE"
        
        ridx = node.get_id().get_index()
        h_node = RmfHierarchyNode(
            name=node.get_name(),
            rmf_index=ridx,
            node_type=ntype,
            chain_id=rhi.chain_id,
            res_num=rhi.res_num,
            res_type=rhi.res_type,
            copy_index=rhi.copy_index
        )
        
        # We only collect particles that have Coordinate trait
        # ParticleConstFactory usually checks for mass/radius, but we also need XYZ.
        # Actually in RMF all Particles should have coords.
        if self.particlef.get_is(node):
            p_idx = len(self.particle_nodes)
            self.particle_nodes.append(node)
            self.rmf_index_to_particle_idx[ridx] = p_idx
            h_node.atom_indices = [p_idx]
            h_node.radius = self.particlef.get(node).get_radius()
            
        for child in node.get_children():
            # Skip Representation and Provenance nodes in the main hierarchy tree
            if self.represf.get_is(child): continue
            if child.get_type() == RMF.PROVENANCE: continue
            
            child_h = self._handle_node(child, rhi)
            child_h.parent = h_node
            child_h.parent_index = ridx
            h_node.children.append(child_h)
            h_node.atom_indices.extend(child_h.atom_indices)
            
        return h_node

    def _extract_stat_keys(self, rmf_handle: Any) -> list[tuple[Any, str]]:
        """Return RMF stat keys that can be read as frame metadata."""
        try:
            category = rmf_handle.get_category("stat")
            keys = rmf_handle.get_keys(category)
            return [(key, rmf_handle.get_name(key)) for key in keys]
        except Exception:
            return []

    def _extract_stat_values(
        self,
        root_node: RMF.NodeConstHandle,
        stat_keys: list[tuple[Any, str]],
        series: Dict[str, list[float]],
        metadata: Dict[str, list[Any]],
    ) -> None:
        """Append current-frame RMF stat values to series containers."""
        for key, name in stat_keys:
            try:
                value = root_node.get_value(key)
                numeric = _as_numeric_stat_value(value)
                metadata[name].append(value)
                series[name].append(numeric)
            except Exception:
                metadata[name].append(None)
                series[name].append(float("nan"))

    def _extract_metadata(self, node: RMF.NodeConstHandle, restraints: list, provenance: list, states: list, bond_pairs: list):
        # Software
        if self.softwaref and self.softwaref.get_is(node):
            sw = self.softwaref.get(node)
            get_type = getattr(sw, "get_type", None)
            type_text = ""
            if get_type is not None:
                try:
                    type_text = f" ({get_type()})"
                except Exception:
                    type_text = ""
            provenance.append({"name": sw.get_name(), "value": f"{sw.get_version()}{type_text}"})
            
        # Explicit Bonds
        if self.bondf.get_is(node):
            b = self.bondf.get(node)
            nodes = [b.get_bonded_0(), b.get_bonded_1()]
            indices = []
            for n in nodes:
                idx = n.get_id().get_index()
                if idx in self.rmf_index_to_particle_idx:
                    indices.append(self.rmf_index_to_particle_idx[idx])
            if len(indices) == 2:
                bond_pairs.append((indices[0], indices[1]))

        # Restraints (Representation nodes)
        if self.represf.get_is(node):
            rep = self.represf.get(node)
            targets = rep.get_representation() # Returns handles
            indices = [self.rmf_index_to_particle_idx[t.get_id().get_index()] 
                       for t in targets if t.get_id().get_index() in self.rmf_index_to_particle_idx]
            
            # CRITICAL: Only add as distance restraints if exactly 2 particles.
            # Otherwise it's just a grouping representation and we should NOT draw it as spaghetti.
            if len(indices) == 2:
                restraints.append({"indices": (indices[0], indices[1]), "name": node.get_name()})
        
        # States
        if self.statef.get_is(node):
            states.append(node.get_id().get_index())
            
        for child in node.get_children():
            self._extract_metadata(child, restraints, provenance, states, bond_pairs)

def _as_numeric_stat_value(value: Any) -> float:
    """Convert RMF stat values to floats for Chimol plotting."""
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
    else:
        try:
            numeric = float(value)
        except Exception:
            return float("nan")
    if not np.isfinite(numeric):
        return float("nan")
    return numeric


def load_rmf_full(path: Path) -> Dict[str, Any]:
    loader = _RmfLoader()
    return loader.load(path)

def load_rmf_frames(path: Path, frame_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    data = load_rmf_full(path)
    frames = data["frames"]
    if frame_indices is not None:
        return frames[frame_indices]
    return frames

__all__ = ["load_rmf_frames", "load_rmf_full", "RmfNotAvailableError", "RmfHierarchyNode"]
