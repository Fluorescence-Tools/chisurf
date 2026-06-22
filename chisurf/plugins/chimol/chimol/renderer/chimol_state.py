from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _copy_array(value):
    """Return a defensive copy of arbitrary array-like payloads."""

    if value is None:
        return None
    try:
        return np.array(value, copy=True)
    except Exception:
        try:
            return copy.deepcopy(value)
        except Exception:
            return value


def _coerce_covariance_array(value):
    """Normalize covariance payloads into ``(N, 3, 3)`` arrays."""

    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return arr
    if arr.ndim == 2 and arr.shape[1] == 9:
        return arr.reshape(-1, 3, 3)
    if arr.ndim == 2 and arr.shape[1] == 6:
        out = np.zeros((arr.shape[0], 3, 3), dtype=float)
        out[:, 0, 0] = arr[:, 0]
        out[:, 1, 1] = arr[:, 1]
        out[:, 2, 2] = arr[:, 2]
        out[:, 0, 1] = out[:, 1, 0] = arr[:, 3]
        out[:, 0, 2] = out[:, 2, 0] = arr[:, 4]
        out[:, 1, 2] = out[:, 2, 1] = arr[:, 5]
        return out
    return None


@dataclass
class _MolViewObjectState:
    coords: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    radius: float = 1.0
    atoms: Optional[np.ndarray] = None
    all_atom_coords: Optional[np.ndarray] = None
    all_atom_res_ids: Optional[np.ndarray] = None
    all_atom_radii: Optional[np.ndarray] = None
    atom_features: dict[str, object] = field(default_factory=dict)
    atom_feature_meta: dict[str, dict] = field(default_factory=dict)
    residue_ids: Optional[np.ndarray] = None
    residue_names: Optional[np.ndarray] = None
    residue_oneletter: Optional[np.ndarray] = None
    residue_chain_ids: Optional[np.ndarray] = None
    selected_residues: list[int] = field(default_factory=list)
    color_mode: str = "single"
    colors_per_ca: Optional[np.ndarray] = None
    colors_per_residue_override: Optional[np.ndarray] = None
    colors_per_atom_override: Optional[np.ndarray] = None
    secondary_structure: Optional[np.ndarray] = None
    representation_mode: str = "cartoon"
    trace_ups: Optional[np.ndarray] = None
    show_cartoon: bool = True
    show_trace: bool = False
    show_atoms: bool = False
    show_dots: bool = False
    show_sticks: bool = False
    sidechains_visible: bool = True
    show_atom_gaussians: bool = False
    cartoon_mask: Optional[np.ndarray] = None
    ball_mask: Optional[np.ndarray] = None
    sticks_mask: Optional[np.ndarray] = None
    bond_pairs: Optional[np.ndarray] = None
    surface_visible: bool = False
    metaballs_visible: bool = False
    point_overlays: dict[str, dict] = field(default_factory=dict)
    frames: Optional[np.ndarray] = None
    active_frame: int = 0
    measurements: dict[str, dict] = field(default_factory=dict)
    bead_radii: Optional[np.ndarray] = None
    rmf_hierarchy: Optional[object] = None  # RmfHierarchyNode
    restraints: list[dict] = field(default_factory=list)
    rmf_provenance: list[dict] = field(default_factory=list)


@dataclass
class _MolViewObjectEntry:
    object_id: str
    name: str
    state: _MolViewObjectState = field(default_factory=_MolViewObjectState)
    visible: bool = True
    placeholder: bool = False
    source_path: Optional[str] = None


class _StateField:
    """Descriptor that proxies attribute access to the active object state."""

    def __init__(self, attr_name: str):
        self.attr_name = attr_name

    def __get__(self, instance, owner):  # type: ignore[override]
        if instance is None:
            return self
        state = instance._get_active_state()
        return getattr(state, self.attr_name)

    def __set__(self, instance, value):  # type: ignore[override]
        state = instance._get_active_state()
        setattr(state, self.attr_name, value)


__all__ = [
    "_copy_array",
    "_coerce_covariance_array",
    "_MolViewObjectState",
    "_MolViewObjectEntry",
    "_StateField",
]
