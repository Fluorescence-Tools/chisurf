from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module
import copy
from typing import Optional

import numpy as np

from qtpy import QtCore, QtWidgets, QtGui

from ..config import _DISPLAY_CONFIG
from ..geometry import (
    _compute_center_radius,
    _estimate_ambient_occlusion,
    _build_sphere_mesh,
    _build_trace_ups,
    _extract_ca_trace,
    _build_bond_pairs,
    _generate_cartoon_tube_arrays,
    _generate_trace_arrays,
)
from ..analysis.ss import assign_ss_c3_from_atoms
from .base import Renderer
from .qtgl import QtGLRenderer
from .scene import Scene, SceneObject, Geometry
from .chimol_state import _MolViewObjectState, _MolViewObjectEntry, _StateField
from ..colors import (
    _three_to_one_array,
    _build_residue_color_array,
    _build_ss_color_array,
    _build_sequence_gradient_colors,
    _build_element_color_array,
    _build_chain_color_array,
)


def _get_picking_module():
    try:
        return import_module("..app.picking", __package__)
    except Exception:
        return None


def _copy_array(value):
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


def _coerce_rotation_translation(rotation, translation):
    rot = np.asarray(rotation, dtype=float)
    if rot.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3)")
    trans = np.asarray(translation, dtype=float).reshape(3)
    return rot, trans


def _apply_rigid_transform(data, rotation, translation):
    if data is None:
        return None

    arr = np.asarray(data)
    # Skip structured/record arrays (e.g. atoms with an 'xyz' field). Those
    # store coordinates in a dedicated field and are not transformed here.
    if getattr(arr.dtype, "fields", None):
        return data

    arr = arr.astype(float, copy=False)

    if arr.ndim == 2 and arr.shape[1] == 3:
        return (arr @ rotation.T) + translation

    if arr.ndim >= 3 and arr.shape[-1] == 3:
        flat = arr.reshape(-1, 3)
        flat = (flat @ rotation.T) + translation
        return flat.reshape(arr.shape)

    if arr.ndim == 1 and arr.shape[0] == 3:
        return (arr @ rotation.T) + translation

    return arr




class MolView(QtWidgets.QWidget):

    # Emitted when residues are selected via picking in the 3D view. The
    # payload is a list of integer residue indices along the CA trace.
    residueSelectionChanged = QtCore.Signal(object)
    objectResidueSelectionChanged = QtCore.Signal(object, object)
    """Minimal 3D protein viewer widget (Chimol).

    This widget embeds a :class:`QtWidgets.QOpenGLWidget`-based renderer and
    draws a simple backbone trace (through CA atoms where possible). It is
    designed to be embedded in existing Qt layouts and does not manage its own
    QApplication.

    Public methods
    --------------
    - :meth:`set_structure(structure)`: accept a ChiSurf ``Structure``-like
      object with ``atoms``/``xyz`` attributes.
    - :meth:`set_coordinates(xyz)`: accept an ``(N, 3)`` coordinate array.
    """

    _coords = _StateField("coords")
    _center = _StateField("center")
    _radius = _StateField("radius")
    _atoms = _StateField("atoms")
    _all_atom_coords = _StateField("all_atom_coords")
    _all_atom_res_ids = _StateField("all_atom_res_ids")
    _all_atom_radii = _StateField("all_atom_radii")
    _atom_features = _StateField("atom_features")
    _atom_feature_meta = _StateField("atom_feature_meta")
    _residue_ids = _StateField("residue_ids")
    _residue_names = _StateField("residue_names")
    _residue_oneletter = _StateField("residue_oneletter")
    _residue_chain_ids = _StateField("residue_chain_ids")
    _selected_residues = _StateField("selected_residues")
    _color_mode = _StateField("color_mode")
    _colors_per_ca = _StateField("colors_per_ca")
    _colors_per_residue_override = _StateField("colors_per_residue_override")
    _colors_per_atom_override = _StateField("colors_per_atom_override")
    _secondary_structure = _StateField("secondary_structure")
    _representation_mode = _StateField("representation_mode")
    _trace_ups = _StateField("trace_ups")
    _show_cartoon = _StateField("show_cartoon")
    _show_trace = _StateField("show_trace")
    _show_atoms = _StateField("show_atoms")
    _show_dots = _StateField("show_dots")
    _show_sticks = _StateField("show_sticks")
    _sidechains_visible = _StateField("sidechains_visible")
    _show_atom_gaussians = _StateField("show_atom_gaussians")
    _cartoon_mask = _StateField("cartoon_mask")
    _ball_mask = _StateField("ball_mask")
    _sticks_mask = _StateField("sticks_mask")
    _bond_pairs = _StateField("bond_pairs")
    _surface_visible = _StateField("surface_visible")
    _point_overlays = _StateField("point_overlays")
    _measurements = _StateField("measurements")
    _bead_radii = _StateField("bead_radii")
    _rmf_hierarchy = _StateField("rmf_hierarchy")
    _restraints = _StateField("restraints")
    _rmf_provenance = _StateField("rmf_provenance")

    def set_rmf_data(
        self,
        hierarchy: object,
        frames: np.ndarray,
        radii: np.ndarray,
        restraints: list[dict] = None,
        rmf_provenance: list[dict] = None,
        bond_pairs: Optional[np.ndarray] = None,
        *,
        object_id: Optional[str] = None
    ) -> None:
        """Load full RMF data (hierarchy, trajectory, radii) into an object."""
        with self._activate_object(object_id):
            state = self._get_active_state()
            state.rmf_hierarchy = hierarchy
            state.frames = frames
            state.bead_radii = radii
            if restraints:
                state.restraints = restraints
            if rmf_provenance:
                state.rmf_provenance = rmf_provenance
            if bond_pairs is not None:
                state.bond_pairs = bond_pairs
                state.show_sticks = True
            # If we have frames, set the first one as active
            if frames is not None and len(frames) > 0:
                state.all_atom_coords = frames[0]
                state.coords = frames[0]
                state.active_frame = 0
                self._total_frames = max(self._total_frames, len(frames))
            
            # If we have radii, we likely want to show beads (mode 'spheres')
            if radii is not None and np.any(radii > 0):
                state.show_atoms = True  # We use the atoms/spheres path for beads
                state.show_cartoon = False
                state.show_trace = False
                
        self._update_view()

    def _prune_placeholders(self) -> None:
        """Drop any placeholder-only entries."""
        removed_any = False
        for oid in list(self._objects.keys()):
            entry = self._objects.get(oid)
            if entry is not None and entry.placeholder:
                self._objects.pop(oid, None)
                removed_any = True
        if removed_any and self._active_object_id not in self._objects:
            self._active_object_id = next(iter(self._objects), None)
        if not self._objects:
            self._auto_create_enabled = False

    def _create_object(
        self,
        name: Optional[str] = None,
        source_path: Optional[str] = None,
        *,
        placeholder: bool = False,
    ) -> _MolViewObjectEntry:
        if not placeholder:
            self._prune_placeholders()
        self._object_counter += 1
        object_id = f"obj{self._object_counter}"
        entry = _MolViewObjectEntry(
            object_id=object_id,
            name=name or f"Object {self._object_counter}",
            source_path=source_path,
            placeholder=placeholder,
        )
        defaults = _DISPLAY_CONFIG.get("defaults", {})
        try:
            entry.state.color_mode = str(defaults.get("color_mode", "single"))
        except Exception:
            entry.state.color_mode = "single"
        self._objects[object_id] = entry
        self._active_object_id = object_id
        self._auto_create_enabled = True
        return entry

    def _ensure_active_entry(self, create_if_missing: bool = True) -> Optional[_MolViewObjectEntry]:
        if self._active_object_id in self._objects:
            return self._objects[self._active_object_id]
        if self._objects:
            # Prefer a non-placeholder entry
            for oid, entry in self._objects.items():
                if not entry.placeholder:
                    self._active_object_id = oid
                    return entry
            # Fallback to first placeholder if that's all we have
            oid, entry = next(iter(self._objects.items()))
            self._active_object_id = oid
            return entry
        if create_if_missing and self._auto_create_enabled:
            return self._create_object(placeholder=True)
        return None

    def _get_active_state(self) -> _MolViewObjectState:
        entry = self._ensure_active_entry()
        if entry is None:
            raise RuntimeError("No active object available")
        return entry.state

    def get_active_object_id(self) -> Optional[str]:
        return self._active_object_id

    def set_active_object(self, object_id: str) -> bool:
        if object_id not in self._objects:
            return False
        if self._active_object_id == object_id:
            return True
        self._active_object_id = object_id
        self._update_view()
        return True

    def clear_color_overrides(self) -> None:
        self._colors_per_residue_override = None
        self._colors_per_atom_override = None
        if self._coords is not None:
            self._update_view()

    def set_atom_features(
        self,
        features: Optional[dict[str, object]],
        *,
        meta: Optional[dict[str, dict]] = None,
        object_id: Optional[str] = None,
    ) -> None:
        """Attach arbitrary per-atom feature payloads to the active object."""

        def _sanitize_dict(data: Optional[dict[str, object]]) -> dict[str, object]:
            if not data:
                return {}
            cleaned: dict[str, object] = {}
            for key, value in data.items():
                cleaned[str(key)] = _copy_array(value)
            return cleaned

        with self._activate_object(object_id):
            state = self._get_active_state()
            state.atom_features = _sanitize_dict(features)
            if meta is None:
                state.atom_feature_meta = {}
            else:
                cleaned_meta: dict[str, dict] = {}
                for key, info in meta.items():
                    if not isinstance(info, dict):
                        continue
                    cleaned_meta[str(key)] = {
                        str(sub_key): _copy_array(sub_val)
                        for sub_key, sub_val in info.items()
                    }
                state.atom_feature_meta = cleaned_meta

        if self._coords is not None:
            self._update_view()

    def clear_atom_features(self, *, object_id: Optional[str] = None) -> None:
        self.set_atom_features(None, object_id=object_id)

    def set_atom_colors(self, colors: Optional[np.ndarray]) -> None:
        if colors is None:
            self._colors_per_atom_override = None
        else:
            arr = np.asarray(colors, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                return
            if self._all_atom_coords is None:
                return
            n_atoms = int(np.asarray(self._all_atom_coords).shape[0])
            if arr.shape[0] != n_atoms:
                return
            if arr.shape[1] == 3:
                alpha = np.ones((arr.shape[0], 1), dtype=float)
                arr = np.concatenate([arr, alpha], axis=1)
            self._colors_per_atom_override = arr
        if self._coords is not None:
            self._update_view()

    def set_residue_colors(self, colors: Optional[np.ndarray]) -> None:
        if colors is None:
            self._colors_per_residue_override = None
        else:
            arr = np.asarray(colors, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                return
            if self._residue_ids is None:
                return
            n_res = int(np.asarray(self._residue_ids).shape[0])
            if arr.shape[0] != n_res:
                return
            if arr.shape[1] == 3:
                alpha = np.ones((arr.shape[0], 1), dtype=float)
                arr = np.concatenate([arr, alpha], axis=1)
            self._colors_per_residue_override = arr
        if self._coords is not None:
            self._update_view()

    def set_object_visible(self, object_id: str, visible: bool) -> None:
        entry = self._objects.get(object_id)
        if entry is None:
            return
        entry.visible = bool(visible)
        self._update_view()

    def get_residue_positions(self, indices, *, object_id: Optional[str] = None) -> np.ndarray:
        with self._activate_object(object_id):
            coords = self._coords
            if coords is None:
                return np.zeros((0, 3), dtype=float)
            arr = np.asarray(coords, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 3:
                return np.zeros((0, 3), dtype=float)
            if indices is None:
                idx = np.arange(arr.shape[0], dtype=int)
            else:
                try:
                    idx = np.asarray(list(indices), dtype=int)
                except Exception:
                    idx = np.zeros(0, dtype=int)
            if idx.size:
                idx = idx[(idx >= 0) & (idx < arr.shape[0])]
            if idx.size == 0:
                return np.zeros((0, 3), dtype=float)
            return arr[idx].copy()

    def apply_transform_to_object(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
        *,
        object_id: Optional[str] = None,
    ) -> None:
        rot, trans = _coerce_rotation_translation(rotation, translation)
        target_id = object_id if object_id is not None else self._active_object_id

        with self._activate_object(object_id):
            state = self._get_active_state()
            state.coords = _apply_rigid_transform(state.coords, rot, trans)
            state.center = _apply_rigid_transform(state.center, rot, trans)
            state.atoms = _apply_rigid_transform(state.atoms, rot, trans)
            state.all_atom_coords = _apply_rigid_transform(state.all_atom_coords, rot, trans)
            state.frames = _apply_rigid_transform(state.frames, rot, trans)

            if state.coords is not None:
                center, radius = _compute_center_radius(state.coords)
                state.center = center
                state.radius = float(radius)

            # Keep global mirrors in sync when transforming the active object.
            if target_id == self._active_object_id:
                self._coords = state.coords
                self._center = state.center
                self._radius = state.radius

        self._update_view()

    def remove_object(self, object_id: str) -> bool:
        if object_id not in self._objects:
            return False
        self._objects.pop(object_id, None)
        if self._active_object_id == object_id:
            self._active_object_id = None
        if self._objects:
            # Promote the first remaining non-placeholder object to active.
            for oid, entry in self._objects.items():
                if not entry.placeholder:
                    self._active_object_id = oid
                    break
            else:
                self._active_object_id = next(iter(self._objects))
        else:
            # Disable auto-creation when all objects are gone to avoid ghost entries.
            self._auto_create_enabled = False
        # If only placeholders remain, drop them.
        if self._objects and all(entry.placeholder for entry in self._objects.values()):
            self._objects.clear()
            self._active_object_id = None
            self._auto_create_enabled = False
        self._update_view()
        return True

    # ------------------------------------------------------------------
    # Animation API
    # ------------------------------------------------------------------
    def get_total_frames(self) -> int:
        return self._total_frames

    def set_total_frames(self, count: int) -> None:
        self._total_frames = max(1, int(count))
        if self._current_frame >= self._total_frames:
            self._current_frame = self._total_frames - 1
        self._update_view()

    def get_current_frame(self) -> int:
        return self._current_frame

    def set_current_frame(self, frame_idx: int) -> None:
        new_idx = max(0, min(int(frame_idx), self._total_frames - 1))
        if new_idx == self._current_frame:
            return
        self._current_frame = new_idx
        self._apply_frame_states()
        self._update_view(fit_camera=False)

    def _apply_frame_states(self) -> None:
        """Update scene objects based on the current frame index."""
        # For objects with 'frames' coordinate sets, update their active_frame
        for entry in self._objects.values():
            state = entry.state
            if state.frames is not None and state.frames.ndim == 3:
                # If the object has frames, map the global timeline to its states.
                # Simplest mapping: state_idx = global_idx % n_states
                n_states = state.frames.shape[0]
                state.active_frame = self._current_frame % n_states
                state.all_atom_coords = state.frames[state.active_frame]
                state.coords = state.all_atom_coords
                # Also update center/radius if needed, but maybe defer for performance?
                # PyMOL usually doesn't re-center automatically during movie playback.

    def list_objects(self) -> list[dict]:
        objects: list[dict] = []
        for entry in self._objects.values():
            if entry.placeholder:
                continue
            state = entry.state
            has_geometry = state.coords is not None and state.coords.size > 0 if state.coords is not None else False
            objects.append(
                {
                    "id": entry.object_id,
                    "name": entry.name,
                    "visible": entry.visible,
                    "source_path": entry.source_path,
                    "has_geometry": bool(has_geometry),
                }
            )
        return objects

    # ------------------------------------------------------------------
    # Chain utilities
    # ------------------------------------------------------------------
    def get_chain_ids(self, object_id: Optional[str] = None) -> list[str]:
        """Return sorted chain identifiers for the given object (or active)."""
        with self._activate_object(object_id):
            state = self._get_active_state()
            chains = state.residue_chain_ids
            if chains is None:
                return []
            try:
                arr = np.asarray(chains)
                uniq = np.unique(arr)
                uniq = uniq[uniq != ""]
                return [str(x) for x in uniq]
            except Exception:
                return []

    def split_chains(self, *, prefix: Optional[str] = None, object_ids: Optional[list[str]] = None) -> int:
        """Create a new object for each chain in the specified objects.

        Returns the number of new objects created. Original objects are
        hidden (disabled) after splitting, similar to PyMOL.
        """
        target_ids = object_ids or list(self._objects.keys())
        created = 0

        for obj_id in target_ids:
            entry = self._objects.get(obj_id)
            if entry is None:
                continue
            atoms = entry.state.atoms
            if atoms is None or getattr(atoms.dtype, "fields", None) is None:
                continue
            fields = set(atoms.dtype.fields or {})
            if "chain" not in fields:
                continue

            try:
                chains = np.char.strip(atoms["chain"].astype(str))
            except Exception:
                chains = np.array([str(c).strip() for c in atoms["chain"]])

            unique_chains = np.unique(chains)
            for chain_id in unique_chains:
                chain_str = str(chain_id)
                mask = chains == chain_id
                if not mask.any():
                    continue
                sub_atoms = atoms[mask].copy()
                # Build a simple structure-like container
                class _Struct:
                    pass
                struct = _Struct()
                struct.atoms = sub_atoms

                created += 1
                if prefix:
                    name = f"{prefix}{created:04d}"
                else:
                    base = entry.name or obj_id
                    name = f"{base}_{chain_str or ''}"

                self.add_structure(struct, name=name, source_path=entry.source_path)

            # Hide original object after splitting
            entry.visible = False

        if created > 0:
            self._update_view()
        return created

    def add_structure(
        self,
        structure: object,
        *,
        name: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> str:
        entry = self._create_object(name=name, source_path=source_path)
        self.set_structure(structure)
        return entry.object_id

    # ------------------------------------------------------------------
    # Info overlay API
    # ------------------------------------------------------------------

    def set_system_info_visible(self, visible: bool) -> None:
        self._info_visible = bool(visible)
        if self._info_overlay is not None:
            self._info_overlay.setVisible(self._info_visible)

    def set_system_info_text(self, text: str) -> None:
        self._info_text = str(text)
        if self._info_overlay is not None:
            self._info_overlay.setPlainText(self._info_text)
            self._info_overlay.setVisible(self._info_visible)

    def add_coordinates(
        self,
        coords: np.ndarray,
        *,
        name: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> str:
        entry = self._create_object(name=name, source_path=source_path)
        self.set_coordinates(coords)
        return entry.object_id

    @contextmanager
    def _activate_object(self, object_id: Optional[str]):
        prev = self._active_object_id
        if object_id is not None and object_id in self._objects:
            self._active_object_id = object_id
        self._ensure_active_entry(create_if_missing=False)
        try:
            yield
        finally:
            self._active_object_id = prev

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        background: Optional[str] = None,
        *,
        representation_mode: Optional[str] = None,
        show_cartoon: Optional[bool] = None,
        show_trace: Optional[bool] = None,
        show_atoms: Optional[bool] = None,
        show_sticks: Optional[bool] = None,
        sidechains_visible: Optional[bool] = None,
        grid_visible: Optional[bool] = None,
        surface_visible: Optional[bool] = None,
        scale_factor: Optional[float] = None,
    ) -> None:
        super().__init__(parent)

        self._objects: OrderedDict[str, _MolViewObjectEntry] = OrderedDict()
        self._active_object_id: Optional[str] = None
        self._object_counter: int = 0
        # Allow creating an initial entry during startup; turned off when last object is deleted.
        self._auto_create_enabled: bool = True
        
        # Animation / Timeline state
        self._total_frames: int = 1
        self._current_frame: int = 0  # 0-indexed internally
        self._keyframes: dict[int, dict] = {}
        self._animation_running: bool = False
        self._animation_timer: Optional[QtCore.QTimer] = None
        self.selection_mode: str = "Residues"

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.view: Optional[QtWidgets.QWidget] = None
        self._disabled_label: Optional[QtWidgets.QLabel] = None

        # Stored geometry
        self._coords: Optional[np.ndarray] = None
        self._center: Optional[np.ndarray] = None
        self._radius: float = 1.0

        self._atoms = None
        self._all_atom_coords = None
        self._all_atom_res_ids = None
        self._all_atom_radii = None
        self._atom_features = {}
        self._atom_feature_meta = {}

        # Residue / sequence metadata (valid when a Structure is set)
        self._residue_ids = None
        self._residue_names = None
        self._residue_oneletter = None
        self._residue_chain_ids = None
        self._selected_residues = []

        display_defaults = _DISPLAY_CONFIG.get("defaults", {})
        scaling_cfg = _DISPLAY_CONFIG.get("scaling", {})
        camera_cfg = _DISPLAY_CONFIG.get("camera", {})

        # Coloring state
        # "single", "by_residue" (AA type), "by_secondary_structure" (H/E/C)
        # or "by_sequence" (gradient along the CA index)
        self._color_mode = str(
            display_defaults.get("color_mode", "single")
        )
        colors_cfg = _DISPLAY_CONFIG.get("colors", {})
        base_col = np.asarray(colors_cfg.get("base", [0.8, 0.8, 1.0, 1.0]), dtype=float)
        if base_col.shape[0] != 4:
            base_col = np.array([0.8, 0.8, 1.0, 1.0], dtype=float)
        self._base_color_single = tuple(float(x) for x in base_col)
        self._colors_per_ca = None
        self._colors_per_residue_override = None
        self._colors_per_atom_override = None
        self._secondary_structure = None

        rep_mode_default = str(display_defaults.get("representation_mode", "cartoon")).lower()
        self._representation_mode = str(
            (representation_mode or rep_mode_default)
        ).lower()
        self._trace_ups = None

        # Global representation flags to allow independent toggling from the
        # plugin UI (cartoon / trace / atoms / sticks).
        self._show_cartoon = bool(
            show_cartoon if show_cartoon is not None else display_defaults.get("show_cartoon", True)
        )
        self._show_trace = bool(
            show_trace if show_trace is not None else display_defaults.get("show_trace", False)
        )
        self._show_atoms = bool(
            show_atoms if show_atoms is not None else display_defaults.get("show_atoms", False)
        )
        self._show_dots = bool(display_defaults.get("show_dots", False))
        self._show_sticks = bool(
            show_sticks if show_sticks is not None else display_defaults.get("show_sticks", False)
        )

        # Side-chain visibility flag for atom / ball view.
        self._sidechains_visible = bool(
            sidechains_visible if sidechains_visible is not None else display_defaults.get("sidechains_visible", True)
        )

        # Per-residue representation masks (per CA index)
        self._cartoon_mask = None
        self._ball_mask = None

        # Cached bond list for sticks representation: array of shape (M, 2)
        # with integer indices into the all-atom coordinate array.
        self._bond_pairs = None

        self._info_text: str = ""
        self._info_visible: bool = False
        self._info_overlay: Optional[QtWidgets.QPlainTextEdit] = None

        # Render backend
        self._renderer: Optional[Renderer] = None
        self._point_overlays = {}

        # Reference plane (grid) is hidden by default; the toolbar button
        # can toggle it on when needed.
        self._grid_visible: bool = bool(
            grid_visible if grid_visible is not None else display_defaults.get("grid_visible", False)
        )

        self._surface_visible: bool = bool(
            surface_visible if surface_visible is not None else display_defaults.get("surface_visible", False)
        )

        self._scale_factor = float(
            scale_factor if scale_factor is not None else scaling_cfg.get("structure", 10.0)
        )

        def _camera_val(key: str, default: float) -> float:
            try:
                return float(camera_cfg.get(key, default))
            except Exception:
                return float(default)

        self._camera_min_near_clip = max(_camera_val("min_near_clip", 0.005), 1e-4)
        self._camera_max_near_clip = max(
            _camera_val("max_near_clip", 5.0), self._camera_min_near_clip * 1.01
        )
        self._camera_near_clip = min(
            max(_camera_val("near_clip", 0.1), self._camera_min_near_clip), self._camera_max_near_clip
        )
        far_default = _camera_val("far_clip", 1000.0)
        self._camera_far_clip = far_default if far_default > self._camera_near_clip else self._camera_near_clip * 200.0
        clip_wheel_scale = _camera_val("clip_wheel_scale", 0.85)
        if not (0.0 < clip_wheel_scale < 1.0):
            clip_wheel_scale = 0.85
        self._camera_clip_wheel_scale = clip_wheel_scale

        # Camera defaults
        self._default_elevation = 20
        self._default_azimuth = 45
        self._scene: Optional[Scene] = None

        info_cfg = _DISPLAY_CONFIG.get("info_overlay", {})

        try:
            renderer = QtGLRenderer(controller=self, parent=self)
        except Exception:
            renderer = None

        if renderer is not None:
            container = QtWidgets.QWidget(self)
            container_layout = QtWidgets.QGridLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(0)
            container_layout.setRowStretch(0, 1)
            container_layout.setColumnStretch(0, 1)

            self._info_overlay = QtWidgets.QPlainTextEdit(container)
            self._info_overlay.setReadOnly(True)
            max_width = int(info_cfg.get("max_width", 260))
            min_width = int(info_cfg.get("min_width", 180))
            self._info_overlay.setMaximumWidth(max_width)
            self._info_overlay.setMinimumWidth(min_width)
            self._info_overlay.setSizePolicy(
                QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.MinimumExpanding
            )
            full_height = bool(info_cfg.get("full_height", True))
            self._info_overlay.setPlainText(self._info_text or "(no system loaded)")
            style = info_cfg.get(
                "stylesheet",
                "background-color: rgba(0, 0, 0, 180);"
                "color: white;"
                "border: 1px solid rgba(255, 255, 255, 80);",
            )
            self._info_overlay.setStyleSheet(style)
            self._info_overlay.setFrameStyle(QtWidgets.QFrame.NoFrame)
            self._info_overlay.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
            self._info_overlay.viewport().setAutoFillBackground(False)
            self._info_overlay.setVisible(self._info_visible)

            self._renderer = renderer
            self.view = self._renderer.widget()
            bg = background if background is not None else _DISPLAY_CONFIG.get(
                "background", "k"
            )
            self._renderer.set_background_color(bg)
            grid_cfg = _DISPLAY_CONFIG.get("grid", {})
            g_size = float(grid_cfg.get("size", 20.0))
            g_spacing = float(grid_cfg.get("spacing", 1.0))
            self._renderer.configure_grid(g_size, g_spacing)
            self._renderer.set_grid_visible(self._grid_visible)
            configure_camera = getattr(self._renderer, "configure_camera", None)
            if callable(configure_camera):
                configure_camera(
                    near_clip=self._camera_near_clip,
                    far_clip=self._camera_far_clip,
                    min_near_clip=self._camera_min_near_clip,
                    max_near_clip=self._camera_max_near_clip,
                    clip_wheel_scale=self._camera_clip_wheel_scale,
                )
            renderer_widget = self._renderer.widget()
            renderer_widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            container_layout.addWidget(renderer_widget, 0, 0)
            container_layout.addWidget(
                self._info_overlay,
                0,
                0,
                2 if full_height else 1,
                1,
                alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
            )
            self._info_overlay.raise_()
            layout.addWidget(container, 1)
        else:
            self._renderer = None

        if self._renderer is None:
            self._show_disabled_label()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_structure(self, structure: object) -> None:
        """Set structure from a ChiSurf ``Structure``-like object.

        The object is expected to provide either:
        - ``atoms``: NumPy structured array with fields ``'xyz'`` and
          ``'atom_name'`` (as in :mod:`chisurf.structure`), or
        - ``xyz``: array-like of shape ``(N, 3)``.
        """

        self._atoms = None
        self._all_atom_coords = None
        self._all_atom_res_ids = None
        self._all_atom_radii = None
        self._atom_features = {}
        self._atom_feature_meta = {}
        self._show_atom_gaussians = False
        self._bond_pairs = None
        self._bond_pairs = None
        self._secondary_structure = None

        # Case 1: explicit atoms array with xyz/atom_name fields (preferred)
        atoms = getattr(structure, "atoms", None)
        if isinstance(atoms, np.ndarray) and {"xyz", "atom_name"}.issubset(
            set(atoms.dtype.fields or {})
        ):
            self._atoms = atoms
            fields_atoms = set(atoms.dtype.fields or {})

            coords_all_raw: Optional[np.ndarray]
            try:
                coords_all_raw = np.asarray(atoms["xyz"], dtype=float)
            except Exception:
                coords_all_raw = None
            self._all_atom_coords = None if coords_all_raw is None else coords_all_raw.copy()
            if "res_id" in fields_atoms:
                try:
                    self._all_atom_res_ids = np.asarray(atoms["res_id"])
                except Exception:
                    self._all_atom_res_ids = None
            else:
                self._all_atom_res_ids = None
            if "radius" in fields_atoms:
                try:
                    self._all_atom_radii = np.asarray(atoms["radius"], dtype=float)
                except Exception:
                    self._all_atom_radii = None
            else:
                self._all_atom_radii = None

            # Pre-compute simple covalent bonds for sticks representation
            # using a distance cutoff in *raw* (unscaled) coordinates so the
            # list is independent of the global scaling we apply for viewing.
            sticks_cfg = _DISPLAY_CONFIG.get("sticks", {})
            max_bond_len = float(sticks_cfg.get("bond_max_length", 1.9))
            if coords_all_raw is not None and np.isfinite(max_bond_len) and max_bond_len > 0.0:
                self._bond_pairs = _build_bond_pairs(coords_all_raw, max_bond_len)
            else:
                self._bond_pairs = None

            coords, res_ids, res_names, chain_ids = _extract_ca_trace(atoms)
            if coords is None:
                coords = np.asarray(atoms["xyz"], dtype=float)
                res_ids = None
                res_names = None
                chain_ids = None

            self._residue_ids = res_ids
            self._residue_names = res_names
            self._residue_oneletter = _three_to_one_array(res_names)
            self._residue_chain_ids = chain_ids

            coords_arr = np.asarray(coords, dtype=float)

            # Determine a common center/radius from all atoms if available,
            # otherwise from the CA trace, then center and scale geometry.
            if self._all_atom_coords is not None:
                center, radius = _compute_center_radius(self._all_atom_coords)
            else:
                center, radius = _compute_center_radius(coords_arr)

            scale = float(self._scale_factor)
            coords_arr = (coords_arr - center) * scale
            self._coords = coords_arr

            if self._all_atom_coords is not None:
                self._all_atom_coords = (self._all_atom_coords - center) * scale
            if self._all_atom_radii is not None:
                try:
                    self._all_atom_radii = (
                        np.asarray(self._all_atom_radii, dtype=float) * scale
                    )
                except Exception:
                    self._all_atom_radii = None

            self._center = np.zeros(3, dtype=float)
            self._radius = float(radius * scale)
            self._trace_ups = _build_trace_ups(atoms, self._residue_ids, self._coords)

            try:
                n_res = int(self._coords.shape[0])
            except Exception:
                n_res = 0
            if atoms is not None and n_res > 0:
                try:
                    ss_codes = assign_ss_c3_from_atoms(atoms, n_res, verbose=False)
                except Exception:
                    ss_codes = None
                if ss_codes:
                    try:
                        self._secondary_structure = np.asarray(ss_codes, dtype="U1")
                    except Exception:
                        self._secondary_structure = None

            n_res = self._coords.shape[0]
            self._cartoon_mask = np.ones(n_res, dtype=bool)
            
            n_atoms = self._all_atom_coords.shape[0] if self._all_atom_coords is not None else 0
            self._ball_mask = np.zeros(n_atoms, dtype=bool)
            self._sticks_mask = np.zeros(n_atoms, dtype=bool)
            
            self._update_view()
            return

        # Case 2: fallback to ``structure.xyz`` attribute
        xyz_attr = getattr(structure, "xyz", None)
        if xyz_attr is not None:
            self.set_coordinates(np.asarray(xyz_attr, dtype=float))
            return

        raise TypeError(
            "Unsupported structure type for Chimol: " f"{type(structure)!r}"
        )

    def set_coordinates(self, xyz: np.ndarray) -> None:
        """Set raw coordinates for visualization.

        Parameters
        ----------
        xyz:
            Array of shape ``(N, 3)`` with Cartesian coordinates.
        """

        arr = np.asarray(xyz, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")

        self._atoms = None
        self._all_atom_coords = None
        self._all_atom_res_ids = None
        self._all_atom_radii = None
        self._atom_features = {}
        self._atom_feature_meta = {}
        self._show_atom_gaussians = False

        center, radius = _compute_center_radius(arr)
        scale = float(self._scale_factor)
        arr = (arr - center) * scale

        self._coords = arr
        self._center = np.zeros(3, dtype=float)
        self._radius = float(radius * scale)
        self._update_view()

        # No sequence information when only raw coordinates are provided
        if self._residue_ids is None and self._residue_names is None:
            self._colors_per_ca = None
        self._colors_per_residue_override = None
        self._colors_per_atom_override = None
        self._cartoon_mask = None
        self._ball_mask = None

    def set_frames(self, frames: np.ndarray, *, object_id: Optional[str] = None) -> None:
        arr = np.asarray(frames, dtype=float)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("frames must have shape (T, N, 3)")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("frames must contain at least one frame and one point")

        flat = arr.reshape(-1, 3)
        center, radius = _compute_center_radius(flat)
        scale = float(self._scale_factor)
        arr_scaled = (arr - center) * scale

        with self._activate_object(object_id):
            state = self._get_active_state()
            state.frames = arr_scaled
            state.active_frame = 0
            self._coords = arr_scaled[0]
            self._center = np.zeros(3, dtype=float)
            self._radius = float(radius * scale)
            self._selected_residues = []
            try:
                self._update_view()
            except Exception:
                pass

    def set_active_frame(self, index: int, *, object_id: Optional[str] = None) -> None:
        with self._activate_object(object_id):
            state = self._get_active_state()
            frames = getattr(state, "frames", None)
            if frames is None:
                return
            try:
                arr = np.asarray(frames, dtype=float)
            except Exception:
                return
            if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0:
                return
            n_frames = arr.shape[0]
            try:
                idx = int(index)
            except Exception:
                return
            if idx < 0:
                idx = 0
            if idx >= n_frames:
                idx = n_frames - 1
            state.active_frame = idx
            self._coords = arr[idx]
            try:
                self._update_view(fit_camera=False)
            except Exception:
                pass

    def get_frame_count(self, object_id: Optional[str] = None) -> int:
        with self._activate_object(object_id):
            state = self._get_active_state()
            frames = getattr(state, "frames", None)
            if frames is None:
                return 0
            try:
                arr = np.asarray(frames, dtype=float)
            except Exception:
                return 0
            if arr.ndim != 3 or arr.shape[2] != 3:
                return 0
            return int(arr.shape[0])

    def get_active_frame_index(self, object_id: Optional[str] = None) -> int:
        with self._activate_object(object_id):
            state = self._get_active_state()
            try:
                idx = int(getattr(state, "active_frame", 0))
            except Exception:
                idx = 0
            frames = getattr(state, "frames", None)
            if frames is None:
                return 0
            try:
                n = int(np.asarray(frames).shape[0])
            except Exception:
                return max(idx, 0)
            if n <= 0:
                return 0
            if idx < 0:
                idx = 0
            if idx >= n:
                idx = n - 1
            return idx

    def set_background_color(self, color) -> None:
        """Set the viewer background color.

        This is a thin wrapper around the renderer's ``set_background_color``
        method and accepts the same Qt-compatible color values (strings like
        "black" or RGB(A) tuples).
        """

        renderer = self._renderer
        if renderer is None:
            return
        try:
            renderer.set_background_color(color)
        except Exception:
            pass

    def reset_view(self) -> None:
        """Reset the camera to show all visible objects at default orientation."""
        if self._renderer is None:
            return

        radius = 0.0
        try:
            scene = self._scene
            if scene is not None:
                radius = float(getattr(scene, "radius", 0.0))
        except Exception:
            pass

        if radius <= 0.0:
            radius = float(getattr(self, "_radius", 10.0))

        # Use defaults
        self._renderer.reset_view(
            distance=max(radius * 3.0, 5.0),
            elevation=float(self._default_elevation),
            azimuth=float(self._default_azimuth)
        )
        self._renderer.fit_to_radius(radius)

    def center(self, indices: Optional[Sequence[int]] = None, *, object_id: Optional[str] = None) -> None:
        """Center camera on the geometric center of target residues."""
        coords = self.get_residue_positions(indices, object_id=object_id)
        if coords.size == 0 or self._renderer is None:
            return
        center = coords.mean(axis=0)
        self._renderer.look_at(center)

    def zoom(self, indices: Optional[Sequence[int]] = None, *, buffer: float = 2.0, object_id: Optional[str] = None) -> None:
        """Zoom camera to fit target residues."""
        coords = self.get_residue_positions(indices, object_id=object_id)
        if coords.size == 0:
            self.reset_view()
            return

        # Attempt to use geometry utils if reachable, else use simple bounds
        try:
            from ..geometry import _compute_center_radius
            center, radius = _compute_center_radius(coords)
        except ImportError:
            # Fallback to simple mean/std or box-center
            mn, mx = coords.min(axis=0), coords.max(axis=0)
            center = (mn + mx) * 0.5
            radius = np.linalg.norm(mx - mn) * 0.5

        if self._renderer is not None:
            self._renderer.look_at(center)
            self._renderer.fit_to_radius(radius + float(buffer))

    def orient(self, indices: Optional[Sequence[int]] = None, *, object_id: Optional[str] = None) -> None:
        """Orient view to principal axes of target residues."""
        # TODO: Implement PCA-based alignment once QtGLRenderer supports arbitrary rotation matrices.
        # For now, zoom to fit provides the best "orient" approximation.
        self.zoom(indices, object_id=object_id)

    # ------------------------------------------------------------------
    # Representation / interaction helpers
    # ------------------------------------------------------------------
    def set_residue_representation(
        self,
        indices,
        cartoon: Optional[bool] = None,
        ball: Optional[bool] = None,
        *,
        object_id: Optional[str] = None,
    ) -> None:
        """Enable/disable cartoon and ball view for selected residues.

        Parameters
        ----------
        indices:
            Iterable of integer residue indices (0-based along the CA trace).
        cartoon:
            If True, enable cartoon for these residues; if False, disable;
            if None, leave unchanged.
        ball:
            If True, enable ball view for these residues; if False, disable;
            if None, leave unchanged.
        """

        changed = False
        with self._activate_object(object_id):
            coords = self._coords
            if coords is None:
                return
            n = coords.shape[0]
            if n == 0:
                return

            if self._cartoon_mask is None or len(self._cartoon_mask) != n:
                self._cartoon_mask = np.ones(n, dtype=bool)
            if self._ball_mask is None or len(self._ball_mask) != n:
                self._ball_mask = np.zeros(n, dtype=bool)

            if indices is None:
                idx_list = np.arange(n, dtype=int)
            else:
                try:
                    idx_arr = np.asarray(list(indices), dtype=int)
                except Exception:
                    idx_list = np.arange(n, dtype=int)
                else:
                    if idx_arr.size == 0:
                        idx_list = np.arange(n, dtype=int)
                    else:
                        idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < n)]
                        idx_list = idx_arr if idx_arr.size else np.arange(n, dtype=int)

            if cartoon is not None:
                self._cartoon_mask[idx_list] = bool(cartoon)
                try:
                    self._show_cartoon = bool(self._cartoon_mask.any())
                except Exception:
                    self._show_cartoon = True
                changed = True
            if ball is not None:
                self._ball_mask[idx_list] = bool(ball)
                try:
                    self._show_atoms = bool(self._ball_mask.any())
                except Exception:
                    self._show_atoms = True
                changed = True

        if changed:
            self._update_view()

    def set_secondary_structure_codes(self, codes) -> None:
        """Set per-residue secondary-structure codes for coloring.

        Parameters
        ----------
        codes:
            Iterable of single-character secondary-structure labels (e.g.
            'H', 'E', 'C') aligned to the CA trace.
        """

        try:
            arr = np.asarray(list(codes), dtype="U1")
        except Exception:
            return
        if arr.size == 0:
            self._secondary_structure = None
            return
        self._secondary_structure = arr
        if self._coords is not None and self._coords.shape[0] > 0:
            self._update_view()

    def set_plane_visible(self, visible: bool) -> None:
        """Show or hide the reference plane (grid)."""

        self._grid_visible = bool(visible)
        if self._renderer is not None:
            try:
                self._renderer.set_grid_visible(self._grid_visible)
            except Exception:
                pass

    def toggle_plane(self) -> None:
        self.set_plane_visible(not self._grid_visible)

    def set_surface_visible(self, visible: bool) -> None:
        self._surface_visible = bool(visible)
        if self._renderer is not None and self._coords is not None:
            self._update_view()

    def set_color_mode(self, mode: str) -> None:
        """Set coloring mode.

        Parameters
        ----------
        mode:
            "single" for uniform coloring, "by_residue" to color amino
            acids differently, or "by_secondary_structure" to color by
            secondary-structure state.
        """

        if mode not in (
            "single",
            "by_residue",
            "by_secondary_structure",
            "by_sequence",
            "by_element",
            "by_chain",
            "spectrum",
        ):
            return
        self._color_mode = mode
        if self._coords is not None:
            self._update_view(fit_camera=False)

    def set_representation(self, mode: str) -> None:
        """Legacy mode-style API (cartoon / ca_trace / atoms).

        This is primarily used by keyboard shortcuts and :class:`MolViewPlot`.
        Internally it configures the independent representation toggles
        (``_show_cartoon``, ``_show_trace``, ``_show_atoms``) and the
        per-residue ball mask, then refreshes the view.
        """

        mode_l = str(mode).lower()
        if mode_l not in ("cartoon", "ca_trace", "atoms"):
            return
        self._representation_mode = mode_l

        if self._coords is None:
            return
        n = self._coords.shape[0]
        if n <= 0:
            return

        if mode_l == "cartoon":
            self._show_cartoon = True
            self._show_trace = False
            self._show_atoms = False
        elif mode_l == "ca_trace":
            self._show_cartoon = False
            self._show_trace = True
            self._show_atoms = False
        else:  # "atoms"
            self._show_cartoon = False
            self._show_trace = False
            self._show_atoms = True

        # For atoms mode, default to showing balls on all residues.
        if mode_l == "atoms":
            self._cartoon_mask = np.ones(n, dtype=bool)
            self._ball_mask = np.ones(n, dtype=bool)
        else:
            if self._cartoon_mask is None or len(self._cartoon_mask) != n:
                self._cartoon_mask = np.ones(n, dtype=bool)
            else:
                self._cartoon_mask[:] = True
            if self._ball_mask is None or len(self._ball_mask) != n:
                self._ball_mask = np.zeros(n, dtype=bool)
            else:
                self._ball_mask[:] = False

        self._update_view()

    def set_cartoon_visible(self, visible: bool) -> None:
        """Enable or disable the cartoon tube globally."""

        self._show_cartoon = bool(visible)
        if self._coords is not None:
            self._update_view()

    def set_trace_visible(self, visible: bool) -> None:
        """Enable or disable the CA trace line globally."""

        self._show_trace = bool(visible)
        if self._coords is not None:
            self._update_view()

    def set_atoms_visible(self, visible: bool) -> None:
        """Enable or disable the atom/ball representation globally."""

        state = self._get_active_state()
        self._set_state_atoms_visible(state, bool(visible))
        self._update_view(fit_camera=False)

    def set_atom_gaussians_visible(self, visible: bool) -> None:
        self._show_atom_gaussians = bool(visible)
        if self._coords is not None:
            self._update_view()

    def set_dots_visible(self, visible: bool) -> None:
        self._show_dots = bool(visible)
        if self._coords is not None:
            self._update_view()

    def toggle_dots(self) -> None:
        self.set_dots_visible(not self._show_dots)

    def set_atoms_visible_all(self, visible: bool) -> None:
        """Toggle atoms representation for every loaded object."""

        changed = False
        vis = bool(visible)
        for entry in self._objects.values():
            prev_mask = entry.state.ball_mask
            self._set_state_atoms_visible(entry.state, vis)
            if entry.state.ball_mask is not prev_mask or entry.state.show_atoms != vis:
                changed = True
        if changed:
            self._update_view()

    def _set_state_atoms_visible(self, state: _MolViewObjectState, visible: bool) -> None:
        state.show_atoms = bool(visible)

        coords = state.coords
        if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[0] > 0:
            n = coords.shape[0]
            mask = np.ones(n, dtype=bool) if visible else np.zeros(n, dtype=bool)
            state.ball_mask = mask
        else:
            if not visible:
                state.ball_mask = None

    def set_sticks_visible(self, visible: bool) -> None:
        """Enable or disable the sticks (bond) representation globally."""

        self._show_sticks = bool(visible)
        if self._coords is not None:
            self._update_view()

    def handle_key_event(self, ev: QtGui.QKeyEvent) -> bool:  # type: ignore[name-defined]
        """Handle keyboard shortcuts for basic viewer controls.

        r - cartoon/ribbon mode
        c - CA trace mode
        b - atoms/ball mode
        s - toggle sidechains on/off (atoms view)
        q - close the containing window
        """

        try:
            ch = ev.text().lower()
        except Exception:
            return False

        if ch == "r":
            self.set_representation("cartoon")
            return True
        if ch == "c":
            self.set_representation("ca_trace")
            return True
        if ch == "b":
            self.set_representation("atoms")
            return True
        if ch == "d":
            self.toggle_dots()
            return True
        if ch == "s":
            self.toggle_sidechains()
            return True
        if ch == "q":
            w = self.window()
            if w is not None:
                try:
                    w.close()
                except Exception:
                    pass
            return True
        return False

    # ------------------------------------------------------------------
    # Picking / mouse interaction
    # ------------------------------------------------------------------

    def handle_mouse_click(self, ev: QtGui.QMouseEvent) -> None:  # type: ignore[name-defined]
        """Handle a mouse-click in the GL view for residue picking.

        A left-click near the backbone/CA trace selects the nearest residue;
        clicking in empty space clears the selection.
        """

        indices = []
        mods = None
        if self._coords is not None and getattr(self, "_gl_enabled", False) and self.view is not None:
            picking_mod = _get_picking_module()
            try:
                sel_cfg = _DISPLAY_CONFIG.get("selection", {})
            except Exception:
                sel_cfg = {}
            try:
                radius_px = float(sel_cfg.get("click_radius_px", 8.0))
            except Exception:
                radius_px = 8.0
            if not np.isfinite(radius_px) or radius_px <= 0.0:
                radius_px = 8.0

            try:
                mods = ev.modifiers()
            except Exception:
                mods = None

            picked_idx = None
            if picking_mod is not None:
                try:
                    picked_idx = picking_mod.pick_residue_from_click(
                        self._coords,
                        self.view,
                        ev,
                        radius_px,
                    )
                except Exception:
                    picked_idx = None

            try:
                pos = ev.pos()
                x = int(pos.x())
                y = int(pos.y())
            except Exception:
                x = int(ev.x())
                y = int(ev.y())

            if picked_idx is not None:
                indices = [picked_idx]
            else:
                half = int(max(1, round(radius_px)))
                rect = QtCore.QRect(x - half, y - half, 2 * half, 2 * half)

                if picking_mod is not None:
                    try:
                        idx_arr = picking_mod.pick_residues_in_rect(self._coords, self.view, rect)
                    except Exception:
                        idx_arr = np.zeros(0, dtype=int)
                else:
                    idx_arr = np.zeros(0, dtype=int)

                try:
                    indices = [int(i) for i in np.asarray(idx_arr, dtype=int) if int(i) >= 0]
                except Exception:
                    indices = []

        try:
            self._apply_selection_indices(indices, mods)
        except Exception:
            # Fallback: ignore selection errors for robustness.
            pass

        if self._coords is not None and getattr(self, "_gl_enabled", False):
            try:
                self._update_view()
            except Exception:
                pass

    def _apply_selection_indices(self, indices, modifiers=None) -> None:
        try:
            mods = modifiers
            ctrl = bool(mods & QtCore.Qt.ControlModifier) if mods is not None else False
        except Exception:
            ctrl = False

        try:
            idx_list = [int(i) for i in list(indices)]
        except Exception:
            idx_list = []

        if idx_list:
            if ctrl:
                try:
                    current = set(
                        int(i)
                        for i in getattr(self, "_selected_residues", [])
                        if int(i) >= 0
                    )
                except Exception:
                    current = set()
                region = set(i for i in idx_list if i >= 0)
                new_sel = sorted(current.symmetric_difference(region))
                self._selected_residues = new_sel
            else:
                self._selected_residues = idx_list
            selection = list(self._selected_residues)
            try:
                self.residueSelectionChanged.emit(selection)
            except Exception:
                pass
            try:
                self.objectResidueSelectionChanged.emit(self.get_active_object_id(), selection)
            except Exception:
                pass
        else:
            self._selected_residues = []
            try:
                self.residueSelectionChanged.emit([])
            except Exception:
                pass
            try:
                self.objectResidueSelectionChanged.emit(self.get_active_object_id(), [])
            except Exception:
                pass


    def handle_rect_selection(self, rect, modifiers=None) -> None:
        if self._coords is None or not getattr(self, "_gl_enabled", False) or self.view is None:
            return

        picking_mod = _get_picking_module()
        if picking_mod is not None:
            try:
                idx_arr = picking_mod.pick_residues_in_rect(self._coords, self.view, rect)
            except Exception:
                idx_arr = np.zeros(0, dtype=int)
        else:
            idx_arr = np.zeros(0, dtype=int)

        try:
            indices = [int(i) for i in np.asarray(idx_arr, dtype=int) if int(i) >= 0]
        except Exception:
            indices = []
        try:
            self._apply_selection_indices(indices, modifiers)
        except Exception:
            pass

    def _show_disabled_label(self, reason: Optional[str] = None) -> None:
        message = (
            "Chimol OpenGL viewer is disabled.\n"
            "Enable OpenGL support to use this tool."
        )
        if reason:
            message = f"{message}\n\n{reason}"

        if self._disabled_label is None:
            label = QtWidgets.QLabel(self)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setWordWrap(True)
            self._disabled_label = label
            self.layout().addWidget(label)

        self._disabled_label.setText(message)
        self._disabled_label.show()

    def on_renderer_error(self, message: str) -> None:
        self._renderer = None
        self._show_disabled_label(message)

        try:
            self._update_view()
        except Exception:
            pass

    def set_selected_residues(self, indices, *, object_id: Optional[str] = None) -> None:
        """Update selection from external widgets (e.g. sequence view)."""

        try:
            idx_iter = list(indices)
        except Exception:
            idx_iter = []

        with self._activate_object(object_id):
            if self._coords is None:
                self._selected_residues = []
                return

            n = self._coords.shape[0]
            if n <= 0:
                self._selected_residues = []
                return

            idx_list: list[int] = []
            for idx in idx_iter:
                try:
                    i = int(idx)
                except Exception:
                    continue
                if 0 <= i < n:
                    idx_list.append(i)

            self._selected_residues = idx_list

        try:
            self._update_view()
        except Exception:
            pass

    def get_sequence_arrays(
        self, object_id: Optional[str] = None
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._activate_object(object_id):
            seq = _copy_array(self._residue_oneletter)
            res = _copy_array(self._residue_names)
        return seq, res

    def get_residue_numbers(self, object_id: Optional[str] = None) -> Optional[np.ndarray]:
        with self._activate_object(object_id):
            ids = _copy_array(self._residue_ids)
        return ids

    def get_residue_colors(
        self, object_id: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Return per-residue RGBA colors for an object.

        The result matches the colors used for the CA trace and other
        residue-based representations, including the current ``color_mode``
        and any explicit per-residue overrides.
        """

        with self._activate_object(object_id):
            coords = self._coords
            if coords is None or getattr(coords, "size", 0) <= 0:
                return None
            try:
                n_points = int(np.asarray(coords, dtype=float).shape[0])
            except Exception:
                return None
            if n_points <= 0:
                return None

            # Reuse the same color computation path used for rendering.
            try:
                self._build_scene_for_current_object(object_prefix=None)
            except Exception:
                pass

            cols = getattr(self, "_colors_per_ca", None)
            if cols is None:
                return None
            try:
                arr = np.asarray(cols, dtype=float)
            except Exception:
                return None
            if arr.ndim != 2 or arr.shape[0] != n_points:
                return None
            return arr.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clear_items(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.clear()
            except Exception:
                pass

    def _update_cartoon(
        self, coords: np.ndarray, n_points: int, config: dict, colors: Optional[np.ndarray]
    ) -> list[SceneObject]:
        scene_objects: list[SceneObject] = []
        if not self._show_cartoon:
            return scene_objects

        radius_scale = float(config.get("radius_scale", 0.05))
        min_radius = float(config.get("min_radius", 0.4))
        ao_radius = float(config.get("ao_radius", 4.0))
        ao_max = int(config.get("ao_max_neighbors", 16))
        ao_strength = float(config.get("ao_strength", 0.45))

        coords_cartoon = coords
        colors_for_tube = colors
        idx_all = np.arange(n_points, dtype=int)
        idx_cartoon = idx_all

        # Apply per-residue cartoon mask by subselecting the CA points
        # used to build the tube.
        if self._cartoon_mask is not None and len(self._cartoon_mask) == n_points:
            mask = self._cartoon_mask.astype(bool)
            if mask.any():
                coords_cartoon = coords[mask]
                idx_cartoon = idx_all[mask]
                if colors_for_tube is not None:
                    colors_for_tube = colors_for_tube[mask]
            else:
                coords_cartoon = coords[:0]
                idx_cartoon = idx_all[:0]

        try:
            occ_ca = _estimate_ambient_occlusion(
                coords_cartoon,
                radius=ao_radius,
                max_neighbors=ao_max,
            )
        except Exception:
            occ_ca = None

        if (
            occ_ca is not None
            and np.asarray(occ_ca).shape[0] == coords_cartoon.shape[0]
            and colors_for_tube is not None
        ):
            colors_for_tube = colors_for_tube.copy()
            occ_ca_arr = np.asarray(occ_ca, dtype=float)
            shade = (1.0 - ao_strength) + ao_strength * (1.0 - occ_ca_arr)
            colors_for_tube[:, :3] *= shade.reshape(-1, 1)
            colors_for_tube = np.clip(colors_for_tube, 0.0, 1.0)

        if coords_cartoon is not None and coords_cartoon.shape[0] >= 2:
            n_cartoon = coords_cartoon.shape[0]
            idx_cartoon_arr = np.asarray(idx_cartoon, dtype=int)
            if idx_cartoon_arr.shape[0] != n_cartoon:
                idx_cartoon_arr = np.arange(n_cartoon, dtype=int)

            res_ids_full = getattr(self, "_residue_ids", None)
            chain_ids_full = getattr(self, "_residue_chain_ids", None)
            ss_full = getattr(self, "_secondary_structure", None)
            trace_ups_full = getattr(self, "_trace_ups", None)

            res_ids_cartoon = None
            if res_ids_full is not None:
                try:
                    res_full_arr = np.asarray(res_ids_full)
                    if res_full_arr.shape[0] == n_points:
                        res_ids_cartoon = res_full_arr[idx_cartoon_arr]
                except Exception:
                    res_ids_cartoon = None

            chain_ids_cartoon = None
            if chain_ids_full is not None:
                try:
                    chain_full_arr = np.asarray(chain_ids_full)
                    if chain_full_arr.shape[0] == n_points:
                        chain_ids_cartoon = chain_full_arr[idx_cartoon_arr]
                except Exception:
                    chain_ids_cartoon = None

            seg_bounds = [(0, n_cartoon)]
            if res_ids_cartoon is not None or chain_ids_cartoon is not None:
                seg_bounds = []
                start = 0
                for i in range(n_cartoon - 1):
                    gap = False
                    if chain_ids_cartoon is not None:
                        try:
                            ch0 = str(chain_ids_cartoon[i]).strip()
                            ch1 = str(chain_ids_cartoon[i + 1]).strip()
                        except Exception:
                            ch0 = ch1 = ""
                        if ch0 != ch1:
                            gap = True
                    if not gap and res_ids_cartoon is not None:
                        try:
                            r0 = int(res_ids_cartoon[i])
                            r1 = int(res_ids_cartoon[i + 1])
                            if (r1 - r0) != 1:
                                gap = True
                        except Exception:
                            pass
                    if gap:
                        if i + 1 - start >= 2:
                            seg_bounds.append((start, i + 1))
                        start = i + 1
                if n_cartoon - start >= 2:
                    seg_bounds.append((start, n_cartoon))
                if not seg_bounds and n_cartoon >= 2:
                    seg_bounds = [(0, n_cartoon)]

            ss_full_arr = None
            if ss_full is not None:
                try:
                    ss_full_arr = np.asarray(ss_full)
                except Exception:
                    ss_full_arr = None

            trace_ups_cartoon = None
            if trace_ups_full is not None:
                try:
                    ups_full_arr = np.asarray(trace_ups_full, dtype=float)
                    if ups_full_arr.shape[0] == n_points:
                        trace_ups_cartoon = ups_full_arr[idx_cartoon_arr]
                except Exception:
                    trace_ups_cartoon = None

            for start, end in seg_bounds:
                seg_coords = coords_cartoon[start:end]
                if seg_coords.shape[0] < 2:
                    continue

                if colors_for_tube is not None:
                    seg_colors = colors_for_tube[start:end]
                else:
                    seg_colors = None

                if trace_ups_cartoon is not None and trace_ups_cartoon.shape[0] == n_cartoon:
                    seg_trace_ups = trace_ups_cartoon[start:end]
                else:
                    seg_trace_ups = None

                seg_ss = None
                if ss_full_arr is not None and ss_full_arr.shape[0] == n_points:
                    seg_indices = idx_cartoon_arr[start:end]
                    seg_ss = ss_full_arr[seg_indices]

                arrays = _generate_cartoon_tube_arrays(
                    seg_coords,
                    seg_colors,
                    seg_trace_ups,
                    base_radius=max(self._radius * radius_scale, min_radius),
                    style=str(config.get("style", "tube")),
                    ss_codes=seg_ss,
                    config=config,
                )

                if arrays is not None:
                    verts, norms, faces_arr, cols = arrays
                    geom = Geometry(
                        kind="mesh",
                        positions=verts,
                        indices=faces_arr,
                        normals=norms,
                        colors=cols,
                    )
                    scene_objects.append(
                        SceneObject(id="cartoon", geometry=geom, render_mode="opaque")
                    )

        return scene_objects

    def _update_trace(self, coords: np.ndarray, colors: Optional[np.ndarray]) -> list[SceneObject]:
        scene_objects: list[SceneObject] = []
        if not self._show_trace:
            return scene_objects

        coords_line, colors_line = _generate_trace_arrays(coords, colors)
        if colors_line is None:
            colors_line = colors

        geom = Geometry(kind="line", positions=coords_line, colors=colors_line)
        scene_objects.append(SceneObject(id="trace", geometry=geom, render_mode="opaque"))

        return scene_objects

    def _update_atoms(
        self, 
        coords: np.ndarray, 
        n_points: int, 
        balls_cfg: dict,
        colors_per_ca: Optional[np.ndarray]
    ) -> Optional[list[SceneObject]]:
        scene_objects: list[SceneObject] = []
        if not self._show_atoms:
            return scene_objects

        balls_size_scale = float(balls_cfg.get("size_scale", 0.04))
        balls_min_size = float(balls_cfg.get("min_size", 3.0))
        balls_radius_multiplier = float(balls_cfg.get("radius_multiplier", 1.0))
        balls_ao_radius = float(balls_cfg.get("ao_radius", 4.0))
        balls_ao_max = int(balls_cfg.get("max_neighbors", 24))
        balls_ao_strength = float(balls_cfg.get("ao_strength", 0.5))
        balls_max_atoms = int(balls_cfg.get("max_atoms", 8000))
        base_global_radius = max(self._radius * balls_size_scale, balls_min_size)

        used_all_atoms_for_balls = False
        if (
            self._ball_mask is not None
            and len(self._ball_mask) == n_points
            and self._ball_mask.any()
            and self._atoms is not None
            and self._residue_ids is not None
        ):
            atoms = self._atoms
            fields_atoms = set(atoms.dtype.fields or {})
            atom_names = None
            if "res_id" in fields_atoms:
                try:
                    atom_res_id = np.asarray(atoms["res_id"])
                except Exception:
                    atom_res_id = None
            else:
                atom_res_id = None

            # Prefer the already-centered and scaled all-atom coordinates so
            # atoms and cartoon/surface share the exact same frame.
            if self._all_atom_coords is not None:
                atom_xyz = np.asarray(self._all_atom_coords, dtype=float)
            elif "xyz" in fields_atoms:
                try:
                    atom_xyz = np.asarray(atoms["xyz"], dtype=float)
                except Exception:
                    atom_xyz = None
            else:
                atom_xyz = None

            if "atom_name" in fields_atoms:
                try:
                    atom_names = np.char.strip(atoms["atom_name"].astype(str))
                except Exception:
                    atom_names = None

            if atom_xyz is not None and atom_res_id is not None:
                n_atoms_total = atom_xyz.shape[0]
                
                if self._ball_mask is not None and len(self._ball_mask) == n_atoms_total:
                    # Per-atom mask
                    atom_mask = self._ball_mask.astype(bool)
                elif self._ball_mask is not None and len(self._ball_mask) == n_points:
                    # Legacy: residue-level mask
                    sel_idx = np.nonzero(self._ball_mask)[0]
                    sel_res_ids = np.unique(self._residue_ids[sel_idx])
                    atom_mask = np.isin(atom_res_id, sel_res_ids)
                else:
                    atom_mask = np.zeros(n_atoms_total, dtype=bool)

                # Optionally hide sidechains and keep only backbone atoms
                if not getattr(self, "_sidechains_visible", True) and atom_names is not None:
                    backbone_names = np.array(["N", "CA", "C", "O", "CB"], dtype=atom_names.dtype)
                    backbone_mask = np.isin(atom_names, backbone_names)
                    atom_mask = atom_mask & backbone_mask
                
                pts = atom_xyz[atom_mask]
                if pts.size:
                    pts = np.asarray(pts, dtype=float)
                    radii_sel: Optional[np.ndarray]
                    if (
                        self._all_atom_radii is not None
                        and len(self._all_atom_radii) == atom_xyz.shape[0]
                    ):
                        try:
                            radii_sel = np.asarray(self._all_atom_radii, dtype=float)[atom_mask]
                        except Exception:
                            radii_sel = None
                    else:
                        radii_sel = None

                    colors = np.zeros((pts.shape[0], 4), dtype=float)
                    colors[:] = np.asarray(self._base_color_single, dtype=float)

                    # Start from per-residue colors_per_ca if present.
                    if (
                        colors_per_ca is not None
                        and len(colors_per_ca) == len(self._residue_ids)
                    ):
                        color_map = {rid: colors_per_ca[i_res] for i_res, rid in enumerate(self._residue_ids)}
                    else:
                        color_map = {}

                    sel_atom_res = atom_res_id[atom_mask]
                    for i_atom, rid in enumerate(sel_atom_res):
                        base_col = color_map.get(rid, self._base_color_single)
                        colors[i_atom, :] = base_col

                    # Apply per-atom override if present.
                    if (
                        getattr(self, "_colors_per_atom_override", None) is not None
                        and len(self._colors_per_atom_override) == atom_res_id.shape[0]
                    ):
                        ov = np.asarray(self._colors_per_atom_override, dtype=float)
                        ov_sel = ov[atom_mask]
                        for i_atom in range(pts.shape[0]):
                            col_ov = ov_sel[i_atom]
                            if np.isfinite(col_ov).all():
                                colors[i_atom, :] = col_ov

                    # Make balls fully opaque
                    colors[:, 3] = 1.0

                    # Lightweight ambient-occlusion style darkening to
                    # improve depth perception in ball view.
                    try:
                        occ_balls = _estimate_ambient_occlusion(
                            pts,
                            radius=balls_ao_radius,
                            max_neighbors=balls_ao_max,
                        )
                    except Exception:
                        occ_balls = None
                    if (
                        occ_balls is not None
                        and np.asarray(occ_balls).shape[0] == pts.shape[0]
                    ):
                        occ_b = np.asarray(occ_balls, dtype=float)
                        shade_balls = (1.0 - balls_ao_strength) + (
                            balls_ao_strength * (1.0 - occ_b)
                        )
                        colors[:, :3] *= shade_balls.reshape(-1, 1)
                        colors = np.clip(colors, 0.0, 1.0)

                    max_atoms = max(1, balls_max_atoms)
                    if pts.shape[0] > max_atoms:
                        step = max(1, pts.shape[0] // max_atoms)
                        pts = pts[::step]
                        colors = colors[::step]
                        if radii_sel is not None and radii_sel.shape[0] >= pts.shape[0]:
                            radii_sel = radii_sel[::step]

                    if radii_sel is not None and radii_sel.shape[0] == pts.shape[0]:
                        radii_for_mesh = radii_sel * balls_radius_multiplier
                    else:
                        radii_for_mesh = np.full(
                            pts.shape[0], base_global_radius, dtype=float
                        )

                    invalid_r = (~np.isfinite(radii_for_mesh)) | (radii_for_mesh <= 0.0)
                    if invalid_r.any():
                        radii_for_mesh[invalid_r] = base_global_radius
                    radii_for_mesh = np.maximum(radii_for_mesh, balls_min_size)

                    # Render all balls for the current selection as a
                    # single merged mesh. This is much faster than
                    # creating one GLMeshItem per atom while still
                    # providing proper shaded spheres, similar to pyball.
                    sphere_mesh = _build_sphere_mesh(radius=1.0)
                    if sphere_mesh is not None and pts.shape[0] > 0:
                        base_verts = sphere_mesh.get("vertices")
                        base_norms = sphere_mesh.get("normals")
                        base_faces = sphere_mesh.get("faces")
                        if (
                            base_verts is not None
                            and base_faces is not None
                            and base_norms is not None
                            and base_verts.size
                            and base_faces.size
                            and base_norms.size
                        ):
                            if (
                                base_verts is not None
                                and base_faces is not None
                                and base_norms is not None
                                and base_verts.size
                                and base_faces.size
                                and base_norms.size
                            ):
                                n_atoms = pts.shape[0]
                                n_verts = base_verts.shape[0]

                                # Duplicate and translate sphere vertices for
                                # each atom center: (N_atoms, N_verts, 3)
                                verts = base_verts[np.newaxis, :, :] * (
                                    radii_for_mesh[:, np.newaxis, np.newaxis]
                                )
                                verts += pts[:, np.newaxis, :]
                                verts = verts.reshape(-1, 3)

                                # Duplicate faces with index offsets
                                faces = np.repeat(
                                    base_faces[np.newaxis, :, :], n_atoms, axis=0
                                )
                                offsets = (
                                    np.arange(n_atoms, dtype=base_faces.dtype)
                                    * n_verts
                                )
                                faces += offsets[:, np.newaxis, np.newaxis]
                                faces = faces.reshape(-1, 3)

                                norms = np.repeat(
                                    base_norms[np.newaxis, :, :], n_atoms, axis=0
                                ).reshape(-1, 3)

                                vcols = None
                                try:
                                    col_arr = np.asarray(colors, dtype=float)
                                    if col_arr.shape[0] == n_atoms:
                                        vcols = np.repeat(
                                            col_arr[:, np.newaxis, :],
                                            n_verts,
                                            axis=1,
                                        ).reshape(-1, 4)
                                except Exception:
                                    vcols = None

                                geom = Geometry(
                                    kind="mesh",
                                    positions=verts,
                                    indices=faces,
                                    normals=norms,
                                    colors=vcols,
                                )
                                scene_objects.append(
                                    SceneObject(id="atoms_mesh", geometry=geom, render_mode="opaque")
                                )

                                used_all_atoms_for_balls = True

        if self._show_atoms and not used_all_atoms_for_balls:
            sphere_radius = max(self._radius * balls_size_scale * 0.5, balls_min_size * 0.1)
            sphere = _build_sphere_mesh(radius=sphere_radius)
            if (
                self._ball_mask is not None
                and len(self._ball_mask) == n_points
                and self._ball_mask.any()
            ):
                indices = np.nonzero(self._ball_mask)[0]
            else:
                # Default: sparse sampling along the chain
                step = max(1, n_points // 50)
                indices = np.arange(0, n_points, step, dtype=int)

            point_positions: list[np.ndarray] = []
            point_colors: list[np.ndarray] = []

            for i in indices:
                center = coords[i]
                color = (
                    colors_per_ca[i]
                    if colors_per_ca is not None
                    else self._base_color_single
                )
                color_local = np.array(color, dtype=float)
                color_local[3] = 1.0
                point_positions.append(center)
                point_colors.append(color_local)

            if point_positions:
                radii_vals = None
                beads = getattr(self, "_bead_radii", None)
                if beads is not None and len(beads) == len(indices):
                    radii_vals = np.asarray(beads[indices], dtype=float)

                geom = Geometry(
                    kind="points",
                    positions=np.asarray(point_positions, dtype=float),
                    colors=np.asarray(point_colors, dtype=float),
                    radii=radii_vals,
                    meta={"glyph": "sphere", "radius": sphere_radius},
                )
                scene_objects.append(
                    SceneObject(id="atoms_points", geometry=geom, render_mode="opaque")
                )

        return scene_objects if scene_objects else None

    def _update_atom_gaussians(self, config: dict) -> Optional[list[SceneObject]]:
        if not self._show_atom_gaussians:
            return None

        features = getattr(self, "_atom_features", None) or {}
        covariances = features.get("gaussian_covariances")
        if covariances is None:
            return None

        coords = self._all_atom_coords
        if coords is None:
            coords = self._coords
        if coords is None:
            return None
        coords = np.asarray(coords, dtype=float)
        cov_arr = _coerce_covariance_array(covariances)
        if cov_arr is None:
            return None

        n_atoms = coords.shape[0]
        if n_atoms == 0:
            return None
        if cov_arr.shape[0] != n_atoms:
            count = min(n_atoms, cov_arr.shape[0])
            coords = coords[:count]
            cov_arr = cov_arr[:count]
            n_atoms = count

        valid = np.isfinite(coords).all(axis=1) & np.isfinite(cov_arr.reshape(n_atoms, -1)).all(axis=1)
        if not valid.any():
            return None

        meta_map = getattr(self, "_atom_feature_meta", None) or {}
        feature_meta = meta_map.get("gaussian_covariances", {}) or {}

        idx_override = feature_meta.get("indices") if isinstance(feature_meta, dict) else None
        if idx_override is not None:
            try:
                idx_override = np.asarray(idx_override, dtype=int)
                idx_override = idx_override[(idx_override >= 0) & (idx_override < n_atoms)]
            except Exception:
                idx_override = None

        indices = np.nonzero(valid)[0]
        if idx_override is not None and idx_override.size:
            indices = np.intersect1d(indices, idx_override, assume_unique=False)
        if indices.size == 0:
            return None

        def _resolve_float(key: str, default: float) -> float:
            value = default
            if isinstance(feature_meta, dict) and key in feature_meta:
                value = feature_meta[key]
            elif key in config:
                value = config[key]
            try:
                return float(value)
            except Exception:
                return float(default)

        scale = _resolve_float("scale", float(config.get("scale", 1.0)))
        min_axis = _resolve_float("min_axis", float(config.get("min_axis", 0.15)))
        max_axis = _resolve_float("max_axis", float(config.get("max_axis", 5.0)))
        max_atoms = int(_resolve_float("max_atoms", float(config.get("max_atoms", 256))))

        if max_atoms > 0 and indices.size > max_atoms:
            indices = indices[:max_atoms]

        color_default = config.get("color", [1.0, 0.5, 0.2, 0.35])
        if isinstance(feature_meta, dict) and "color" in feature_meta:
            color_default = feature_meta["color"]
        base_color = np.asarray(color_default, dtype=float)
        if base_color.shape[0] != 4:
            base_color = np.array([1.0, 0.5, 0.2, 0.35], dtype=float)

        per_atom_colors = None
        if isinstance(feature_meta, dict) and "colors" in feature_meta:
            try:
                per_atom_colors = np.asarray(feature_meta["colors"], dtype=float)
            except Exception:
                per_atom_colors = None

        sphere_mesh = _build_sphere_mesh(radius=1.0)
        if sphere_mesh is None:
            return None
        base_verts = np.asarray(sphere_mesh.get("vertices"))
        base_normals = np.asarray(sphere_mesh.get("normals"))
        base_faces = np.asarray(sphere_mesh.get("faces"), dtype=int)
        if base_verts.size == 0 or base_faces.size == 0:
            return None

        n_sel = indices.size
        n_verts = base_verts.shape[0]
        positions = np.zeros((n_sel * n_verts, 3), dtype=float)
        normals = np.zeros_like(positions)
        colors = np.zeros((n_sel * n_verts, 4), dtype=float)

        def _color_for_atom(idx_atom: int) -> np.ndarray:
            if per_atom_colors is not None and 0 <= idx_atom < per_atom_colors.shape[0]:
                col = np.asarray(per_atom_colors[idx_atom], dtype=float)
                if col.shape[0] == 3:
                    col = np.concatenate([col, np.array([base_color[3]])])
                return col
            return base_color

        for i, atom_idx in enumerate(indices):
            center = coords[atom_idx]
            cov_local = cov_arr[atom_idx]
            cov_local = 0.5 * (cov_local + cov_local.T)
            try:
                evals, evecs = np.linalg.eigh(cov_local)
            except np.linalg.LinAlgError:
                continue
            evals = np.clip(evals, 0.0, None)
            axes = np.sqrt(evals) * scale
            axes = np.clip(axes, min_axis, None)
            axes = np.clip(axes, None, max_axis)
            axes[axes <= 0.0] = min_axis

            transform = evecs @ np.diag(axes)
            verts_local = base_verts @ transform.T + center

            try:
                inv_t = np.linalg.pinv(transform).T
            except np.linalg.LinAlgError:
                inv_t = np.linalg.pinv(transform + np.eye(3) * 1e-6).T
            normals_local = base_normals @ inv_t
            norms = np.linalg.norm(normals_local, axis=1, keepdims=True)
            normals_local = normals_local / np.clip(norms, 1e-6, None)

            start = i * n_verts
            end = start + n_verts
            positions[start:end] = verts_local
            normals[start:end] = normals_local
            colors[start:end] = np.clip(_color_for_atom(atom_idx), 0.0, 1.0)

        if positions.size == 0:
            return None

        faces = np.repeat(base_faces[np.newaxis, :, :], n_sel, axis=0)
        offsets = (np.arange(n_sel, dtype=base_faces.dtype) * n_verts)[:, np.newaxis, np.newaxis]
        faces = (faces + offsets).reshape(-1, 3)

        geom = Geometry(
            kind="mesh",
            positions=positions,
            indices=faces,
            normals=normals,
            colors=colors,
        )
        render_mode = "transparent" if np.any(colors[:, 3] < 0.999) else "opaque"
        return [SceneObject(id="atom_gaussians", geometry=geom, render_mode=render_mode)]

    def _update_sticks(self, sticks_cfg: dict, colors_per_ca: Optional[np.ndarray]) -> Optional[list[SceneObject]]:
        if not self._show_sticks:
            return None
        if self._bond_pairs is None or self._all_atom_coords is None:
            return None

        sticks_width = float(sticks_cfg.get("width", 2.0))
        sticks_max_bonds = int(sticks_cfg.get("max_bonds", 20000))

        bonds = np.asarray(self._bond_pairs, dtype=int)
        if bonds.ndim == 2 and bonds.shape[1] == 2 and bonds.size:
            n_atoms_all = self._all_atom_coords.shape[0]
            # Clamp indices to valid range just in case.
            mask_valid = (
                (bonds[:, 0] >= 0)
                & (bonds[:, 0] < n_atoms_all)
                & (bonds[:, 1] >= 0)
                & (bonds[:, 1] < n_atoms_all)
                & (bonds[:, 0] != bonds[:, 1])
            )
            bonds = bonds[mask_valid]

            if self._sticks_mask is not None and len(self._sticks_mask) == n_atoms_all:
                 mask_bonds = self._sticks_mask[bonds[:, 0]] & self._sticks_mask[bonds[:, 1]]
                 bonds = bonds[mask_bonds]

            if bonds.size:
                # Optional bond downsampling for performance.
                if sticks_max_bonds > 0 and bonds.shape[0] > sticks_max_bonds:
                    step = max(1, bonds.shape[0] // sticks_max_bonds)
                    bonds = bonds[::step]

                pts_all = np.asarray(self._all_atom_coords, dtype=float)

                # Build per-atom colors from residue colors and/or explicit overrides.
                atom_res = np.asarray(self._all_atom_res_ids)
                atom_colors = np.tile(
                    np.asarray(self._base_color_single, dtype=float),
                    (pts_all.shape[0], 1),
                )

                if (
                    self._residue_ids is not None
                    and colors_per_ca is not None
                    and len(colors_per_ca) == len(self._residue_ids)
                ):
                    color_map = {rid: colors_per_ca[i_res] for i_res, rid in enumerate(self._residue_ids)}
                    for i_atom, rid in enumerate(atom_res):
                        atom_colors[i_atom, :] = color_map.get(
                            rid, self._base_color_single
                        )

                if (
                    getattr(self, "_colors_per_atom_override", None) is not None
                    and len(self._colors_per_atom_override) == atom_res.shape[0]
                ):
                    ov = np.asarray(self._colors_per_atom_override, dtype=float)
                    for i_atom in range(atom_res.shape[0]):
                        col_ov = ov[i_atom]
                        if np.isfinite(col_ov).all():
                            atom_colors[i_atom, :] = col_ov

                n_bonds = bonds.shape[0]
                seg_pos = np.empty((n_bonds * 2, 3), dtype=np.float32)
                seg_col = np.empty((n_bonds * 2, 4), dtype=np.float32)

                seg_pos[0::2, :] = pts_all[bonds[:, 0]]
                seg_pos[1::2, :] = pts_all[bonds[:, 1]]
                seg_col[0::2, :] = atom_colors[bonds[:, 0]]
                seg_col[1::2, :] = atom_colors[bonds[:, 1]]
                seg_col[:, 3] = 1.0

                geom = Geometry(kind="line", positions=seg_pos, colors=seg_col)
                scene_obj = SceneObject(id="sticks", geometry=geom, render_mode="opaque")

                return [scene_obj]

        return None

    def _update_restraints(self, state: _MolViewObjectState) -> Optional[list[SceneObject]]:
        """Build geometry for RMF restraint pseudobonds."""
        if not state.restraints or state.all_atom_coords is None:
            return None
        
        pts = state.all_atom_coords
        n_restraints = len(state.restraints)
        seg_pos = np.empty((n_restraints * 2, 3), dtype=np.float32)
        
        for i, r in enumerate(state.restraints):
            idx1, idx2 = r["indices"]
            if idx1 < pts.shape[0] and idx2 < pts.shape[0]:
                seg_pos[i*2] = pts[idx1]
                seg_pos[i*2 + 1] = pts[idx2]
            else:
                seg_pos[i*2] = [0, 0, 0]
                seg_pos[i*2 + 1] = [0, 0, 0]
            
        geom = Geometry(kind="line", positions=seg_pos)
        # Warm orange/yellow for restraints
        geom.colors = np.tile([1.0, 0.6, 0.2, 1.0], (n_restraints * 2, 1)).astype(np.float32)
        
        return [SceneObject(id="restraints", geometry=geom, render_mode="opaque")]

    def _update_surface(
        self, 
        coords: np.ndarray, 
        surface_cfg: dict, 
        colors_per_ca: Optional[np.ndarray]
    ) -> Optional[list[SceneObject]]:
        if not self._surface_visible:
            return None

        surface_size_scale = float(surface_cfg.get("size_scale", 0.03))
        surface_min_size = float(surface_cfg.get("min_size", 2.5))
        surface_ao_radius = float(surface_cfg.get("ao_radius", 4.5))
        surface_ao_max = int(surface_cfg.get("max_neighbors", 24))
        surface_ao_strength = float(surface_cfg.get("ao_strength", 0.6))
        surface_alpha = float(surface_cfg.get("alpha", 0.8))
        surface_max_points = int(surface_cfg.get("max_points", 10000))
        surface_color_mode = str(surface_cfg.get("color_mode", "ao_gray")).lower()
        surface_base_color = np.asarray(
            surface_cfg.get("base_color", self._base_color_single), dtype=float
        )
        if surface_base_color.shape[0] != 4:
            surface_base_color = np.asarray(self._base_color_single, dtype=float)

        pts_surface = None
        colors_surface = None

        if self._all_atom_coords is not None:
            pts_surface = np.asarray(self._all_atom_coords, dtype=float)
        else:
            pts_surface = coords.copy()

        if pts_surface.size:
            # Base surface colors from residue mapping or uniform base color.
            if (
                surface_color_mode == "by_residue"
                and self._all_atom_res_ids is not None
                and self._residue_ids is not None
                and colors_per_ca is not None
                and len(colors_per_ca) == len(self._residue_ids)
            ):
                color_map = {rid: colors_per_ca[i_res] for i_res, rid in enumerate(self._residue_ids)}
                colors_surface = np.zeros(
                    (pts_surface.shape[0], 4), dtype=float
                )
                for i_atom, rid in enumerate(self._all_atom_res_ids):
                    colors_surface[i_atom, :] = color_map.get(
                        rid, self._base_color_single
                    )
            else:
                colors_surface = np.tile(
                    surface_base_color,
                    (pts_surface.shape[0], 1),
                )

            # Apply per-atom overrides if present.
            if (
                getattr(self, "_colors_per_atom_override", None) is not None
                and self._all_atom_res_ids is not None
                and len(self._colors_per_atom_override) == self._all_atom_res_ids.shape[0]
            ):
                ov = np.asarray(self._colors_per_atom_override, dtype=float)
                for i_atom in range(min(colors_surface.shape[0], ov.shape[0])):
                    col_ov = ov[i_atom]
                    if np.isfinite(col_ov).all():
                        colors_surface[i_atom, :] = col_ov

            max_pts = max(1, surface_max_points)
            if pts_surface.shape[0] > max_pts:
                step = max(1, pts_surface.shape[0] // max_pts)
                pts_surface = pts_surface[::step]
                colors_surface = colors_surface[::step]

            colors_surface = colors_surface.copy()

            # Apply a simple ambient-occlusion style darkening based on
            # local point density so that pockets and grooves appear
            # darker, similar to pyball-style surface shading.
            try:
                occ_surf = _estimate_ambient_occlusion(
                    pts_surface,
                    radius=surface_ao_radius,
                    max_neighbors=surface_ao_max,
                )
            except Exception:
                occ_surf = None
            if (
                occ_surf is not None
                and np.asarray(occ_surf).shape[0] == pts_surface.shape[0]
            ):
                occ_s = np.asarray(occ_surf, dtype=float)
                shade_surf = (1.0 - surface_ao_strength) + (
                    surface_ao_strength * (1.0 - occ_s)
                )
                colors_surface[:, :3] *= shade_surf.reshape(-1, 1)

            colors_surface = np.clip(colors_surface, 0.0, 1.0)
            colors_surface[:, 3] *= surface_alpha

            size_world = max(self._radius * surface_size_scale, surface_min_size)
            meta = {"size": float(size_world), "glyph": "sphere"}

            geom = Geometry(
                kind="points",
                positions=pts_surface,
                colors=colors_surface,
                meta=meta,
            )
            scene_objects = [SceneObject(id="surface", geometry=geom, render_mode="transparent")]

            return scene_objects

        return None

    def _update_dots(
        self,
        coords: np.ndarray,
        colors_per_ca: Optional[np.ndarray],
    ) -> Optional[list[SceneObject]]:
        if not self._show_dots:
            return None

        dots_cfg = _DISPLAY_CONFIG.get("dots", {})
        try:
            max_points = int(dots_cfg.get("max_points", 250_000))
        except Exception:
            max_points = 250_000
        try:
            size_px = float(dots_cfg.get("size_px", 8.0))
        except Exception:
            size_px = 8.0
        try:
            alpha = float(dots_cfg.get("alpha", 1.0))
        except Exception:
            alpha = 1.0
        try:
            px_mode = bool(dots_cfg.get("px_mode", True))
        except Exception:
            px_mode = True
        try:
            size_scale = float(dots_cfg.get("size_scale", 0.04))
        except Exception:
            size_scale = 0.04
        try:
            min_size = float(dots_cfg.get("min_size", 3.0))
        except Exception:
            min_size = 3.0

        positions: Optional[np.ndarray]
        colors_local: Optional[np.ndarray]

        if self._all_atom_coords is not None and self._all_atom_coords.size:
            positions = np.asarray(self._all_atom_coords, dtype=float)
            atom_res = np.asarray(self._all_atom_res_ids)
            base_col = np.asarray(self._base_color_single, dtype=float)
            colors_local = np.tile(base_col, (positions.shape[0], 1))

            if (
                self._residue_ids is not None
                and colors_per_ca is not None
                and len(colors_per_ca) == len(self._residue_ids)
            ):
                color_map = {rid: colors_per_ca[i_res] for i_res, rid in enumerate(self._residue_ids)}
                for i_atom, rid in enumerate(atom_res):
                    colors_local[i_atom, :] = color_map.get(rid, self._base_color_single)

            if (
                getattr(self, "_colors_per_atom_override", None) is not None
                and len(self._colors_per_atom_override) == atom_res.shape[0]
            ):
                ov = np.asarray(self._colors_per_atom_override, dtype=float)
                for i_atom in range(atom_res.shape[0]):
                    col_ov = ov[i_atom]
                    if np.isfinite(col_ov).all():
                        colors_local[i_atom, :] = col_ov
        else:
            positions = np.asarray(coords, dtype=float) if coords is not None else None
            colors_local = None if colors_per_ca is None else np.asarray(colors_per_ca, dtype=float)
        
        if positions is None or positions.size == 0:
            return None

        if max_points > 0 and positions.shape[0] > max_points:
            step = max(1, positions.shape[0] // max_points)
            positions = positions[::step]
            if colors_local is not None and colors_local.shape[0] >= positions.shape[0]:
                colors_local = colors_local[::step]
        
        if colors_local is None:
            base = np.asarray(dots_cfg.get("base_color", self._base_color_single), dtype=float)
            if base.shape[0] != 4:
                base = np.asarray(self._base_color_single, dtype=float)
            colors_local = np.tile(base, (positions.shape[0], 1))
        else:
            colors_local = np.asarray(colors_local, dtype=float)

        if colors_local.shape[0] != positions.shape[0]:
            colors_local = np.resize(colors_local, (positions.shape[0], colors_local.shape[1]))

        colors_local = colors_local.copy()
        colors_local[:, 3] = np.clip(colors_local[:, 3] * alpha, 0.0, 1.0)

        if px_mode:
            size_value = size_px
            meta = {"size": size_value, "px_mode": True, "glyph": "sphere"}
        else:
            size_world = max(self._radius * size_scale, min_size)
            size_world = max(size_world, 0.5)
            meta = {"size": size_world, "glyph": "sphere"}

        geom = Geometry(
            kind="points",
            positions=positions,
            colors=colors_local,
            meta=meta,
        )
        return [SceneObject(id="dots", geometry=geom, render_mode="opaque")]

    def _update_custom_overlays(self, surface_cfg: dict) -> Optional[list[SceneObject]]:
        overlays = getattr(self, "_point_overlays", None)
        if not overlays:
            return None

        surface_size_scale = float(surface_cfg.get("size_scale", 0.03))
        surface_min_size = float(surface_cfg.get("min_size", 2.5))

        scene_objects = []
        for overlay in list(overlays.values()):
            try:
                pts_ov = np.asarray(overlay.get("coords"), dtype=float)
            except Exception:
                continue
            if pts_ov.ndim != 2 or pts_ov.shape[0] == 0 or pts_ov.shape[1] != 3:
                continue

            try:
                col = overlay.get("color", None)
                if col is None:
                    base_col = np.asarray(self._base_color_single, dtype=float)
                    colors_ov = np.tile(base_col, (pts_ov.shape[0], 1))
                else:
                    arr_col = np.asarray(col, dtype=float)
                    if arr_col.ndim == 1 and arr_col.shape[0] == 4:
                        colors_ov = np.tile(arr_col, (pts_ov.shape[0], 1))
                    elif (
                        arr_col.ndim == 2
                        and arr_col.shape[1] == 4
                        and arr_col.shape[0] == pts_ov.shape[0]
                    ):
                        colors_ov = arr_col
                    else:
                        base_col = np.asarray(self._base_color_single, dtype=float)
                        colors_ov = np.tile(base_col, (pts_ov.shape[0], 1))
            except Exception:
                base_col = np.asarray(self._base_color_single, dtype=float)
                colors_ov = np.tile(base_col, (pts_ov.shape[0], 1))

            try:
                size_scale_ov = float(overlay.get("size_scale", surface_size_scale))
            except Exception:
                size_scale_ov = surface_size_scale
            try:
                min_size_ov = float(overlay.get("min_size", surface_min_size))
            except Exception:
                min_size_ov = surface_min_size
            try:
                alpha_ov = float(overlay.get("alpha", 1.0))
            except Exception:
                alpha_ov = 1.0
            try:
                px_mode_ov = bool(overlay.get("px_mode", False))
            except Exception:
                px_mode_ov = False

            size_ov = max(self._radius * size_scale_ov, min_size_ov)
            colors_ov = np.asarray(colors_ov, dtype=float).copy()
            if colors_ov.shape[1] >= 4:
                colors_ov[:, 3] *= alpha_ov

            geom = Geometry(kind="points", positions=pts_ov, colors=colors_ov)
            scene_objects.append(SceneObject(id="overlay", geometry=geom, render_mode="transparent"))

        return scene_objects

    def _update_measurements(self) -> list[SceneObject]:
        measurements = getattr(self, "_measurements", None)
        if not measurements:
            return []
            
        scene_objects = []
        for mid, mdata in measurements.items():
            kind = mdata.get("kind", "distance")
            coords = np.asarray(mdata.get("positions", []), dtype=float)
            if coords.size == 0: continue
            
            color = np.asarray(mdata.get("color", [1.0, 1.0, 1.0, 1.0]), dtype=float)
            label = str(mdata.get("label", ""))
            
            if kind == "distance" and coords.shape[0] >= 2:
                 # Line between two points
                 line_geom = Geometry(kind="line", positions=coords[:2], colors=np.tile(color, (2, 1)))
                 scene_objects.append(SceneObject(id=f"meas_line_{mid}", geometry=line_geom, render_mode="overlay"))
                 
                 # Label at midpoint
                 midpoint = np.mean(coords[:2], axis=0)
                 label_geom = Geometry(kind="text", positions=midpoint.reshape(1, 3), colors=color.reshape(1, 4), meta={"labels": [label]})
                 scene_objects.append(SceneObject(id=f"meas_text_{mid}", geometry=label_geom, render_mode="overlay"))
            
            elif kind == "angle" and coords.shape[0] >= 3:
                 # Lines 0-1, 1-2
                 line_coords = np.array([coords[0], coords[1], coords[1], coords[2]])
                 line_geom = Geometry(kind="line", positions=line_coords, colors=np.tile(color, (4, 1)))
                 scene_objects.append(SceneObject(id=f"meas_line_{mid}", geometry=line_geom, render_mode="overlay"))
                 
                 # Label at center point (1)
                 label_geom = Geometry(kind="text", positions=coords[1].reshape(1, 3), colors=color.reshape(1, 4), meta={"labels": [label]})
                 scene_objects.append(SceneObject(id=f"meas_text_{mid}", geometry=label_geom, render_mode="overlay"))

            elif kind == "dihedral" and coords.shape[0] >= 4:
                 # Lines 0-1, 1-2, 2-3
                 line_coords = np.array([coords[0], coords[1], coords[1], coords[2], coords[2], coords[3]])
                 line_geom = Geometry(kind="line", positions=line_coords, colors=np.tile(color, (6, 1)))
                 scene_objects.append(SceneObject(id=f"meas_line_{mid}", geometry=line_geom, render_mode="overlay"))
                 
                 # Label at midpoint of central bond (1-2)
                 midpoint = np.mean(coords[1:3], axis=0)
                 label_geom = Geometry(kind="text", positions=midpoint.reshape(1, 3), colors=color.reshape(1, 4), meta={"labels": [label]})
                 scene_objects.append(SceneObject(id=f"meas_text_{mid}", geometry=label_geom, render_mode="overlay"))

        return scene_objects

    def _update_selection_highlight(self, coords: np.ndarray) -> Optional[list[SceneObject]]:
        sel = getattr(self, "_selected_residues", None)
        if not sel or self._coords is None:
            return None

        try:
            idx_sel = np.asarray(list(sel), dtype=int)
        except Exception:
            idx_sel = np.zeros(0, dtype=int)
        n = self._coords.shape[0]
        if idx_sel.size and n > 0:
            idx_sel = idx_sel[(idx_sel >= 0) & (idx_sel < n)]
        if idx_sel.size:
            try:
                centers = coords[idx_sel]
            except Exception:
                centers = None
            if centers is not None and centers.size:
                try:
                    sel_cfg = _DISPLAY_CONFIG.get("selection", {})
                except Exception:
                    sel_cfg = {}
                try:
                    col = np.asarray(
                        sel_cfg.get("color", [1.0, 1.0, 0.0, 1.0]),
                        dtype=float,
                    )
                except Exception:
                    col = np.array([1.0, 1.0, 0.0, 1.0], dtype=float)
                if col.shape[0] != 4:
                    col = np.array([1.0, 1.0, 0.0, 1.0], dtype=float)
                size_scale = float(sel_cfg.get("size_scale", 0.08))
                min_size = float(sel_cfg.get("min_size", 6.0))
                px_mode = bool(sel_cfg.get("px_mode", False))
                size = max(self._radius * size_scale, min_size)
                color_arr = np.tile(col, (centers.shape[0], 1))

                if px_mode:
                    geom = Geometry(
                        kind="points",
                        positions=centers,
                        colors=color_arr,
                        meta={
                            "glyph": "sphere",
                            "radius": size * 0.1,
                            "size": size,
                            "px_mode": True,
                        },
                    )
                    scene_objects = [SceneObject(id="selection", geometry=geom, render_mode="overlay")]
                else:
                    template = _build_sphere_mesh(radius=1.0)
                    verts = template.get("vertices") if template else None
                    norms = template.get("normals") if template else None
                    faces = template.get("faces") if template else None
                    if (
                        verts is not None
                        and norms is not None
                        and faces is not None
                        and verts.size
                        and faces.size
                    ):
                        n_sel = centers.shape[0]
                        n_verts = verts.shape[0]
                        radius_ws = max(size * 0.5, 1e-3)
                        verts_scaled = verts[np.newaxis, :, :] * radius_ws
                        verts_translated = verts_scaled + centers[:, np.newaxis, :]
                        positions = verts_translated.reshape(-1, 3)

                        faces_rep = np.repeat(faces[np.newaxis, :, :], n_sel, axis=0)
                        idx_offsets = (
                            np.arange(n_sel, dtype=faces.dtype) * n_verts
                        )[:, np.newaxis, np.newaxis]
                        faces_rep = (faces_rep + idx_offsets).reshape(-1, 3)

                        normals = np.repeat(
                            norms[np.newaxis, :, :], n_sel, axis=0
                        ).reshape(-1, 3)
                        colors = np.repeat(
                            color_arr[:, np.newaxis, :], n_verts, axis=1
                        ).reshape(-1, 4)

                        geom = Geometry(
                            kind="mesh",
                            positions=positions,
                            indices=faces_rep,
                            normals=normals,
                            colors=colors,
                        )
                        scene_objects = [
                            SceneObject(id="selection", geometry=geom, render_mode="overlay")
                        ]

                return scene_objects

        return None

    def _update_view(self, fit_camera: bool = True) -> None:
        if self._renderer is None:
            return

        visible_entries = [
            entry
            for entry in self._objects.values()
            if entry.visible
            and entry.state.coords is not None
            and getattr(entry.state.coords, "size", 0) > 0
        ]

        if not visible_entries:
            self._clear_items()
            self._scene = None
            return

        self._clear_items()

        scene_objects: list[SceneObject] = []
        centers: list[np.ndarray] = []
        radii: list[float] = []

        for entry in visible_entries:
            with self._activate_object(entry.object_id):
                objects = self._build_scene_for_current_object(object_prefix=entry.object_id)
                if objects:
                    scene_objects.extend(objects)
                center = (
                    np.asarray(self._center, dtype=float)
                    if isinstance(self._center, np.ndarray)
                    else np.zeros(3, dtype=float)
                )
                centers.append(center)
                try:
                    radii.append(float(self._radius))
                except Exception:
                    radii.append(1.0)

        if not radii:
            self._scene = None
            return

        radius = max(radii)
        try:
            centers_arr = np.vstack(centers)
            center = centers_arr.mean(axis=0)
        except Exception:
            center = np.zeros(3, dtype=float)

        if fit_camera:
            self._fit_camera_to_radius(radius)

        self._scene = Scene(objects=scene_objects, center=center, radius=radius)
        try:
            self._renderer.set_scene(self._scene)
        except Exception:
            pass

    def _build_scene_for_current_object(self, object_prefix: Optional[str] = None) -> list[SceneObject]:
        if self._coords is None or self._coords.size == 0:
            return []

        coords = self._coords.copy()
        n_points = coords.shape[0]

        cartoon_cfg = _DISPLAY_CONFIG.get("cartoon", {})
        balls_cfg = _DISPLAY_CONFIG.get("balls", {})
        sticks_cfg = _DISPLAY_CONFIG.get("sticks", {})
        surface_cfg = _DISPLAY_CONFIG.get("surface", {})

        if (
            self._color_mode == "by_secondary_structure"
            and self._secondary_structure is not None
        ):
            self._colors_per_ca = _build_ss_color_array(
                self._secondary_structure, n_points
            )
        elif self._color_mode == "by_residue" and self._residue_names is not None:
            self._colors_per_ca = _build_residue_color_array(
                self._residue_names, n_points
            )
        elif self._color_mode == "by_sequence":
            self._colors_per_ca = _build_sequence_gradient_colors(n_points)
        elif self._color_mode == "by_element":
            elements = None
            if self._atoms is not None and "element" in self._atoms.dtype.names:
                elements = self._atoms["element"]
            self._colors_per_ca = _build_element_color_array(elements, n_points)
        elif self._color_mode == "by_chain":
            self._colors_per_ca = _build_chain_color_array(self._residue_chain_ids, n_points)
        elif self._color_mode == "spectrum":
            # PyMOL 'spectrum' usually defaults to b-factor or sequence.
            # For now, let's use sequence gradient if no values provided.
            # In a fuller impl, we'd check for B-factor data.
            self._colors_per_ca = _build_sequence_gradient_colors(n_points)
        else:
            base = np.asarray(self._base_color_single, dtype=float)
            self._colors_per_ca = np.tile(base, (n_points, 1))

        # Apply optional per-residue color overrides on top of the base colors.
        try:
            ov = getattr(self, "_colors_per_residue_override", None)
        except Exception:
            ov = None
        if ov is not None:
            try:
                ov_arr = np.asarray(ov, dtype=float)
            except Exception:
                ov_arr = None
            if (
                ov_arr is not None
                and ov_arr.ndim == 2
                and ov_arr.shape[0] == n_points
            ):
                base_cols = self._colors_per_ca
                if base_cols is None or base_cols.shape != ov_arr.shape:
                    base = np.asarray(self._base_color_single, dtype=float)
                    base_cols = np.tile(base, (n_points, 1))
                base_cols = np.asarray(base_cols, dtype=float)
                mask = np.all(np.isfinite(ov_arr), axis=1)
                if mask.any():
                    base_cols[mask] = ov_arr[mask]
                self._colors_per_ca = base_cols

        scene_objects: list[SceneObject] = []
        scene_objects += self._update_cartoon(coords, n_points, cartoon_cfg, self._colors_per_ca)
        scene_objects += self._update_trace(coords, self._colors_per_ca)
        scene_objects += self._update_atoms(coords, n_points, balls_cfg, self._colors_per_ca) or []
        if self._show_atom_gaussians:
            feature_cfg = _DISPLAY_CONFIG.get("atom_features", {})
            gaussian_cfg = feature_cfg.get("gaussian_covariances", {})
            if not isinstance(gaussian_cfg, dict):
                gaussian_cfg = {}
            scene_objects += self._update_atom_gaussians(gaussian_cfg) or []
        scene_objects += self._update_sticks(sticks_cfg, self._colors_per_ca) or []
        scene_objects += self._update_surface(coords, surface_cfg, self._colors_per_ca) or []
        scene_objects += self._update_dots(coords, self._colors_per_ca) or []
        scene_objects += self._update_custom_overlays(surface_cfg) or []
        scene_objects += self._update_measurements() or []
        scene_objects += self._update_restraints(self._get_active_state()) or []
        scene_objects += self._update_selection_highlight(coords) or []

        if object_prefix:
            for obj in scene_objects:
                obj.id = f"{object_prefix}:{obj.id}"

        return scene_objects

    def _fit_camera_to_radius(self, radius: float) -> None:
        if self._renderer is None:
            return
        self._renderer.fit_to_radius(radius)



