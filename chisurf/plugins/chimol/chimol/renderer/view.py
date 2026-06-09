from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Optional, Union

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from ..analysis.ss import assign_ss_c3_from_atoms
from ..colors import (
    _build_chain_color_array,
    _build_element_color_array,
    _build_residue_color_array,
    _build_sequence_gradient_colors,
    _build_ss_color_array,
    _three_to_one_array,
)
from ..config import _DISPLAY_CONFIG
from ..geometry import (
    _build_bond_pairs,
    _build_sphere_mesh,
    _build_stick_mesh,
    _build_trace_ups,
    _compute_center_radius,
    _estimate_ambient_occlusion,
    _extract_ca_trace,
    _generate_cartoon_tube_arrays,
    _generate_nucleic_cartoon_arrays,
    _generate_surface_mesh_from_density,
    _generate_surface_mesh_edt,
    _generate_surface_mesh_from_gaussians,
    _generate_trace_arrays,
)
from .base import Renderer
from .chimol_state import _MolViewObjectEntry, _MolViewObjectState, _StateField
from .qtgl import QtGLRenderer
from .scene import Geometry, Scene, SceneObject


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
    _metaballs_visible = _StateField("metaballs_visible")
    _point_overlays = _StateField("point_overlays")
    _ca_indices = _StateField("_ca_indices")
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
        bond_pairs: np.ndarray | None = None,
        *,
        object_id: str | None = None
    ) -> None:
        """Load full RMF data (hierarchy, trajectory, radii) into an object."""
        with self._activate_object(object_id):
            state = self._get_active_state()
            state.rmf_hierarchy = hierarchy
            state.frames = frames
            state.frames_raw = frames
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
                self._select_state_frame(state, 0)
                self._total_frames = max(self._total_frames, len(frames))
                n_points = int(np.asarray(frames).shape[1])
                state.cartoon_mask = np.zeros(n_points, dtype=bool)
                state.ball_mask = np.ones(n_points, dtype=bool)
                state.sticks_mask = np.ones(n_points, dtype=bool)

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
        name: str | None = None,
        source_path: str | None = None,
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

    def _ensure_active_entry(self, create_if_missing: bool = True) -> _MolViewObjectEntry | None:
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

    def get_active_object_id(self) -> str | None:
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
        features: dict[str, object] | None,
        *,
        meta: dict[str, dict] | None = None,
        object_id: str | None = None,
    ) -> None:
        """Attach arbitrary per-atom feature payloads to the active object."""

        def _sanitize_dict(data: dict[str, object] | None) -> dict[str, object]:
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

    def clear_atom_features(self, *, object_id: str | None = None) -> None:
        self.set_atom_features(None, object_id=object_id)

    def set_atom_colors(self, colors: np.ndarray | None) -> None:
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

    def set_residue_colors(self, colors: np.ndarray | None) -> None:
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

    def get_residue_positions(self, indices, *, object_id: str | None = None) -> np.ndarray:
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
        object_id: str | None = None,
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

    def add_point_overlay(
        self,
        key: str,
        coords: np.ndarray,
        color: np.ndarray | Sequence[float] = (0.0, 1.0, 0.5, 0.6),
        size_scale: float = 0.03,
        min_size: float = 2.5,
        alpha: float = 0.6,
    ) -> None:
        """Add or replace a named point-cloud overlay in the 3D view.

        The overlay is rendered as transparent spheres on top of the structure.
        Suitable for displaying AV point clouds, dye density distributions, or
        any set of 3D positions.

        Parameters
        ----------
        key : str
            Unique identifier for this overlay. Calling again with the same key
            replaces the existing overlay.
        coords : (N, 3) ndarray
            3D coordinates of the overlay points.
        color : (4,) array-like or (N, 4) array-like
            RGBA colour in [0, 1]. Broadcast scalar or per-point.
        size_scale : float
            Point size relative to the scene radius.
        min_size : float
            Minimum point size in world units.
        alpha : float
            Global alpha multiplier applied on top of the per-point alpha.
        """
        if self._point_overlays is None:
            self._point_overlays = {}
        self._point_overlays[key] = {
            "coords": coords,
            "color": color,
            "size_scale": size_scale,
            "min_size": min_size,
            "alpha": alpha,
        }
        self._update_view()

    def update_point_overlay(
        self,
        key: str,
        coords: np.ndarray,
        **kwargs,
    ) -> None:
        """Update the coordinates (and optionally style) of an existing overlay.

        If no overlay with *key* exists, behaves identically to
        :meth:`add_point_overlay`.
        """
        if self._point_overlays is None:
            self._point_overlays = {}
        if key not in self._point_overlays:
            self.add_point_overlay(key, coords, **kwargs)
        else:
            self._point_overlays[key]["coords"] = coords
            for k, v in kwargs.items():
                self._point_overlays[key][k] = v
            self._update_view()

    def remove_point_overlay(self, key: str) -> bool:
        """Remove a named point-cloud overlay.

        Returns True if the overlay existed and was removed.
        """
        if self._point_overlays is not None and key in self._point_overlays:
            del self._point_overlays[key]
            self._update_view()
            return True
        return False

    def clear_point_overlays(self) -> None:
        """Remove all point-cloud overlays from the view."""
        self._point_overlays = {}
        self._update_view()

    def add_sphere(
        self,
        center: np.ndarray,
        radius: float = 1.5,
        color: Sequence[float] = (1.0, 0.8, 0.2, 0.9),
        label: str | None = None,
        key: str | None = None,
    ) -> str:
        """Place a single sphere at *center* (e.g. an AV mean position or attachment point).

        Parameters
        ----------
        center : (3,) array-like
            Sphere centre in Å.
        radius : float
            Sphere radius in Å (default 1.5 — about a Cβ).
        color : (4,) array-like
            RGBA colour in [0, 1].
        label : str, optional
            Text label placed next to the sphere.
        key : str, optional
            Overlay key; auto-generated as ``'sphere_<n>'`` if not provided.

        Returns
        -------
        key : str
            The overlay key used, for subsequent removal.
        """
        if self._point_overlays is None:
            self._point_overlays = {}
        if key is None:
            n = len([k for k in self._point_overlays if k.startswith("sphere_")])
            key = f"sphere_{n}"

        coords = np.asarray(center, dtype=float).reshape(1, 3)
        self._point_overlays[key] = {
            "coords": coords,
            "color": color,
            "size_scale": 0.0,
            "min_size": 2 * radius,
            "alpha": color[3] if len(color) > 3 else 1.0,
            "glyph": "sphere",
        }
        if label is not None:
            self._point_overlays[key]["label"] = label
        self._update_view()
        return key

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

    def copy_object(self, object_id: str, *, name: str | None = None) -> str | None:
        """Create a deep copy of an existing loaded object."""
        entry = self._objects.get(object_id)
        if entry is None:
            return None
        copied = self._create_object(
            name=name or f"{entry.name}_copy",
            source_path=entry.source_path,
            placeholder=entry.placeholder,
        )
        copied.state = _copy_state(entry.state)
        copied.visible = bool(entry.visible)
        self._active_object_id = copied.id
        self._update_view()
        return copied.id

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
        # If frames have been attached without going through the public
        # timeline (e.g. via ``set_frames`` + ``set_active_frame``), derive
        # the timeline length from the active state so the spinbox/UI match.
        try:
            active_state = self._get_active_state()
            state_frames = getattr(active_state, "frames", None)
        except Exception:
            state_frames = None
        if state_frames is not None and getattr(state_frames, "ndim", 0) == 3:
            try:
                n_states = int(state_frames.shape[0])
            except Exception:
                n_states = 0
            if n_states > 0 and n_states > self._total_frames:
                self._total_frames = n_states
        new_idx = max(0, min(int(frame_idx), self._total_frames - 1))
        if new_idx == self._current_frame:
            return
        self._current_frame = new_idx
        self._apply_frame_states()
        self._update_view(fit_camera=False)

    def _select_state_frame(
        self,
        state: _MolViewObjectState,
        index: int,
    ) -> int:
        """Select one trajectory frame and expose it as render geometry.

        Chimol stores trajectories in ``state.frames`` but all render paths
        consume ``state.coords`` / ``state.all_atom_coords``. Keeping that
        derived state in one place prevents accidental rendering of the whole
        ``(T, N, 3)`` trajectory array.
        """
        frames = getattr(state, "frames", None)
        if frames is None:
            return 0
        try:
            arr = np.asarray(frames, dtype=float)
        except Exception:
            return 0
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] == 0:
            return 0

        n_frames = int(arr.shape[0])
        idx = max(0, min(int(index), n_frames - 1))
        frame = np.asarray(arr[idx], dtype=float)

        state.active_frame = idx
        # Coordinate-only trajectories should not leave stale all-atom data in
        # atom rendering paths. Reuse all-atom coords only when dimensions match.
        all_atom_coords = getattr(state, "all_atom_coords", None)
        frame_matches_all_atoms = False
        if all_atom_coords is None:
            state.all_atom_coords = frame
        else:
            try:
                all_atom_arr = np.asarray(all_atom_coords)
                if all_atom_arr.ndim == 2 and all_atom_arr.shape == frame.shape:
                    state.all_atom_coords = frame
                    frame_matches_all_atoms = True
                elif all_atom_arr.ndim == 3 and all_atom_arr.shape[1:] == frame.shape:
                    state.all_atom_coords = np.asarray(all_atom_arr[idx], dtype=float)
                    frame_matches_all_atoms = True
                else:
                    state.all_atom_coords = frame
            except Exception:
                state.all_atom_coords = frame

        if state.atoms is not None and not frame_matches_all_atoms:
            try:
                atoms_xyz = np.asarray(state.atoms["xyz"], dtype=float)
                if atoms_xyz.shape != frame.shape:
                    state.atoms = None
                    state.all_atom_res_ids = None
                    state.all_atom_radii = None
                    state.bond_pairs = None
                    if state.residue_ids is not None and len(state.residue_ids) != frame.shape[0]:
                        state.residue_ids = None
                        state.residue_names = None
                        state.residue_oneletter = None
                        state.residue_chain_ids = None
                        state.secondary_structure = None
                        state.trace_ups = None
            except Exception:
                state.atoms = None
                state.all_atom_res_ids = None
                state.all_atom_radii = None
                state.bond_pairs = None
                if state.residue_ids is not None and len(state.residue_ids) != frame.shape[0]:
                    state.residue_ids = None
                    state.residue_names = None
                    state.residue_oneletter = None
                    state.residue_chain_ids = None
                    state.secondary_structure = None
                    state.trace_ups = None

        # All-atom frames: set_frames already re-centered and scaled them.
        # Extract the CA trace so residue-level rendering (cartoon, trace,
        # labels) stays intact. Always replace state.coords; frames usually
        # have the same shape, so a shape-change guard would keep rendering
        # the first conformation while only all-atom overlays move.
        selected_coords = None
        if frame_matches_all_atoms and state.residue_ids is not None:
            try:
                ca_idx = getattr(state, "_ca_indices", None)
                if ca_idx is not None and len(ca_idx) > 0 and ca_idx.max() < frame.shape[0]:
                    ca_coords = frame[ca_idx]
                    if ca_coords.shape[0] == len(state.residue_ids):
                        selected_coords = ca_coords
            except Exception:
                pass
        if selected_coords is None:
            selected_coords = frame
        state.coords = selected_coords

        # Sync atoms["xyz"] to the raw (unscaled) frame coordinates so that
        # _build_trace_ups uses the current backbone geometry rather than the
        # stale coordinates from the initial add_structure call.  Without this,
        # the ribbon normals (C→O vectors) are frozen at the starting
        # conformation and the cartoon appears twisted/tangled as the MC moves
        # the backbone.
        if (
            frame_matches_all_atoms
            and state.atoms is not None
            and state.residue_ids is not None
        ):
            try:
                frames_raw = getattr(state, "frames_raw", None)
                if frames_raw is not None:
                    raw_arr = np.asarray(frames_raw, dtype=float)
                    if raw_arr.ndim == 3 and raw_arr.shape[0] > idx and raw_arr.shape[1:] == frame.shape:
                        raw_frame = raw_arr[idx]
                    else:
                        raw_frame = None
                else:
                    raw_frame = None
                if raw_frame is not None and raw_frame.shape == state.atoms["xyz"].shape:
                    state.atoms = state.atoms.copy()
                    state.atoms["xyz"] = raw_frame
                    state.trace_ups = _build_trace_ups(
                        state.atoms,
                        state.residue_ids,
                        selected_coords,
                        state.residue_chain_ids,
                    )
            except Exception:
                pass

        try:
            center, radius = _compute_center_radius(
                state.all_atom_coords
                if frame_matches_all_atoms and state.all_atom_coords is not None
                else state.coords
            )
            state.center = center
            state.radius = float(radius)
        except Exception:
            pass

        coords_len = state.coords.shape[0] if state.coords is not None else frame.shape[0]
        all_atom_len = (
            np.asarray(state.all_atom_coords).shape[0]
            if state.all_atom_coords is not None
            else coords_len
        )

        if state.cartoon_mask is None or len(state.cartoon_mask) != coords_len:
            state.cartoon_mask = np.ones(coords_len, dtype=bool)
        if state.ball_mask is None or len(state.ball_mask) not in (coords_len, all_atom_len):
            state.ball_mask = np.zeros(coords_len, dtype=bool)
        return idx

    def _apply_frame_states(self) -> None:
        """Update scene objects based on the current frame index."""
        # For objects with 'frames' coordinate sets, update their active_frame
        for entry in self._objects.values():
            state = entry.state
            if state.frames is not None and state.frames.ndim == 3:
                # If the object has frames, map the global timeline to its states.
                # Simplest mapping: state_idx = global_idx % n_states
                n_states = state.frames.shape[0]
                self._select_state_frame(state, self._current_frame % n_states)
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
    def get_chain_ids(self, object_id: str | None = None) -> list[str]:
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

    def split_chains(self, *, prefix: str | None = None, object_ids: list[str] | None = None) -> int:
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
        name: str | None = None,
        source_path: str | None = None,
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
        name: str | None = None,
        source_path: str | None = None,
    ) -> str:
        entry = self._create_object(name=name, source_path=source_path)
        self.set_coordinates(coords)
        return entry.object_id

    @contextmanager
    def _activate_object(self, object_id: str | None):
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
        parent: QtWidgets.QWidget | None = None,
        background: str | None = None,
        *,
        representation_mode: str | None = None,
        show_cartoon: bool | None = None,
        show_trace: bool | None = None,
        show_atoms: bool | None = None,
        show_sticks: bool | None = None,
        sidechains_visible: bool | None = None,
        grid_visible: bool | None = None,
        surface_visible: bool | None = None,
        scale_factor: float | None = None,
    ) -> None:
        super().__init__(parent)

        self._objects: OrderedDict[str, _MolViewObjectEntry] = OrderedDict()
        self._active_object_id: str | None = None
        self._object_counter: int = 0
        # Allow creating an initial entry during startup; turned off when last object is deleted.
        self._auto_create_enabled: bool = True

        # Animation / Timeline state
        self._total_frames: int = 1
        self._current_frame: int = 0  # 0-indexed internally
        self._keyframes: dict[int, dict] = {}
        self._animation_running: bool = False
        self._animation_timer: QtCore.QTimer | None = None
        self.selection_mode: str = "Residues"

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.view: QtWidgets.QWidget | None = None
        self._disabled_label: QtWidgets.QLabel | None = None
        self._container: QtWidgets.QWidget | None = None
        self._ray_overlay: QtWidgets.QLabel | None = None

        # Stored geometry
        self._coords: np.ndarray | None = None
        self._center: np.ndarray | None = None
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
            representation_mode or rep_mode_default
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
        self._info_overlay: QtWidgets.QPlainTextEdit | None = None

        # Render backend
        self._renderer: Renderer | None = None
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
        self._scene: Scene | None = None

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
            self._container = container
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
          ``'atom_name'`` (as in :mod:`chisurf.core.structure`), or
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

            coords_all_raw: np.ndarray | None
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
            if getattr(self, "_ca_indices", None) is None:
                try:
                    _atom_names = np.asarray(atoms["atom_name"], dtype=str)
                    self._ca_indices = np.where(
                        np.char.strip(_atom_names) == "CA"
                    )[0]
                except Exception:
                    self._ca_indices = None
            self._trace_ups = _build_trace_ups(atoms, self._residue_ids, self._coords, chain_ids)

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
        self._all_atom_res_ids = None
        self._all_atom_radii = None
        self._atom_features = {}
        self._atom_feature_meta = {}
        self._show_atom_gaussians = False

        center, radius = _compute_center_radius(arr)
        scale = float(self._scale_factor)
        arr = (arr - center) * scale

        self._all_atom_coords = arr
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

    def set_frames(
        self,
        frames: np.ndarray,
        *,
        object_id: str | None = None,
        active_frame: int | None = None,
    ) -> None:
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
            state.frames_raw = arr
            idx = 0 if active_frame is None else int(active_frame)
            idx = self._select_state_frame(state, idx)
            self._center = np.zeros(3, dtype=float)
            self._radius = float(radius * scale)
            self._selected_residues = []
            # Keep the global timeline in sync with the new trajectory so
            # ``set_current_frame`` / ``get_total_frames`` reflect reality.
            self._total_frames = max(self._total_frames, int(arr.shape[0]))
            self._current_frame = idx
            if self._current_frame >= self._total_frames:
                self._current_frame = self._total_frames - 1
            try:
                self._update_view()
            except Exception:
                pass

    def append_frame(self, frame: np.ndarray, *, object_id: str | None = None) -> int:
        """Append one raw coordinate frame to an object's trajectory.

        Parameters
        ----------
        frame : np.ndarray
            Coordinate array with shape ``(N, 3)`` in Angstrom.
        object_id : str, optional
            Object to update. The active object is used by default.

        Returns
        -------
        int
            Number of frames after appending.
        """
        arr = np.asarray(frame, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("frame must have shape (N, 3)")
        with self._activate_object(object_id):
            state = self._get_active_state()
            raw = getattr(state, "frames_raw", None)
            if raw is None:
                frames = arr[np.newaxis, :, :]
            else:
                raw_arr = np.asarray(raw, dtype=float)
                if raw_arr.ndim != 3 or raw_arr.shape[1:] != arr.shape:
                    raise ValueError("frame shape does not match existing trajectory")
                frames = np.concatenate([raw_arr, arr[np.newaxis, :, :]], axis=0)
        self.set_frames(frames, object_id=object_id)
        self.set_active_frame(frames.shape[0] - 1, object_id=object_id)
        return int(frames.shape[0])

    def set_active_frame(self, index: int, *, object_id: str | None = None) -> None:
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
            self._select_state_frame(state, idx)
            # Keep ``_total_frames`` consistent with the trajectory length
            # so the public ``get_total_frames`` / ``set_current_frame`` API
            # behaves correctly for externally attached frames.
            self._total_frames = max(self._total_frames, n_frames)
            self._current_frame = idx
            if self._current_frame >= self._total_frames:
                self._current_frame = self._total_frames - 1
            try:
                self._update_view(fit_camera=False)
            except Exception:
                pass

    def get_frame_count(self, object_id: str | None = None) -> int:
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

    def get_active_frame_index(self, object_id: str | None = None) -> int:
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

    def get_view_state(self) -> list[float]:
        """Return an 18-float view tuple for PyMOL-style ``get_view``."""
        renderer = self._renderer
        if renderer is not None and hasattr(renderer, "get_view_state"):
            return list(renderer.get_view_state())
        return [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            float(self._radius * 3.0), 20.0, 45.0,
            0.0, 0.0, 0.0,
            float(self._camera_near_clip), float(self._camera_far_clip), 45.0,
        ]

    def set_view_state(self, view) -> None:
        """Restore an 18-float view tuple from PyMOL-style ``set_view``."""
        vals = [float(v) for v in view]
        if len(vals) != 18:
            raise ValueError("view must contain 18 floats")
        renderer = self._renderer
        if renderer is not None and hasattr(renderer, "set_view_state"):
            renderer.set_view_state(vals)

    def center(self, indices: Sequence[int] | None = None, *, object_id: str | None = None) -> None:
        """Center camera on the geometric center of target residues."""
        coords = self.get_residue_positions(indices, object_id=object_id)
        if coords.size == 0 or self._renderer is None:
            return
        center = coords.mean(axis=0)
        self._renderer.look_at(center)

    def zoom(self, indices: Sequence[int] | None = None, *, buffer: float = 2.0, object_id: str | None = None) -> None:
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

    def orient(self, indices: Sequence[int] | None = None, *, object_id: str | None = None) -> None:
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
        cartoon: bool | None = None,
        ball: bool | None = None,
        *,
        object_id: str | None = None,
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

    def set_metaballs_visible(self, visible: bool) -> None:
        self._metaballs_visible = bool(visible)
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

    def set_representation(self, mode: str, *, object_id: str | None = None) -> None:
        """Legacy mode-style API (cartoon / ca_trace / atoms).

        This is primarily used by keyboard shortcuts and :class:`MolViewPlot`.
        Internally it configures the independent representation toggles
        (``_show_cartoon``, ``_show_trace``, ``_show_atoms``) and the
        per-residue ball mask, then refreshes the view.
        """
        mode_l = str(mode).lower()
        if mode_l not in ("cartoon", "ca_trace", "atoms"):
            return
        with self._activate_object(object_id):
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
        if visible and self._all_atom_coords is not None:
            n_atoms = int(np.asarray(self._all_atom_coords).shape[0])
            if (
                self._sticks_mask is None
                or len(self._sticks_mask) != n_atoms
                or not np.asarray(self._sticks_mask, dtype=bool).any()
            ):
                self._sticks_mask = np.ones(n_atoms, dtype=bool)
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

    def _show_disabled_label(self, reason: str | None = None) -> None:
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

    def set_selected_residues(self, indices, *, object_id: str | None = None) -> None:
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
        self, object_id: str | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._activate_object(object_id):
            seq = _copy_array(self._residue_oneletter)
            res = _copy_array(self._residue_names)
        return seq, res

    def get_residue_numbers(self, object_id: str | None = None) -> np.ndarray | None:
        with self._activate_object(object_id):
            ids = _copy_array(self._residue_ids)
        return ids

    def get_residue_colors(
        self, object_id: str | None = None
    ) -> np.ndarray | None:
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
    def _has_nucleic_acids(self, residue_names: np.ndarray | None = None) -> bool:
        """Check if the current structure contains nucleic acid residues."""
        if residue_names is None:
            residue_names = getattr(self, "_residue_names", None)
        if residue_names is None:
            return False
        try:
            names_arr = np.asarray(residue_names, dtype=str)
            names_upper = np.char.upper(names_arr)
            # Check for common nucleic acid residue names
            nucleic_names = {"DA", "DC", "DG", "DT", "A", "C", "G", "T", "U"}
            return bool(nucleic_names.intersection(set(names_upper)))
        except Exception:
            return False

    def _clear_items(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.clear()
            except Exception:
                pass

    def _update_cartoon(
        self, coords: np.ndarray, n_points: int, config: dict, colors: np.ndarray | None
    ) -> list[SceneObject]:
        scene_objects: list[SceneObject] = []
        if not self._show_cartoon:
            return scene_objects

        ao_radius = float(config.get("ao_radius", 4.0))
        ao_max = int(config.get("ao_max_neighbors", 16))
        ao_strength = float(config.get("ao_strength", 0.45))

        # Check for nucleic acid residues to exclude them from the regular cartoon path
        nucleic_names = {
            "DA", "DC", "DG", "DT", "A", "C", "G", "T", "U",
            "2DA", "2DC", "2DG", "2DT",
            "RA", "RC", "RG", "RU", "I",
            "5MC", "5HC", "OMC", "H2U", "PSU", "M2G", "1MA", "7MG",
            "D2A", "D2C", "D2G", "D2T", "R2A", "R2C", "R2G", "R2U",
        }
        is_nuc_residue = np.zeros(n_points, dtype=bool)
        if self._residue_names is not None and len(self._residue_names) == n_points:
            for i, rname in enumerate(self._residue_names):
                try:
                    rname_str = str(rname).strip().upper()
                except Exception:
                    rname_str = ""
                if rname_str in nucleic_names:
                    is_nuc_residue[i] = True

        non_nuc_mask = ~is_nuc_residue

        coords_cartoon = coords
        colors_for_tube = colors
        idx_all = np.arange(n_points, dtype=int)
        idx_cartoon = idx_all

        # Apply per-residue cartoon mask by subselecting the CA points
        # used to build the tube, and exclude nucleic residues.
        if self._cartoon_mask is not None and len(self._cartoon_mask) == n_points:
            mask = self._cartoon_mask.astype(bool) & non_nuc_mask
        else:
            mask = non_nuc_mask

        coords_cartoon = coords[mask]
        idx_cartoon = idx_all[mask]
        if colors_for_tube is not None:
            colors_for_tube = colors_for_tube[mask]

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
                    base_radius=float(config.get("tube_radius", 0.5)),
                    style=str(config.get("style", "tube")),
                    ss_codes=seg_ss,
                    config={**config, "coordinate_scale": float(self._scale_factor)},
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

        # Generate nucleic cartoon for DNA/RNA structures
        if self._has_nucleic_acids() and self._atoms is not None and self._all_atom_coords is not None:
            nuc_arrays = _generate_nucleic_cartoon_arrays(
                self._atoms,
                self._all_atom_coords,
                getattr(self, "_residue_ids", None),
                getattr(self, "_residue_chain_ids", None),
                colors,
                config={**config, "coordinate_scale": float(self._scale_factor)},
            )
            if nuc_arrays is not None:
                nuc_verts, nuc_norms, nuc_faces, nuc_cols = nuc_arrays
                nuc_geom = Geometry(
                    kind="mesh",
                    positions=nuc_verts,
                    indices=nuc_faces,
                    normals=nuc_norms,
                    colors=nuc_cols,
                )
                scene_objects.append(
                    SceneObject(id="cartoon_nucleic", geometry=nuc_geom, render_mode="opaque")
                )

        return scene_objects

    def _update_trace(self, coords: np.ndarray, colors: np.ndarray | None) -> list[SceneObject]:
        scene_objects: list[SceneObject] = []
        if not self._show_trace:
            return scene_objects

        res_ids = getattr(self, "_residue_ids", None)
        chain_ids = getattr(self, "_residue_chain_ids", None)

        # Build segment boundaries at chain/residue-number breaks
        n = coords.shape[0]
        seg_bounds = [(0, n)]
        if n >= 2 and (res_ids is not None or chain_ids is not None):
            seg_bounds = []
            start = 0
            for i in range(n - 1):
                gap = False
                if chain_ids is not None and i < len(chain_ids) and (i + 1) < len(chain_ids):
                    ch0 = str(chain_ids[i]).strip()
                    ch1 = str(chain_ids[i + 1]).strip()
                    if ch0 != ch1:
                        gap = True
                if not gap and res_ids is not None and i < len(res_ids) and (i + 1) < len(res_ids):
                    try:
                        r0 = int(res_ids[i])
                        r1 = int(res_ids[i + 1])
                        if (r1 - r0) != 1:
                            gap = True
                    except Exception:
                        pass
                if gap:
                    if i + 1 - start >= 2:
                        seg_bounds.append((start, i + 1))
                    start = i + 1
            if n - start >= 2:
                seg_bounds.append((start, n))
            if not seg_bounds and n >= 2:
                seg_bounds = [(0, n)]

        for s, e in seg_bounds:
            seg_coords = coords[s:e]
            seg_colors = colors[s:e] if colors is not None and s < len(colors) else None
            coords_line, colors_line = _generate_trace_arrays(seg_coords, seg_colors)
            if colors_line is None:
                colors_line = seg_colors
            geom = Geometry(kind="line", positions=coords_line, colors=colors_line)
            scene_objects.append(SceneObject(id="trace", geometry=geom, render_mode="opaque"))

        return scene_objects

    def _update_atoms(
        self,
        coords: np.ndarray,
        n_points: int,
        balls_cfg: dict,
        colors_per_ca: np.ndarray | None
    ) -> list[SceneObject] | None:
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
                    radii_sel: np.ndarray | None
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

    def get_atom_sphere_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract all atom positions, colors (RGB), and radii for ray tracing.

        Works regardless of the current representation mode (cartoon, sticks,
        etc.) — always returns ALL atoms in the active object.  Radii are
        estimated from the ``radius`` field or the bounding sphere when per-atom
        radii are unavailable.

        Returns three arrays suitable for building a list of :class:`Sphere`:
        (positions, colors, radii) where positions is (N, 3), colors (N, 3),
        and radii (N,) — all in world / model coordinates.  If no atom data
        is available, returns empty arrays.
        """
        positions = np.zeros((0, 3), dtype=float)
        colors_rgb = np.zeros((0, 3), dtype=float)
        radii_arr = np.zeros((0,), dtype=float)

        balls_cfg = _DISPLAY_CONFIG.get("balls", {})
        balls_size_scale = float(balls_cfg.get("size_scale", 0.04))
        balls_min_size = float(balls_cfg.get("min_size", 3.0))
        balls_radius_multiplier = float(balls_cfg.get("radius_multiplier", 1.0))
        base_global_radius = max(self._radius * balls_size_scale, balls_min_size)

        atoms = self._atoms
        atom_xyz = self._all_atom_coords
        if atom_xyz is None and atoms is not None and "xyz" in (atoms.dtype.fields or {}):
            atom_xyz = np.asarray(atoms["xyz"], dtype=float)
        if atom_xyz is None:
            return positions, colors_rgb, radii_arr
        pts = np.asarray(atom_xyz, dtype=float)

        n_atoms_total = pts.shape[0]

        # --- Radii ---
        if (
            self._all_atom_radii is not None
            and len(self._all_atom_radii) == n_atoms_total
        ):
            radii_arr = np.asarray(self._all_atom_radii, dtype=float)
            radii_arr = radii_arr * balls_radius_multiplier
        else:
            radii_arr = np.full(n_atoms_total, base_global_radius, dtype=float)
        radii_arr[np.isnan(radii_arr) | (radii_arr <= 0.0)] = base_global_radius

        # --- Colors ---
        colors_4 = np.zeros((n_atoms_total, 4), dtype=float)
        colors_4[:] = np.asarray(self._base_color_single, dtype=float)

        if atoms is not None:
            n_points = len(self._residue_ids) if self._residue_ids is not None else 0
            if (
                self._colors_per_ca is not None
                and len(self._colors_per_ca) == n_points
            ):
                atom_res_id = None
                if "res_id" in (atoms.dtype.fields or {}):
                    try:
                        atom_res_id = np.asarray(atoms["res_id"])
                    except Exception:
                        atom_res_id = None
                if atom_res_id is not None:
                    for i_atom in range(n_atoms_total):
                        rid = atom_res_id[i_atom]
                        idx_in_res = np.where(self._residue_ids == rid)[0]
                        if len(idx_in_res) > 0:
                            colors_4[i_atom, :3] = self._colors_per_ca[idx_in_res[0], :3]
                        else:
                            colors_4[i_atom, :3] = self._base_color_single[:3]

            # Per-atom overrides
            ov = getattr(self, "_colors_per_atom_override", None)
            if ov is not None and len(ov) == n_atoms_total:
                ov_arr = np.asarray(ov, dtype=float)
                for i_atom in range(n_atoms_total):
                    if np.isfinite(ov_arr[i_atom]).all():
                        colors_4[i_atom, :3] = ov_arr[i_atom, :3]

        colors_rgb = np.clip(colors_4[:, :3], 0.0, 1.0)

        return pts, colors_rgb, radii_arr

    def get_ray_view_state(self) -> list[float]:
        """Return the current 18-float view tuple for ray-traced camera setup."""
        renderer = getattr(self, "_renderer", None)
        if renderer is not None and hasattr(renderer, "get_view_state"):
            return renderer.get_view_state()
        return [1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                50.0, 20.0, 45.0,
                0.0, 0.0, 0.0,
                0.1, 100.0, 45.0]

    def get_current_scene(self) -> Scene | None:
        """Return the currently visible scene description.

        Returns
        -------
        Scene or None
            Current backend-neutral scene object used by the OpenGL renderer.
        """
        return self._scene

    def grab_current_view_image(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> QtGui.QImage | None:
        """Grab the currently visible OpenGL view as a QImage.

        Parameters
        ----------
        width, height:
            Optional output dimensions. When both are provided, the grabbed
            image is scaled to that exact size.

        Returns
        -------
        QtGui.QImage or None
            Snapshot of the current viewport, or ``None`` if unavailable.
        """
        renderer = getattr(self, "_renderer", None)
        widget = renderer.widget() if renderer is not None and hasattr(renderer, "widget") else None
        grab = getattr(widget, "grabFramebuffer", None)
        if not callable(grab):
            return None
        try:
            image = grab()
        except Exception:
            return None
        if image is None or image.isNull():
            return None
        if width and height and width > 0 and height > 0 and (image.width() != width or image.height() != height):
            image = image.scaled(
                int(width),
                int(height),
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        return image

    def show_ray_overlay(self, image: QtGui.QImage) -> bool:
        """Show a rendered image over the OpenGL viewport until interaction.

        Parameters
        ----------
        image:
            Rendered image to display.

        Returns
        -------
        bool
            ``True`` when the overlay was shown.
        """
        container = getattr(self, "_container", None)
        if container is None or image is None or image.isNull():
            return False
        overlay = self._ray_overlay
        if overlay is None:
            overlay = QtWidgets.QLabel(container)
            overlay.setAlignment(QtCore.Qt.AlignCenter)
            overlay.setScaledContents(True)
            overlay.setStyleSheet("background-color: black;")
            overlay.installEventFilter(self)
            self._ray_overlay = overlay
            layout = container.layout()
            if layout is not None:
                layout.addWidget(overlay, 0, 0)
        overlay.setPixmap(QtGui.QPixmap.fromImage(image))
        overlay.show()
        overlay.raise_()
        return True

    def hide_ray_overlay(self) -> None:
        """Hide the ray-rendered overlay if visible."""
        overlay = self._ray_overlay
        if overlay is not None:
            overlay.hide()

    def eventFilter(self, obj, event):  # type: ignore[override]
        """Dismiss ray overlay on user interaction."""
        if obj is self._ray_overlay and event is not None:
            if event.type() in (
                QtCore.QEvent.MouseButtonPress,
                QtCore.QEvent.MouseButtonDblClick,
                QtCore.QEvent.Wheel,
                QtCore.QEvent.KeyPress,
            ):
                self.hide_ray_overlay()
                return True
        return super().eventFilter(obj, event)

    def _update_atom_gaussians(self, config: dict) -> list[SceneObject] | None:
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

    def _update_sticks(self, sticks_cfg: dict, colors_per_ca: np.ndarray | None) -> list[SceneObject] | None:
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
                atom_res = None
                if self._all_atom_res_ids is not None:
                    atom_res = np.asarray(self._all_atom_res_ids)
                atom_colors = np.tile(
                    np.asarray(self._base_color_single, dtype=float),
                    (pts_all.shape[0], 1),
                )

                if (
                    atom_res is not None
                    and self._residue_ids is not None
                    and colors_per_ca is not None
                    and len(colors_per_ca) == len(self._residue_ids)
                ):
                    color_map = {rid: colors_per_ca[i_res] for i_res, rid in enumerate(self._residue_ids)}
                    for i_atom, rid in enumerate(atom_res):
                        atom_colors[i_atom, :] = color_map.get(
                            rid, self._base_color_single
                        )

                if (
                    atom_res is not None
                    and getattr(self, "_colors_per_atom_override", None) is not None
                    and len(self._colors_per_atom_override) == atom_res.shape[0]
                ):
                    ov = np.asarray(self._colors_per_atom_override, dtype=float)
                    for i_atom in range(atom_res.shape[0]):
                        col_ov = ov[i_atom]
                        if np.isfinite(col_ov).all():
                            atom_colors[i_atom, :] = col_ov

                # Use cylinder mesh for sticks (replaces GL_LINES)
                sticks_radius = float(sticks_cfg.get("radius", 0.15)) * float(self._scale_factor)
                sticks_segments = int(sticks_cfg.get("segments_circle", 12))
                mesh = _build_stick_mesh(
                    bonds, pts_all, atom_colors,
                    radius=sticks_radius, segments_circle=sticks_segments,
                )
                if mesh is not None:
                    verts, norms, faces, cols = mesh
                    geom = Geometry(
                        kind="mesh", positions=verts, indices=faces,
                        normals=norms, colors=cols,
                    )
                    return [SceneObject(id="sticks", geometry=geom, render_mode="opaque")]

        return None

    def _update_restraints(self, state: _MolViewObjectState) -> list[SceneObject] | None:
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

    def _update_metaballs(
        self,
        coords: np.ndarray,
        cfg: dict,
        colors_per_ca: np.ndarray | None,
    ) -> list[SceneObject] | None:
        if not getattr(self, "_metaballs_visible", False):
            return None

        iso_value = float(cfg.get("iso_value", 0.15))
        grid_spacing = float(cfg.get("grid_spacing", 0.6))
        padding = float(cfg.get("padding", 4.0))
        max_dim = int(cfg.get("max_dim", 128))
        alpha = float(cfg.get("alpha", 0.6))

        ao_strength = float(cfg.get("ao_strength", 0.5))
        ao_radius = float(cfg.get("ao_radius", 4.5))

        lighting_cfg = _DISPLAY_CONFIG.get("lighting", {})
        # Material properties for the jelly look
        material = {
            "shininess": float(cfg.get("shininess", lighting_cfg.get("shininess", 38.0))),
            "specular_strength": float(cfg.get("specular_strength", lighting_cfg.get("specular_strength", 0.18))),
            "rim_strength": float(cfg.get("rim_strength", lighting_cfg.get("rim_strength", 0.18))),
            "rim_power": float(cfg.get("rim_power", lighting_cfg.get("rim_power", 2.4))),
        }

        pts_surface = None
        colors_surface = None

        if self._all_atom_coords is not None:
            pts_surface = np.asarray(self._all_atom_coords, dtype=float)
        else:
            pts_surface = coords.copy()

        if pts_surface.size == 0:
            return None

        n_pts = pts_surface.shape[0]

        # Use an array of sigmas. Default roughly 1.5, or use atom radii if available
        if self._all_atom_radii is not None and self._all_atom_radii.shape[0] == n_pts:
            sigmas = np.asarray(self._all_atom_radii, dtype=float) * 1.5
        else:
            sigmas = np.ones(n_pts, dtype=float) * 1.5

        field_function = cfg.get("field_function", "wyvill")
        mesh_data = _generate_surface_mesh_from_density(
            pts_surface,
            sigmas,
            grid_spacing=grid_spacing,
            padding=padding,
            iso_value=iso_value,
            max_dim=max_dim,
            field_function=field_function,
        )

        if mesh_data is None:
            return None

        verts, faces, norms = mesh_data

        # Color the mesh
        base_color = np.asarray(self._base_color_single, dtype=float)
        if base_color.shape[0] != 4:
            base_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

        # Color the mesh with weighted blending
        mesh_colors = np.zeros((verts.shape[0], 4), dtype=float)

        # Determine atom colors
        n_pts = pts_surface.shape[0]
        atom_colors = np.tile(base_color, (n_pts, 1))

        if (
            self._all_atom_res_ids is not None
            and self._residue_ids is not None
            and colors_per_ca is not None
            and len(colors_per_ca) == len(self._residue_ids)
        ):
            # Create a robust mapping from residue ID to color
            res_id_to_color = {}
            for i_res, rid in enumerate(self._residue_ids):
                res_id_to_color[rid] = colors_per_ca[i_res]

            for i_atom, rid in enumerate(self._all_atom_res_ids):
                if rid in res_id_to_color:
                    atom_colors[i_atom] = res_id_to_color[rid]

        if getattr(self, "_colors_per_atom_override", None) is not None:
            ov = np.asarray(self._colors_per_atom_override, dtype=float)
            for i_atom in range(min(n_pts, ov.shape[0])):
                if np.isfinite(ov[i_atom]).all():
                    atom_colors[i_atom] = ov[i_atom]

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts_surface)

            # Find atoms contributing to each vertex
            max_sigma = float(np.max(sigmas))
            cutoff = max_sigma * 2.5

            # query_ball_point can be slow for very large systems, but for
            # typical proteins it provides much nicer blending.
            indices = tree.query_ball_point(verts, r=cutoff)

            for i_v, atom_indices in enumerate(indices):
                if not atom_indices:
                    # Fallback to nearest
                    _, nearest = tree.query(verts[i_v])
                    mesh_colors[i_v] = atom_colors[nearest]
                    continue

                v_pos = verts[i_v]
                w_sum = 0.0
                c_sum = np.zeros(4, dtype=float)

                for i_a in atom_indices:
                    d2 = np.sum((v_pos - pts_surface[i_a])**2)
                    s2 = sigmas[i_a]**2
                    w = math.exp(-d2 / (2.0 * s2))
                    c_sum += atom_colors[i_a] * w
                    w_sum += w

                if w_sum > 0:
                    mesh_colors[i_v] = c_sum / w_sum
                else:
                    _, nearest = tree.query(v_pos)
                    mesh_colors[i_v] = atom_colors[nearest]

            # Recalculate normals analytically for buttery smoothness
            # Normal = -Gradient(Density). Gradient of exp(-d2/2s2) is -(d/s2)*exp(-d2/2s2)
            # So Normal(v) is proportional to sum_i [ (v - atom_pos_i) / sigma_i^2 * weight_i ]
            new_norms = np.zeros_like(verts)
            for i_v, atom_indices in enumerate(indices):
                if not atom_indices:
                    continue
                v_pos = verts[i_v]
                grad = np.zeros(3, dtype=float)
                for i_a in atom_indices:
                    diff = v_pos - pts_surface[i_a]
                    d2 = np.sum(diff**2)
                    s2 = sigmas[i_a]**2
                    w = math.exp(-d2 / (2.0 * s2))
                    grad += (diff / s2) * w

                mag = np.linalg.norm(grad)
                if mag > 1e-6:
                    new_norms[i_v] = grad / mag
                else:
                    new_norms[i_v] = norms[i_v] # Fallback

            norms = new_norms

            # Estimate Ambient Occlusion for depth
            if ao_strength > 0:
                occ = _estimate_ambient_occlusion(verts, radius=ao_radius, max_neighbors=32)
                if occ is not None:
                    darken = 1.0 - (occ * ao_strength)
                    mesh_colors[:, :3] *= darken[:, np.newaxis]

        except Exception:
            # Absolute fallback to nearest-neighbor if advanced blending fails
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(pts_surface)
                _, nearest = tree.query(verts)
                mesh_colors = atom_colors[nearest]
            except Exception:
                mesh_colors[:, :] = base_color

        render_mode = "opaque"
        if alpha < 1.0:
            mesh_colors[:, 3] = alpha
            render_mode = "transparent"

        geom = Geometry(
            kind="mesh",
            positions=verts,
            indices=faces,
            normals=norms,
            colors=mesh_colors,
        )
        return [SceneObject(
            id="metaballs",
            geometry=geom,
            render_mode=render_mode,
            material=material
        )]

    def _update_surface(
        self,
        coords: np.ndarray,
        surface_cfg: dict,
        colors_per_ca: np.ndarray | None
    ) -> list[SceneObject] | None:
        if not self._surface_visible:
            return None

        surface_alpha = float(surface_cfg.get("alpha", 0.85))
        surface_ao_radius = float(surface_cfg.get("ao_radius", 4.5))
        surface_ao_strength = float(surface_cfg.get("ao_strength", 0.6))
        surface_base_color = np.asarray(
            surface_cfg.get("base_color", self._base_color_single), dtype=float
        )
        if surface_base_color.shape[0] != 4:
            surface_base_color = np.asarray(self._base_color_single, dtype=float)

        if self._all_atom_coords is not None:
            pts_surface = np.asarray(self._all_atom_coords, dtype=float)
        elif coords is not None:
            pts_surface = np.asarray(coords, dtype=float)
        else:
            return None

        if pts_surface.size == 0:
            return None

        n_pts = pts_surface.shape[0]

        # --- Try mesh surface via Gaussian density + marching cubes ---
        grid_spacing = float(surface_cfg.get("grid_spacing", 0.8))
        iso_value = float(surface_cfg.get("iso_value", 0.5))
        padding = float(surface_cfg.get("padding", 3.0))
        max_dim = int(surface_cfg.get("max_dim", 96))
        mesh_sigma_factor = float(surface_cfg.get("mesh_sigma_factor", 1.0))
        mesh_sigma_default = float(surface_cfg.get("mesh_sigma_default", 1.8))

        method = str(surface_cfg.get("method", "gaussian")).lower()
        probe_radius = float(surface_cfg.get("probe_radius", 1.4))

        if method in ("sas", "ses"):
            if self._all_atom_radii is not None and self._all_atom_radii.shape[0] == n_pts:
                atom_radii = np.asarray(self._all_atom_radii, dtype=float)
            else:
                atom_radii = np.full(n_pts, mesh_sigma_default, dtype=float)

            mesh_data = _generate_surface_mesh_edt(
                pts_surface,
                atom_radii,
                method=method,
                probe_radius=probe_radius,
                grid_spacing=grid_spacing,
                padding=padding,
                max_dim=max_dim,
            )
            mesh_sigmas = atom_radii
        else:
            if self._all_atom_radii is not None and self._all_atom_radii.shape[0] == n_pts:
                sigmas = np.asarray(self._all_atom_radii, dtype=float) * mesh_sigma_factor
            else:
                sigmas = np.full(n_pts, mesh_sigma_default, dtype=float)

            mesh_data = _generate_surface_mesh_from_gaussians(
                pts_surface,
                sigmas,
                grid_spacing=grid_spacing,
                padding=padding,
                iso_value=iso_value,
                max_dim=max_dim,
            )
            mesh_sigmas = sigmas

        if mesh_data is not None:
            verts, faces, norms = mesh_data
            return self._build_surface_mesh_scene(
                verts, faces, norms, pts_surface, mesh_sigmas,
                surface_cfg, colors_per_ca, surface_base_color, surface_alpha,
            )

        # --- Fallback: point-cloud surface ---
        surface_size_scale = float(surface_cfg.get("size_scale", 0.03))
        surface_min_size = float(surface_cfg.get("min_size", 2.5))
        surface_ao_max = int(surface_cfg.get("max_neighbors", 24))
        surface_max_points = int(surface_cfg.get("max_points", 10000))
        surface_color_mode = str(surface_cfg.get("color_mode", "ao_gray")).lower()

        colors_surface = self._build_surface_atom_colors(
            pts_surface, surface_cfg, colors_per_ca, surface_base_color,
        )
        if colors_surface is None:
            colors_surface = np.tile(surface_base_color, (pts_surface.shape[0], 1))

        max_pts = max(1, surface_max_points)
        if pts_surface.shape[0] > max_pts:
            step = max(1, pts_surface.shape[0] // max_pts)
            pts_surface = pts_surface[::step]
            colors_surface = colors_surface[::step]

        colors_surface = colors_surface.copy()

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
        return [SceneObject(id="surface", geometry=geom, render_mode="transparent")]

    def _build_surface_atom_colors(
        self,
        pts_surface: np.ndarray,
        surface_cfg: dict,
        colors_per_ca: np.ndarray | None,
        surface_base_color: np.ndarray,
    ) -> np.ndarray | None:
        """Build per-atom/point colors for the surface representation."""
        surface_color_mode = str(surface_cfg.get("color_mode", "ao_gray")).lower()
        n_pts = pts_surface.shape[0]

        if (
            surface_color_mode == "by_residue"
            and self._all_atom_res_ids is not None
            and self._residue_ids is not None
            and colors_per_ca is not None
            and len(colors_per_ca) == len(self._residue_ids)
        ):
            color_map = {rid: colors_per_ca[i_res] for i_res, rid in enumerate(self._residue_ids)}
            colors = np.zeros((n_pts, 4), dtype=float)
            for i_atom, rid in enumerate(self._all_atom_res_ids):
                colors[i_atom, :] = color_map.get(rid, self._base_color_single)
        else:
            colors = np.tile(surface_base_color, (n_pts, 1))

        if (
            getattr(self, "_colors_per_atom_override", None) is not None
            and self._all_atom_res_ids is not None
            and len(self._colors_per_atom_override) == self._all_atom_res_ids.shape[0]
        ):
            ov = np.asarray(self._colors_per_atom_override, dtype=float)
            for i_atom in range(min(colors.shape[0], ov.shape[0])):
                col_ov = ov[i_atom]
                if np.isfinite(col_ov).all():
                    colors[i_atom, :] = col_ov

        return colors

    def _build_surface_mesh_scene(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        norms: np.ndarray,
        pts_surface: np.ndarray,
        sigmas: np.ndarray,
        surface_cfg: dict,
        colors_per_ca: np.ndarray | None,
        surface_base_color: np.ndarray,
        surface_alpha: float,
    ) -> list[SceneObject] | None:
        """Build a colored mesh SceneObject for the Gaussian surface."""
        surface_ao_radius = float(surface_cfg.get("ao_radius", 4.5))
        surface_ao_strength = float(surface_cfg.get("ao_strength", 0.6))

        base_color = np.asarray(self._base_color_single, dtype=float)
        if base_color.shape[0] != 4:
            base_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

        n_pts = pts_surface.shape[0]
        mesh_colors = np.zeros((verts.shape[0], 4), dtype=float)

        atom_colors = self._build_surface_atom_colors(
            pts_surface, surface_cfg, colors_per_ca, surface_base_color,
        )
        if atom_colors is None:
            atom_colors = np.tile(base_color, (n_pts, 1))

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts_surface)
            max_sigma = float(np.max(sigmas))
            cutoff = max_sigma * 2.5

            indices = tree.query_ball_point(verts, r=cutoff)

            for i_v, atom_indices in enumerate(indices):
                if not atom_indices:
                    _, nearest = tree.query(verts[i_v])
                    mesh_colors[i_v] = atom_colors[nearest]
                    continue

                v_pos = verts[i_v]
                w_sum = 0.0
                c_sum = np.zeros(4, dtype=float)

                for i_a in atom_indices:
                    d2 = np.sum((v_pos - pts_surface[i_a])**2)
                    s2 = sigmas[i_a]**2
                    w = math.exp(-d2 / (2.0 * s2))
                    c_sum += atom_colors[i_a] * w
                    w_sum += w

                if w_sum > 0:
                    mesh_colors[i_v] = c_sum / w_sum
                else:
                    _, nearest = tree.query(v_pos)
                    mesh_colors[i_v] = atom_colors[nearest]

            # Analytical normals from Gaussian gradient
            method = str(surface_cfg.get("method", "gaussian")).lower()
            if method not in ("sas", "ses"):
                new_norms = np.zeros_like(verts)
                for i_v, atom_indices in enumerate(indices):
                    if not atom_indices:
                        continue
                    v_pos = verts[i_v]
                    grad = np.zeros(3, dtype=float)
                    for i_a in atom_indices:
                        diff = v_pos - pts_surface[i_a]
                        d2 = np.sum(diff**2)
                        s2 = sigmas[i_a]**2
                        w = math.exp(-d2 / (2.0 * s2))
                        grad += (diff / s2) * w
                    mag = np.linalg.norm(grad)
                    if mag > 1e-6:
                        new_norms[i_v] = grad / mag
                    else:
                        new_norms[i_v] = norms[i_v]
                norms = new_norms

            if surface_ao_strength > 0:
                occ = _estimate_ambient_occlusion(verts, radius=surface_ao_radius, max_neighbors=32)
                if occ is not None:
                    darken = 1.0 - (occ * surface_ao_strength)
                    mesh_colors[:, :3] *= darken[:, np.newaxis]

        except Exception:
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(pts_surface)
                _, nearest = tree.query(verts)
                mesh_colors = atom_colors[nearest]
            except Exception:
                mesh_colors[:, :] = base_color

        render_mode = "opaque"
        if surface_alpha < 1.0:
            mesh_colors[:, 3] = surface_alpha
            render_mode = "transparent"

        geom = Geometry(
            kind="mesh",
            positions=verts,
            indices=faces,
            normals=norms,
            colors=mesh_colors,
        )
        return [SceneObject(id="surface", geometry=geom, render_mode=render_mode)]

    def _update_dots(
        self,
        coords: np.ndarray,
        colors_per_ca: np.ndarray | None,
    ) -> list[SceneObject] | None:
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

        positions: np.ndarray | None
        colors_local: np.ndarray | None

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

    def _update_custom_overlays(self, surface_cfg: dict) -> list[SceneObject] | None:
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

            geom = Geometry(
                kind="points",
                positions=pts_ov,
                colors=colors_ov,
                meta={"size": size_ov, "glyph": overlay.get("glyph", "sphere")},
            )
            scene_objects.append(SceneObject(id="overlay", geometry=geom, render_mode="transparent"))

            if overlay.get("label"):
                label_geom = Geometry(
                    kind="text",
                    positions=pts_ov[0].reshape(1, 3),
                    colors=colors_ov[0].reshape(1, 4),
                    meta={"labels": [overlay["label"]]},
                )
                scene_objects.append(SceneObject(id="overlay_label", geometry=label_geom, render_mode="overlay"))

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

    def _update_selection_highlight(self, coords: np.ndarray) -> list[SceneObject] | None:
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

    def _build_scene_for_current_object(self, object_prefix: str | None = None) -> list[SceneObject]:
        if self._coords is None or self._coords.size == 0:
            return []

        coords = np.asarray(self._coords, dtype=float)
        if coords.ndim == 3:
            state = self._get_active_state()
            idx = self._select_state_frame(state, getattr(state, "active_frame", 0))
            coords = np.asarray(state.coords, dtype=float)
            state.active_frame = idx
        if coords.ndim != 2 or coords.shape[1] != 3:
            return []
        coords = coords.copy()
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

        metaball_cfg = _DISPLAY_CONFIG.get("metaball", {})
        scene_objects += self._update_metaballs(coords, metaball_cfg, self._colors_per_ca) or []

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
