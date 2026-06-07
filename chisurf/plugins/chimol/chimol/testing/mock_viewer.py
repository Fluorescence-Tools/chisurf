from __future__ import annotations
from typing import Optional, Sequence, Dict, Any, List
import numpy as np
from pathlib import Path
import copy


def _get_qobject_base():
    """Return QtCore.QObject if Qt is available, else plain ``object``."""
    try:
        from qtpy import QtCore
        return QtCore.QObject
    except Exception:
        return object


class MockViewer(_get_qobject_base()):
    """Headless implementation of MolView protocol for testing.

    When Qt is available the class inherits from ``QObject`` so it can be
    used inside the Qt application; when running in a headless/CLI context
    it falls back to plain ``object``.
    """

    class MockState:
        def __init__(self):
            self.atoms = None
            self.all_atom_coords = None
            self.all_atom_res_ids = None
            self.all_atom_chain_ids = None
            self.all_atom_radii = None
            self.residue_names = None
            self.residue_ids = None
            self.residue_chain_ids = None
            self.ball_mask = None
            self.sticks_mask = None
            self.cartoon_mask = None
            self.measurements = {}
            self.frames = None
            self.frames_raw = None
            self.active_frame = 0

    class MockEntry:
        def __init__(self, oid, name):
            self.id = oid
            self.name = name
            self.state = MockViewer.MockState()
            self.visible = True

    def __init__(self):
        super().__init__()
        self._background_color = "black"
        self._objects: Dict[str, MockViewer.MockEntry] = {}
        self._active_object_id: Optional[str] = None
        self._total_frames = 1
        self._current_frame = 0
        self._keyframes = {}
        self._animation_running = False
        self._animation_timer = None
        self._color_mode = "single"
        self._selected_residues: List[int] = []
        self._view_state = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            30.0, 20.0, 45.0,
            0.0, 0.0, 0.0,
            0.1, 1000.0, 45.0,
        ]
        self._reps: Dict[str, bool] = {
            "cartoon": True,
            "ca_trace": False,
            "atoms": False,
            "sticks": True,
            "dots": False,
            "surface": False,
            "plane": False
        }

    def get_total_frames(self): return self._total_frames
    def set_total_frames(self, n): self._total_frames = n
    def get_current_frame(self): return self._current_frame
    def set_current_frame(self, i): self._current_frame = i

    def mset(self, first=1, last=1):
        self._total_frames = max(int(last), int(first), 1)

    def start_animation(self):
        self._animation_running = True

    def stop_animation(self):
        self._animation_running = False
        self._current_frame = 0

    def set_background_color(self, color: str):
        self._background_color = color

    def set_structure(self, structure: Any):
        if self._active_object_id is None:
             self._add_mock_object("obj1", "obj1")
        entry = self._objects[self._active_object_id]
        entry.state.atoms = getattr(structure, "atoms", None)
        entry.state.all_atom_coords = getattr(structure, "xyz", None)
        if entry.state.atoms is not None:
             dtype = entry.state.atoms.dtype
             if "res_id" in dtype.names:
                  entry.state.all_atom_res_ids = entry.state.atoms["res_id"]
                  entry.state.residue_ids = np.unique(entry.state.all_atom_res_ids)
             if "chain_id" in dtype.names:
                  entry.state.all_atom_chain_ids = entry.state.atoms["chain_id"]
                  entry.state.residue_chain_ids = np.array([b"A"] * len(entry.state.residue_ids)) # Mock
        self._update_view()

    def add_structure(self, structure: Any, *, name: Optional[str] = None, source_path: Optional[str] = None):
        oid = self._create_object(name=name)
        self.set_structure(structure)
        return oid

    def add_coordinates(self, coords, *, name: Optional[str] = None, source_path: Optional[str] = None):
        oid = self._create_object(name=name)
        entry = self._objects[oid]
        entry.state.all_atom_coords = np.asarray(coords, dtype=float)
        return oid

    def set_frames(self, frames, *, object_id=None):
        oid = object_id or self._active_object_id
        if not oid or oid not in self._objects:
            oid = self._create_object()
        arr = np.asarray(frames, dtype=float)
        self._objects[oid].state.frames = arr
        self._objects[oid].state.frames_raw = arr
        self._objects[oid].state.active_frame = 0
        self._total_frames = max(self._total_frames, int(arr.shape[0]))

    def append_frame(self, frame, *, object_id=None):
        oid = object_id or self._active_object_id
        if not oid or oid not in self._objects:
            oid = self._create_object()
        arr = np.asarray(frame, dtype=float)
        state = self._objects[oid].state
        if state.frames_raw is None:
            state.frames_raw = arr[np.newaxis, :, :]
        else:
            state.frames_raw = np.concatenate([state.frames_raw, arr[np.newaxis, :, :]], axis=0)
        state.frames = state.frames_raw
        state.active_frame = state.frames.shape[0] - 1
        self._total_frames = max(self._total_frames, int(state.frames.shape[0]))
        return int(state.frames.shape[0])

    def get_active_state(self):
        if self._active_object_id:
             return self._objects[self._active_object_id].state
        return None

    def _update_view(self):
        pass

    def list_objects(self) -> List[Dict[str, Any]]:
        return [
            {"id": oid, "name": entry.name, "visible": entry.visible}
            for oid, entry in self._objects.items()
        ]

    def get_active_object_id(self) -> Optional[str]:
        return self._active_object_id

    def set_active_object(self, object_id: str):
        if object_id in self._objects:
            self._active_object_id = object_id

    def _create_object(self, name: Optional[str] = None) -> str:
        idx = len(self._objects) + 1
        oid = f"obj{idx}"
        oname = name or oid
        self._add_mock_object(oid, oname)
        return oid

    def _add_mock_object(self, object_id: str, name: str):
        entry = MockViewer.MockEntry(object_id, name)
        self._objects[object_id] = entry
        self._active_object_id = object_id

    def remove_object(self, object_id: str) -> bool:
        if object_id in self._objects:
            del self._objects[object_id]
            if self._active_object_id == object_id:
                self._active_object_id = next(iter(self._objects)) if self._objects else None
            return True
        return False

    def copy_object(self, object_id: str, *, name: Optional[str] = None):
        if object_id not in self._objects:
            return None
        old = self._objects[object_id]
        new_id = self._create_object(name=name or f"{old.name}_copy")
        new = self._objects[new_id]
        new.state = copy.deepcopy(old.state)
        new.visible = old.visible
        self._active_object_id = new_id
        return new_id

    def set_object_visible(self, object_id: str, visible: bool):
        if object_id in self._objects:
            self._objects[object_id].visible = bool(visible)

    def set_color_mode(self, mode: str):
        self._color_mode = str(mode)

    def clear_color_overrides(self):
        pass

    def reset_view(self):
        pass

    def set_cartoon_visible(self, visible: bool):
        self._reps["cartoon"] = bool(visible)

    def set_trace_visible(self, visible: bool):
        self._reps["ca_trace"] = bool(visible)

    def set_atoms_visible_all(self, visible: bool):
        self._reps["atoms"] = bool(visible)

    def set_sticks_visible(self, visible: bool):
        self._reps["sticks"] = bool(visible)

    def set_dots_visible(self, visible: bool):
        self._reps["dots"] = bool(visible)

    def set_surface_visible(self, visible: bool):
        self._reps["surface"] = bool(visible)

    def set_metaballs_visible(self, visible: bool):
        self._reps["metaball"] = bool(visible)

    def set_plane_visible(self, visible: bool):
        self._reps["plane"] = bool(visible)

    def save_png(self, path, *, width=None, height=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"mock png")
        return True

    def get_view_state(self):
        return list(self._view_state)

    def set_view_state(self, view):
        vals = [float(v) for v in view]
        if len(vals) != 18:
            raise ValueError("view must contain 18 floats")
        self._view_state = vals

    def get_ray_view_state(self) -> list:
        return self.get_view_state()

    def get_atom_sphere_data(self) -> tuple:
        state = self.get_active_state()
        if state is None:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,))
        coords = state.all_atom_coords
        if coords is None:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,))
        n = coords.shape[0]
        positions = np.asarray(coords, dtype=float)
        colors = np.full((n, 3), 0.8, dtype=float)
        radii = np.full(n, 1.5, dtype=float)
        return positions, colors, radii

    def get_residue_positions(self, indices=None, *, object_id=None):
        oid = object_id or self._active_object_id
        if not oid or oid not in self._objects:
            return np.zeros((0, 3))
        state = self._objects[oid].state
        coords = state.all_atom_coords
        if coords is None:
             return np.zeros((0, 3))
        # For simplicity in mock, just return all if indices is None
        return coords

    def set_selected_residues(self, indices, *, object_id=None):
        self._selected_residues = [int(i) for i in list(indices)]

    def apply_transform_to_object(self, rotation, translation, *, object_id=None):
        oid = object_id or self._active_object_id
        if not oid or oid not in self._objects:
            return
        state = self._objects[oid].state
        if state.all_atom_coords is not None:
             state.all_atom_coords = (state.all_atom_coords @ rotation) + translation
        if state.atoms is not None:
             state.atoms["xyz"] = (state.atoms["xyz"] @ rotation) + translation

class MockWindow:
    """Minimal window substitute for Cmd."""
    def __init__(self, viewer: Optional[MockViewer] = None):
        self.viewer = viewer or MockViewer()
        self._object_counter = 0

    def _refresh_objects_from_viewer(self):
        pass

    def _load_structure_from_path(self, path: Path, name: Optional[str] = None):
        self._object_counter += 1
        obj_id = f"obj_{self._object_counter}"
        obj_name = name or path.stem
        self.viewer._add_mock_object(obj_id, obj_name)
        return obj_id
