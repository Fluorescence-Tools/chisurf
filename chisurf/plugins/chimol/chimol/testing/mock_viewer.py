from __future__ import annotations
from typing import Optional, Sequence, Dict, Any, List
import numpy as np
from pathlib import Path

from qtpy import QtCore

class MockViewer(QtCore.QObject):
    """Headless implementation of MolView protocol for testing."""

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
