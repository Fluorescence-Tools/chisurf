from __future__ import annotations
import numpy as np
import pytest
from dataclasses import dataclass

from chisurf.plugins.chimol.chimol.testing.mock_viewer import MockViewer
from chisurf.plugins.chimol.chimol.cmd.command import Cmd

class MockWindow:
    def __init__(self, viewer):
        self.viewer = viewer
    def _refresh_objects_from_viewer(self):
        pass

@dataclass
class Structure:
    atoms: np.ndarray
    xyz: np.ndarray

def get_test_structure(translated=False):
    atom_dtype = [
        ('xyz', 'f4', (3,)),
        ('atom_name', 'S10'),
        ('res_id', 'i4'),
        ('res_name', 'S10'),
        ('chain_id', 'S4')
    ]
    
    data = np.zeros(5, dtype=atom_dtype)
    offset = 10.0 if translated else 0.0
    jitter = 0.5 if translated else 0.0
    data['xyz'] = np.array([[i + offset, jitter, 0.0] for i in range(5)], dtype='f4')
    data['atom_name'] = b'CA'
    data['res_id'] = np.arange(1, 6)
    data['res_name'] = b'ALA'
    data['chain_id'] = b'A'
    return Structure(atoms=data, xyz=data['xyz'])

def test_chimol_alignment():
    viewer = MockViewer()
    window = MockWindow(viewer)
    cmd = Cmd(window)
    
    struct_t = get_test_structure(translated=False)
    struct_m = get_test_structure(translated=True)
    
    # Load into viewer
    viewer._add_mock_object("obj_target", "target")
    viewer.set_active_object("obj_target")
    viewer.set_structure(struct_t)
    
    viewer._add_mock_object("obj_mobile", "mobile")
    viewer.set_active_object("obj_mobile")
    viewer.set_structure(struct_m)
    
    # Test align
    cmd.do("align mobile, target, cycles=5, cutoff=1.0")
    
    # Coordinate check
    m_coords = viewer._objects["obj_mobile"].state.all_atom_coords
    t_coords = viewer._objects["obj_target"].state.all_atom_coords
    
    dist = np.linalg.norm(m_coords - t_coords, axis=1)
    assert np.all(dist < 1e-3)

def test_chimol_rms_cur():
    viewer = MockViewer()
    window = MockWindow(viewer)
    cmd = Cmd(window)
    
    struct_t = get_test_structure(translated=False)
    struct_m = get_test_structure(translated=False) # No translation
    
    viewer._add_mock_object("obj_target", "target")
    viewer.set_active_object("obj_target")
    viewer.set_structure(struct_t)
    
    viewer._add_mock_object("obj_mobile", "mobile")
    viewer.set_active_object("obj_mobile")
    viewer.set_structure(struct_m)
    
    # rms_cur should work without changing anything
    cmd.do("rms_cur mobile, target")

def test_chimol_super():
    viewer = MockViewer()
    window = MockWindow(viewer)
    cmd = Cmd(window)
    
    struct_t = get_test_structure(translated=False)
    struct_m = get_test_structure(translated=True)
    
    viewer._add_mock_object("obj_target", "target")
    viewer.set_active_object("obj_target")
    viewer.set_structure(struct_t)
    
    viewer._add_mock_object("obj_mobile", "mobile")
    viewer.set_active_object("obj_mobile")
    viewer.set_structure(struct_m)
    
    cmd.do("super mobile, target")
    
    m_coords = viewer._objects["obj_mobile"].state.all_atom_coords
    t_coords = viewer._objects["obj_target"].state.all_atom_coords
    dist = np.linalg.norm(m_coords - t_coords, axis=1)
    assert np.all(dist < 1e-3)
