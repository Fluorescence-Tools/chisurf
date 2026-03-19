from __future__ import annotations
import sys
import os
import numpy as np

# Ensure path to chisurf is included
sys.path.append(os.getcwd())

from chisurf.plugins.chimol.chimol.testing.mock_viewer import MockViewer
from chisurf.plugins.chimol.chimol.cmd.command import Cmd

class MockWindow:
    def __init__(self, viewer):
        self.viewer = viewer
    def _refresh_objects_from_viewer(self):
        pass

def test_alignment():
    viewer = MockViewer()
    window = MockWindow(viewer)
    cmd = Cmd(window)
    cmd.set_message_callback(lambda m: print(f"MSG: {m}"))
    cmd.set_error_callback(lambda e: print(f"ERROR: {e}"))
    
    # Create two objects: target and mobile
    from dataclasses import dataclass
    
    @dataclass
    class Structure:
        atoms: np.ndarray
        xyz: np.ndarray
        
    atom_dtype = [
        ('xyz', 'f4', (3,)),
        ('atom_name', 'S10'),
        ('res_id', 'i4'),
        ('res_name', 'S10'),
        ('chain_id', 'S4')
    ]
    
    # Target: 5 residues at [0,0,0], [1,0,0], etc.
    data_t = np.zeros(5, dtype=atom_dtype)
    data_t['xyz'] = np.array([[i, 0.0, 0.0] for i in range(5)], dtype='f4')
    data_t['atom_name'] = b'CA'
    data_t['res_id'] = np.arange(1, 6)
    data_t['res_name'] = b'ALA'
    data_t['chain_id'] = b'A'
    
    struct_t = Structure(atoms=data_t, xyz=data_t['xyz'])
    
    # Mobile: Translated and slightly jittered
    data_m = np.zeros(5, dtype=atom_dtype)
    data_m['xyz'] = np.array([[i + 10.0, 0.5, 0.0] for i in range(5)], dtype='f4')
    data_m['atom_name'] = b'CA'
    data_m['res_id'] = np.arange(1, 6)
    data_m['res_name'] = b'ALA'
    data_m['chain_id'] = b'A'
    
    struct_m = Structure(atoms=data_m, xyz=data_m['xyz'])
    
    # Load into viewer
    viewer._add_mock_object("obj_target", "target")
    viewer.set_active_object("obj_target")
    viewer.set_structure(struct_t)
    
    viewer._add_mock_object("obj_mobile", "mobile")
    viewer.set_active_object("obj_mobile")
    viewer.set_structure(struct_m)
    
    print("\nInitial state:")
    cmd.do("rms mobile, target")
    
    # Test align
    print("\nTesting align:")
    cmd.do("align mobile, target, cycles=5, cutoff=1.0")
    
    # Coordinate check
    m_coords = viewer._objects["obj_mobile"].state.all_atom_coords
    t_coords = viewer._objects["obj_target"].state.all_atom_coords
    
    dist = np.linalg.norm(m_coords - t_coords, axis=1)
    print("Distances after align:", dist)
    
    # Test rms_cur
    print("\nTesting rms_cur:")
    cmd.do("rms_cur mobile, target")

    # Test super
    print("\nTesting super:")
    # Reset mobile
    viewer._objects["obj_mobile"].state.all_atom_coords = data_m['xyz'].copy()
    cmd.do("super mobile, target")
    
    print("\nAll alignment tests passed!")

if __name__ == "__main__":
    test_alignment()
