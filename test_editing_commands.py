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

def test_editing():
    viewer = MockViewer()
    window = MockWindow(viewer)
    cmd = Cmd(window)
    
    # Create an object with some atoms
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
        ('chain_id', 'S4'),
        ('b_factor', 'f4')
    ]
    
    data = np.zeros(10, dtype=atom_dtype)
    data['xyz'] = np.random.rand(10, 3)
    data['atom_name'] = [f'A{i}'.encode() for i in range(10)]
    data['res_id'] = np.arange(10)
    data['res_name'] = b'ALA'
    data['chain_id'] = b'A'
    data['b_factor'] = 10.0
    
    struct = Structure(atoms=data, xyz=data['xyz'])
    viewer.set_structure(struct)
    
    print("Initial atoms:", len(viewer.get_active_state().atoms))
    
    # Test iterate
    print("\nTesting iterate:")
    messages = []
    cmd.set_message_callback(lambda m: print(f"MSG: {m}"))
    cmd.set_error_callback(lambda e: print(f"ERROR: {e}"))
    cmd.do("iterate all, \"print('Atom', name, 'has B=', b)\"")
    
    # Test alter
    print("\nTesting alter:")
    cmd.do("alter resi 0-4, b = 50.0")
    
    atoms = viewer.get_active_state().atoms
    print("B-factors after alter:", atoms['b_factor'][:6])
    
    # Test remove
    print("\nTesting remove:")
    cmd.do("remove resi 0-2")
    
    new_atoms = viewer.get_active_state().atoms
    print("Atoms after remove:", len(new_atoms))
    
    # Test pseudoatom
    print("\nTesting pseudoatom:")
    cmd.do("pseudoatom ps1, pos=[10.0 10.0 10.0]")
    cmd.do("pseudoatom ps1, pos=[20.0 20.0 20.0]") # Append
    
    print("Objects currently in viewer:", viewer._objects.keys())
    for oid, e in viewer._objects.items():
         print(f" - {oid}: {e.name}")
    
    ps_entry = None
    for oid, e in viewer._objects.items():
         if e.name == "ps1":
              ps_entry = e
              break

    ps_atoms = ps_entry.state.atoms
    print("Pseudoatoms count:", len(ps_atoms))
    assert len(ps_atoms) == 2
    assert np.all(ps_atoms[1]['xyz'] == [20.0, 20.0, 20.0])

    # Verification
    assert len(new_atoms) == 7
    assert np.all(atoms['b_factor'][3:5] == 50.0)
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_editing()
