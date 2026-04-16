from __future__ import annotations
import numpy as np
import pytest
from dataclasses import dataclass

from chisurf.plugins.chimol.chimol.testing.mock_viewer import MockViewer
from chisurf.plugins.chimol.chimol.cmd.command import Cmd

class MockWindow:
    def __init__(self, viewer):
        self.viewer = viewer

@dataclass
class Structure:
    atoms: np.ndarray
    xyz: np.ndarray

@pytest.fixture
def editing_context():
    viewer = MockViewer()
    window = MockWindow(viewer)
    cmd = Cmd(window)
    
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
    return viewer, cmd

def test_chimol_editing_alter(editing_context):
    viewer, cmd = editing_context
    cmd.do("alter resi 0-4, b = 50.0")
    
    atoms = viewer.get_active_state().atoms
    assert np.all(atoms['b_factor'][:5] == 50.0)
    assert np.all(atoms['b_factor'][5:] == 10.0)

def test_chimol_editing_remove(editing_context):
    viewer, cmd = editing_context
    cmd.do("remove resi 0-2")
    
    new_atoms = viewer.get_active_state().atoms
    assert len(new_atoms) == 7

def test_chimol_editing_pseudoatom(editing_context):
    viewer, cmd = editing_context
    cmd.do("pseudoatom ps1, pos=[10.0 10.0 10.0]")
    cmd.do("pseudoatom ps1, pos=[20.0 20.0 20.0]") # Append
    
    ps_entry = None
    for e in viewer._objects.values():
         if e.name == "ps1":
              ps_entry = e
              break

    assert ps_entry is not None
    ps_atoms = ps_entry.state.atoms
    assert len(ps_atoms) == 2
    assert np.all(ps_atoms[1]['xyz'] == [20.0, 20.0, 20.0])
