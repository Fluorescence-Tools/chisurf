import unittest
from unittest.mock import MagicMock
import numpy as np
from chisurf.plugins.chimol.chimol.cmd.sele_parser import Evaluator, Token, tokenize

class TestSelectionGrammar(unittest.TestCase):
    def setUp(self):
        # Mock viewer and structure
        self.viewer = MagicMock()
        
        # Mock an object entry
        state = MagicMock()
        
        # 10 atoms, 5 residues (2 atoms per res), 2 chains
        # res_id 0, 1 -> chain A
        # res_id 2, 3, 4 -> chain B
        # res_ids: 0,0, 1,1, 2,2, 3,3, 4,4
        state.all_atom_res_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        state.residue_ids = np.array([0, 1, 2, 3, 4])
        state.residue_names = np.array(["ALA", "ARG", "ASN", "ASP", "CYS"])
        state.residue_chain_ids = np.array(["A", "A", "B", "B", "B"])
        
        # Atoms structured array
        dtype = [("atom_name", "U4"), ("xyz", float, (3,)), ("element", "U2"), ("res_name", "U3"), ("chain", "U1")]
        atoms = np.zeros(10, dtype=dtype)
        atoms["atom_name"] = ["N", "CA"] * 5
        atoms["element"] = ["N", "C"] * 5
        atoms["res_name"] = ["ALA", "ALA", "ARG", "ARG", "ASN", "ASN", "ASP", "ASP", "CYS", "CYS"]
        atoms["chain"] = ["A", "A", "A", "A", "B", "B", "B", "B", "B", "B"]
        state.atoms = atoms
        
        # Coords
        state.all_atom_coords = np.zeros((10, 3))
        state.all_atom_coords[:, 0] = np.arange(10)
        
        entry = MagicMock()
        entry.state = state
        self.viewer._objects = {"obj1": entry}
        self.evaluator = Evaluator(self.viewer, "obj1")

    def test_basic_tokens(self):
        tokens = tokenize("resi 1-5 and name CA")
        types = [t.type for t in tokens]
        self.assertEqual(types, ["IDENT", "INT", "MINUS", "INT", "AND", "IDENT", "IDENT"])

    def test_all_none(self):
        mask = self.evaluator.evaluate("all")
        self.assertTrue(np.all(mask))
        self.assertEqual(len(mask), 10)
        
        mask = self.evaluator.evaluate("none")
        self.assertFalse(np.any(mask))

    def test_resi(self):
        # res_id are 0-4 in our mock
        mask = self.evaluator.evaluate("resi 1")
        # should match atoms with res_id 1 (global unique)
        # in our mock all_atom_res_ids == [0,0, 1,1, 2,2, 3,3, 4,4]
        # so resi 1 matches indices 2, 3
        self.assertEqual(np.where(mask)[0].tolist(), [2, 3])
        
        mask = self.evaluator.evaluate("resi 1-2")
        self.assertEqual(np.where(mask)[0].tolist(), [2, 3, 4, 5])

    def test_name(self):
        mask = self.evaluator.evaluate("name CA")
        self.assertEqual(np.where(mask)[0].tolist(), [1, 3, 5, 7, 9])

    def test_chain(self):
        mask = self.evaluator.evaluate("chain A")
        self.assertEqual(np.where(mask)[0].tolist(), [0, 1, 2, 3])

    def test_boolean(self):
        mask = self.evaluator.evaluate("chain A and name CA")
        self.assertEqual(np.where(mask)[0].tolist(), [1, 3])
        
        mask = self.evaluator.evaluate("name N or resi 4")
        self.assertEqual(np.where(mask)[0].tolist(), [0, 2, 4, 6, 8, 9])

    def test_not(self):
        mask = self.evaluator.evaluate("not chain A")
        self.assertEqual(np.where(mask)[0].tolist(), [4, 5, 6, 7, 8, 9])

    def test_macro(self):
        # /obj/chain/res/name
        mask = self.evaluator.evaluate("/obj1/A/1/CA")
        self.assertEqual(np.where(mask)[0].tolist(), [3])

if __name__ == "__main__":
    unittest.main()
