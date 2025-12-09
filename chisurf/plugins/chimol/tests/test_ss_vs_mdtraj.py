import unittest
from pathlib import Path

import numpy as np

try:
    import mdtraj as md
except Exception:  # pragma: no cover - optional dependency
    md = None

from chisurf.plugins.chimol.chimol.analysis import assign_ss_c3_from_file


class TestChimolSSAgainstMDTraj(unittest.TestCase):

    def setUp(self) -> None:
        # Locate 148l.pdb relative to repository root.
        # This file lives at
        #   .../chisurf/chisurf/plugins/chimol/tests/test_ss_vs_mdtraj.py
        # so ``parents[4]`` is the repo root (chisurf).
        if md is None:
            self.skipTest("mdtraj not installed; skipping DSSP comparison test")
        here = Path(__file__).resolve()
        repo_root = here.parents[4]
        pdb_path = repo_root / "test" / "data" / "atomic_coordinates" / "pdb_files" / "148l.pdb"
        if not pdb_path.is_file():
            self.skipTest(f"Test PDB not found: {pdb_path}")
        self._pdb_path = pdb_path

    def test_ss_roughly_agrees_with_mdtraj(self) -> None:
        """Chimol DSSP-like SS should roughly agree with mdtraj+DSSP.

        The goal is not bit-for-bit equality, but to ensure:
        - We assign a mix of H/E/C (not all coil).
        - There is a reasonable fraction of residues where our C3 code
          matches mdtraj's simplified DSSP.
        """

        try:
            traj = md.load(str(self._pdb_path))
            dssp = md.compute_dssp(traj, simplified=True)[0]
        except Exception as e:  # pragma: no cover - environment dependent
            self.skipTest(f"mdtraj or external DSSP not available: {e}")

        ref = np.array([c if c in ("H", "E", "C") else "C" for c in dssp], dtype="U1")
        n_res = int(ref.shape[0])

        codes = assign_ss_c3_from_file(str(self._pdb_path), n_res)
        self.assertIsNotNone(codes)

        codes_arr = np.asarray(codes, dtype="U1")
        self.assertEqual(codes_arr.shape[0], n_res)

        # Ensure we actually see helices or strands, not all coil.
        has_helix_or_strand = np.any((codes_arr == "H") | (codes_arr == "E"))
        self.assertTrue(has_helix_or_strand, "Chimol SS assignment produced only coil (C)")

        # Require a modest level of agreement with mdtraj DSSP. The exact
        # fraction is heuristic, but with a PyDSSP-style implementation we
        # expect a reasonably close match.
        frac_same = float(np.mean(codes_arr == ref))
        self.assertGreater(
            frac_same,
            0.4,
            f"Too little agreement with mdtraj DSSP: match fraction = {frac_same:.3f}",
        )


if __name__ == "__main__":  # pragma: no cover - manual test execution
    unittest.main()
