import utils
import os
import unittest
import tempfile
import copy
import numpy as np

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)

import chisurf.structure
import scikit_fluorescence.io.zipped


class Tests(unittest.TestCase):

    pdb_filename = './test/data/atomic_coordinates/pdb_files/hGBP1_closed.pdb'
    s1 = chisurf.structure.ProteinCentroid(
        pdb_filename,
        verbose=True
    )

    s1_ref_xyz = np.array(
        [[72.739, -17.501, 8.879],
         [73.841, -17.042, 9.747],
         [74.361, -18.178, 10.643],
         [73.642, -18.708, 11.489],
         [73.1036816, -14.05035305, 10.73760945]]
    )

    s2 = chisurf.structure.Structure(
        pdb_filename,
        verbose=True
    )

    def test_structure_Structure(self):
        s1 = chisurf.structure.Structure(pdb_id='148L')
        s2 = chisurf.structure.Structure('148L')
        self.assertEqual(
            np.allclose(
                s1.atoms['xyz'],
                s2.atoms['xyz']
            ),
            True
        )
        _, filename = tempfile.mkstemp(
            suffix=".pdb"
        )
        s2.write(
            filename=filename
        )
        s2.write(
            filename=filename + '.gz'
        )
        s3 = chisurf.structure.Structure(
            filename=filename
        )
        self.assertAlmostEqual(
            s3.radius_gyration,
            23.439021926160564
        )

    def test_structure_copy(self):
        s2 = self.s2
        s5 = copy.copy(s2)
        self.assertEqual(
            s5.atoms is s2.atoms,
            True
        )
        s6 = copy.deepcopy(s2)
        self.assertEqual(
            s6.atoms is s2.atoms,
            False
        )
        s1 = self.s1
        s6 = copy.deepcopy(s1)
        self.assertEqual(
            s6.atoms is s1.atoms,
            False
        )
        s7 = copy.copy(s1)
        self.assertEqual(
            s7.atoms is s1.atoms,
            True
        )

    def test_angles(self):
        s1 = self.s1
        a = self.s1_ref_xyz
        # the omega angles
        self.assertEqual(
            s1.n_residues,
            s1.omega.shape[0]
        )
        self.assertEqual(
            s1.n_residues,
            s1.psi.shape[0]
        )
        self.assertEqual(
            s1.n_residues,
            s1.phi.shape[0]
        )
        self.assertEqual(
            s1.auto_update,
            False
        )

    def test_change_dihedral(self):

        self.assertEqual(
            np.allclose(
                self.s1_ref_xyz,
                self.s1.atoms['xyz'][:5]
            ),
            True
        )

        s1 = self.s1
        a = self.s1_ref_xyz
        self.assertEqual(
            np.allclose(
                a,
                s1.atoms['xyz'][:5]
            ),
            True
        )

        s1.omega *= 0.0
        s1.update()
        self.assertEqual(
            np.allclose(
                a,
                s1.atoms['xyz'][:5]
            ),
            False
        )

    def test_traj_opening(self):
        import chisurf.structure
        traj = chisurf.structure.TrajectoryFile(
            './test/data/atomic_coordinates/trajectory/h5-file/hgbp1_transition.h5',
            stride=1
        )
        self.assertEqual(
            isinstance(traj[0], chisurf.structure.Structure),
            True
        )
        self.assertEqual(
            len(traj),
            464
        )

    def test_traj_writing(self):
        import tempfile
        import chisurf.structure

        _, filename = tempfile.mkstemp('.h5')
        structure = chisurf.structure.ProteinCentroid(
            './test/data/atomic_coordinates/pdb_files/hGBP1_closed.pdb',
            auto_update=True
        )
        traj_write = chisurf.structure.TrajectoryFile(
            structure,
            filename=filename
        )
        self.assertEqual(
            len(traj_write),
            1
        )

        # append structures
        structure.omega *= 0.0
        n = 22
        for i in range(n):
            traj_write.append(structure)
        self.assertEqual(
            len(traj_write),
            n + 1
        )
        len(traj_write)

    @unittest.expectedFailure
    def test_labeled_structure(self):
        import chisurf.structure
        import chisurf.structure.labeled_structure
        structure = chisurf.structure.Structure('./test/data/atomic_coordinates/pdb_files/hGBP1_closed.pdb')
        donor_description = {'residue_seq_number': 18, 'atom_name' : 'CB'}
        acceptor_description = {'residue_seq_number': 577, 'atom_name' : 'CB'}
        pRDA, rda = chisurf.structure.labeled_structure.av_distance_distribution(
            structure,
            donor_av_parameter=donor_description,
            acceptor_av_parameter=acceptor_description
        )
