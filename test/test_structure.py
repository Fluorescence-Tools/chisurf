import utils
import os
import unittest
import tempfile
import numpy as np

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)

import chisurf.structure
import chisurf.fio.zipped


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

    def test_structure_Structure(self):
        s1 = chisurf.structure.Structure(
            pdb_id='148L'
        )
        s2 = chisurf.structure.Structure('148L')
        self.assertEqual(
            np.allclose(
                s1.atoms['xyz'],
                s2.atoms['xyz']
            ),
            True
        )
        #file = tempfile.NamedTemporaryFile(suffix='.pdb')
        #filename 0 file.name
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
        s4 = chisurf.structure.Structure(
            filename=filename + '.gz'
        )
        self.assertEqual(
            np.allclose(
                s3.atoms['xyz'],
                s4.atoms['xyz']
            ),
            True
        )
        self.assertEqual(
            os.path.getsize(
                filename
                ) >
            os.path.getsize(
                filename + '.gz'
            ),
            True
        )
        self.assertAlmostEqual(
            s4.radius_gyration,
            23.439021926160564
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
        traj = chisurf.structure.TrajectoryFile(
            './test/data/atomic_coordinates/trajectory/h5-file/hgbp1_transition.h5',
            reading_routine='r',
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

    # def test_traj_writing(self):
    #     _, filename = tempfile.mkstemp('.h5')
    #     traj_write = chisurf.structure.TrajectoryFile(
    #         filename,
    #         mode='w'
    #     )

