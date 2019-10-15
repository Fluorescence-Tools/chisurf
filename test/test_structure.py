import utils
import os
import unittest

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../chisurf/')
)
utils.set_search_paths(TOPDIR)

import tempfile
import numpy as np
import chisurf.structure.structure
import mfm.fio.zipped


class Tests(unittest.TestCase):

    def test_structure_Structure(self):
        s1 = chisurf.structure.structure.Structure(
            pdb_id='148L'
        )
        s2 = chisurf.structure.structure.Structure('148L')
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
        s3 = chisurf.structure.structure.Structure(
            filename=filename
        )
        s4 = chisurf.structure.structure.Structure(
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
