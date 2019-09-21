import utils
import os
import unittest


TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mfm.io
import tempfile
import numpy as np


class Tests(unittest.TestCase):

    def test_fetch_pdb(self):
        pdb_id = "148L"
        mfm.io.pdb.fetch_pdb_string(pdb_id)

    def test_parse_string_pdb(self):
        pdb_id = "148L"
        s = mfm.io.pdb.fetch_pdb_string(pdb_id)
        mfm.io.pdb.parse_string_pdb(s)

    def test_ascii(self):
        x = np.linspace(
            start=0,
            stop=2 * np.pi,
            num=100
        )
        y = np.sin(x)
        file = tempfile.NamedTemporaryFile(suffix='.txt')
        mfm.io.ascii.save_xy(
            filename=file.name,
            x=x,
            y=y,
            fmt="%f\t%f\n"
        )
        x2, y2 = mfm.io.ascii.load_xy(
            filename=file.name,
            usecols=(0, 1),
            delimiter="\t"
        )
        self.assertEqual(
            np.allclose(
                x, x2
            ),
            True
        )
        self.assertEqual(
            np.allclose(
                y, y2
            ),
            True
        )

if __name__ == '__main__':
    unittest.main()

