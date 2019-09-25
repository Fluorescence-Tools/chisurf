import utils
import os
import unittest


TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mfm.io
import tempfile
import numpy as np


class Tests(unittest.TestCase):

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
            fmt="%f\t%f\n",
            header_string="x\ty\n"
        )
        x2, y2 = mfm.io.ascii.load_xy(
            filename=file.name,
            usecols=(0, 1),
            delimiter="\t",
            skiprows=1
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

    def test_csv(self):
        x = np.linspace(
            start=0,
            stop=2 * np.pi,
            num=32
        )
        y = np.sin(x)
        file = tempfile.NamedTemporaryFile(suffix='.txt')
        mfm.io.ascii.save_xy(
            filename=file.name,
            x=x,
            y=y,
            fmt="%f\t%f\n",
            header_string="x\ty\n"
        )

        csv = mfm.io.ascii.Csv(
            filename=file.name,
            skiprows=1
        )
        self.assertEqual(
            np.allclose(
                y,
                csv.data[1]
            ),
            True
        )
        self.assertEqual(
            np.allclose(
                x,
                csv.data[0]
            ),
            True
        )
        self.assertEqual(
            csv.filename,
            file.name
        )

        x = np.linspace(
            start=0,
            stop=2 * np.pi,
            num=32
        )
        y = np.cos(x)
        file2 = tempfile.NamedTemporaryFile(suffix='.txt')
        mfm.io.ascii.save_xy(
            filename=file2.name,
            x=x,
            y=y,
            fmt="%f\t%f\n",
            header_string="x\ty\n"
        )
        csv.load(
            filename=file2.name
        )
        self.assertEqual(
            csv.filename,
            file2.name
        )

    def test_fetch_pdb(self):
        pdb_id = "148L"
        s = mfm.io.pdb.fetch_pdb_string(pdb_id)
        self.assertEqual(
            'HEADER    HYDROLASE/HYDROLASE SUBSTRATE           27-OCT-93   148L              \nTITLE     A COVALEN',
            s[:100]
        )

    def test_parse_string_pdb(self):
        pdb_id = "148L"
        s = mfm.io.pdb.fetch_pdb_string(pdb_id)
        atoms = mfm.io.pdb.parse_string_pdb(s)
        atoms_reference = np.array(
            [[7.71, 28.561, 39.546],
             [8.253, 29.664, 38.758],
             [7.445, 30.133, 37.548],
             [7.189, 29.466, 36.54],
             [9.738, 29.578, 38.445],
             [10.256, 30.962, 38.143],
             [10.845, 31.785, 39.624],
             [11.874, 30.541, 40.499],
             [7.052, 31.375, 37.689],
             [6.241, 32.049, 36.726]]
        )
        self.assertEqual(
            np.allclose(
                atoms['xyz'][:10],
                atoms_reference
            ),
            True
        )

    def test_read_pdb(self):
        pdb_id = "148L"
        file = tempfile.NamedTemporaryFile(
            suffix='.pdb'
        )
        with open(file=file.name, mode='w') as fp:
            fp.write(
                mfm.io.pdb.fetch_pdb_string(pdb_id)
            )

        atoms = mfm.io.pdb.read(
            filename=file.name
        )
        atoms_reference = np.array(
            [(0, 'E', 1, 'MET', 1, 'N', '', [7.71, 28.561, 39.546], 0., 0., 41., 0.),
             (1, 'E', 1, 'MET', 2, 'CA', 'C', [8.253, 29.664, 38.758], 0., 1.7, 41.8, 12.011),
             (2, 'E', 1, 'MET', 3, 'C', '', [7.445, 30.133, 37.548], 0., 0., 20.4, 0.),
             (3, 'E', 1, 'MET', 4, 'O', '', [7.189, 29.466, 36.54], 0., 0., 22.9, 0.),
             (4, 'E', 1, 'MET', 5, 'CB', 'C', [9.738, 29.578, 38.445], 0., 1.7, 58.3, 12.011)],
            dtype=[('i', '<i4'), ('chain', '<U1'), ('res_id', '<i4'), ('res_name', '<U5'),
                   ('atom_id', '<i4'), ('atom_name', '<U5'), ('element', '<U1'),
                   ('xyz', '<f8', (3,)), ('charge', '<f8'), ('radius', '<f8'),
                   ('bfactor', '<f8'), ('mass', '<f8')]
        )
        self.assertEqual(
            np.allclose(
                atoms['xyz'][:5],
                atoms_reference['xyz']
            ),
            True
        )


if __name__ == '__main__':
    unittest.main()

