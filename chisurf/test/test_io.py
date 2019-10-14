import utils
import os
import unittest


TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mfm.io
import tempfile
import glob
import numpy as np


class Tests(unittest.TestCase):

    def test_ascii(self):
        x = np.linspace(
            start=0,
            stop=2 * np.pi,
            num=100
        )
        y = np.sin(x)
        #file = tempfile.NamedTemporaryFile(suffix='.txt')
        #filename = file.name
        _, filename = tempfile.mkstemp(
            suffix='.txt'
        )
        mfm.io.ascii.save_xy(
            filename=filename,
            x=x,
            y=y,
            fmt="%f\t%f\n",
            header_string="x\ty\n"
        )
        x2, y2 = mfm.io.ascii.load_xy(
            filename=filename,
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
        n_points = 32
        reference_x = np.linspace(
            start=0,
            stop=2.0 * np.pi,
            num=n_points
        )
        reference_y = np.sin(reference_x)
        reference_ex = np.zeros_like(reference_x)
        reference_ey = np.ones_like(reference_y)
        reference_data = np.vstack([reference_x, reference_y, reference_ex, reference_ey]).T

        #file = tempfile.NamedTemporaryFile(
        #    suffix='.txt'
        #)
        _, filename = tempfile.mkstemp(
            suffix='.txt'
        )

        # save with basic/simple CSV functions
        mfm.io.ascii.save_xy(
            filename=filename,
            x=reference_x,
            y=reference_y,
            fmt="%f\t%f\n",
            header_string="x\ty\n"
        )

        # CSV class
        csv = mfm.io.ascii.Csv(
            filename=filename,
            skiprows=0,
            use_header=True
        )
        self.assertListEqual(
            csv.header,
            ['x', 'y']
        )
        self.assertEqual(
            np.allclose(
                reference_y,
                csv.data[1]
            ),
            True
        )
        self.assertEqual(
            np.allclose(
                reference_x,
                csv.data[0]
            ),
            True
        )
        self.assertEqual(
            csv.filename,
            filename
        )

        csv.save(
            data=reference_data.T,
            filename=filename,
            delimiter='\t'
        )
        csv.load(
            filename=filename,
            delimiter='\t',
            skiprows=0,
            use_header=False
        )
        self.assertEqual(
            np.allclose(
                reference_data,
                csv.data.T
            ),
            True
        )

        reference_y = np.cos(reference_x)
        #file2 = tempfile.NamedTemporaryFile(
        #    suffix='.txt'
        #)
        #filename2 = file2.name
        _, filename2 = tempfile.mkstemp(
            suffix='.txt'
        )

        mfm.io.ascii.save_xy(
            filename=filename2,
            x=reference_x,
            y=reference_y,
            fmt="%f\t%f\n",
            header_string="x\ty\n"
        )
        csv.load(
            filename=filename2
        )
        self.assertEqual(
            csv.filename,
            filename2
        )

        # test delimiter sniffer
        csv.load(
            filename=filename,
            skiprows=0,
            use_header=False
        )
        self.assertEqual(
            np.allclose(
                reference_data.T,
                csv.data
            ),
            True
        )
        self.assertEqual(
            csv.n_rows,
            n_points
        )
        self.assertEqual(
            csv.n_cols,
            4
        )

    def test_fetch_pdb(self):
        pdb_id = "148L"
        s = mfm.io.coordinates.fetch_pdb_string(pdb_id)
        self.assertEqual(
            'HEADER    HYDROLASE/HYDROLASE SUBSTRATE           27-OCT-93   148L              \nTITLE     A COVALEN',
            s[:100]
        )

    def test_parse_string_pdb(self):
        pdb_id = "148L"
        s = mfm.io.coordinates.fetch_pdb_string(pdb_id)
        atoms = mfm.io.coordinates.parse_string_pdb(s)
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
        #file = tempfile.NamedTemporaryFile(
        #    suffix='.pdb'
        #)
        #filename = file.name
        _, filename = tempfile.mkstemp(
            suffix='.pdb'
        )
        with mfm.io.zipped.open_maybe_zipped(
                filename=filename,
                mode='w'
        ) as fp:
            fp.write(
                mfm.io.coordinates.fetch_pdb_string(pdb_id)
            )

        atoms = mfm.io.coordinates.read(
            filename=filename
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

    def test_spc2hdf(self):
        import mfm.io.tttr
        import glob
        import tempfile

        filetype = "bh132"
        #file = tempfile.NamedTemporaryFile(
        #   suffix='.photon.h5'
        #)
        _, filename = tempfile.mkstemp(
            suffix='.photon.h5'
        )
        output = filename
        spc_files = glob.glob("./test/data/tttr/BH/BH_SPC132.spc")
        h5 = mfm.io.tttr.spc2hdf(
            spc_files,
            routine_name=filetype,
            filename=output
        )
        h5.close()

    def test_photons(self):
        import mfm.io
        directory = './data/tttr/'
        test_data = [
            {
                "routine": "bh132",
                "files": glob.glob(directory + '/BH/132/*.spc'),
                "n_tac": 4095,
                "measurement_time": 62.328307194000004,
                "n_rout": 255,
                "n_photons": 183656,
                "mt_clk": 1.35e-05,
                "dt": 3.2967032967032967e-09
            },
            # {
            #     "routine": "bh132",
            #     "files": glob.glob(directory + '/BH_SPC630_256.spc'),
            #     "n_tac": 4095,
            #     "measurement_time": 62.328307194000004,
            #     "n_rout": 255,
            #     "n_photons": 183656,
            #     "mt_clk": 1.35e-05,
            #     "dt": 3.2967032967032967e-09
            # },
        ]
        for d in test_data:
            photons = mfm.io.photons.Photons(
                d["files"],
                reading_routine=d["routine"]
            )
            print(d["files"])
            self.assertListEqual(
                photons.filenames,
                d["files"]
            )
            self.assertAlmostEqual(
                photons.measurement_time,
                d["measurement_time"]
            )
            self.assertEqual(
                photons.n_rout,
                d["n_rout"]
            )
            self.assertEqual(
                photons.n_tac,
                d["n_tac"]
            )
            self.assertEqual(
                photons.shape[0],
                d["n_photons"]
            )
            self.assertAlmostEqual(
                photons.mt_clk,
                d["mt_clk"]
            )
            self.assertAlmostEqual(
                photons.dt,
                d["dt"]
            )

    # Removed for now because mmcif does not exist for Windows
    # def test_mmcif_read(self):
    #     import mmcif.io.PdbxReader
    #     import mfm.io.zipped
    #     filename = "./data/atomic_coordinates/mmcif/1ffk.cif.gz"
    #
    #     data = []
    #     with mfm.io.zipped.open_maybe_zipped(
    #             filename=filename,
    #             mode='r'
    #     ) as fp:
    #         reader = mmcif.io.PdbxReader.PdbxReader(fp)
    #         reader.read(data)
    #     #mfm.io.coordinates.keys
    #     #atoms = data[0]['atom_site']
    #


if __name__ == '__main__':
    unittest.main()

