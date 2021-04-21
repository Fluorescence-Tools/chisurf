import utils
import os
import unittest
import tempfile
import numpy as np
import scikit_fluorescence.io.zipped

import chisurf.fio.fluorescence.fcs
import chisurf.fio.fluorescence.tcspc

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)


import chisurf.fio.structure.coordinates


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
        chisurf.fio.ascii.save_xy(
            filename=filename,
            x=x,
            y=y,
            fmt="%f\t%f\n",
            header_string="x\ty\n"
        )
        x2, y2 = chisurf.fio.ascii.load_xy(
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
        chisurf.fio.ascii.save_xy(
            filename=filename,
            x=reference_x,
            y=reference_y,
            fmt="%f\t%f\n",
            header_string="x\ty\n"
        )

        # CSV class
        csv = chisurf.fio.ascii.Csv(
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

        chisurf.fio.ascii.save_xy(
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
        s = chisurf.fio.structure.coordinates.fetch_pdb_string(pdb_id)
        self.assertEqual(
            'HEADER    HYDROLASE/HYDROLASE SUBSTRATE           27-OCT-93   148L              \nTITLE     A COVALEN',
            s[:100]
        )

    def test_parse_string_pdb(self):
        pdb_id = "148L"
        s = chisurf.fio.structure.coordinates.fetch_pdb_string(pdb_id)
        atoms = chisurf.fio.structure.coordinates.parse_string_pdb(s)
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
        with scikit_fluorescence.io.zipped.open_maybe_zipped(
                filename=filename,
                mode='w'
        ) as fp:
            fp.write(
                chisurf.fio.structure.coordinates.fetch_pdb_string(pdb_id)
            )

        atoms = chisurf.fio.structure.coordinates.read(
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
        atoms = chisurf.fio.structure.coordinates.read(
            filename="None"
        )
        self.assertEqual(
            len(atoms),
            0
        )

    def test_spc2hdf(self):
        import chisurf.fio.fluorescence.tttr
        import glob
        import tempfile

        filetype = "bh132"
        _, filename = tempfile.mkstemp(
            suffix='.photon.h5'
        )
        output = filename
        spc_files = glob.glob("./test/data/tttr/BH/BH_SPC132.spc")
        h5 = chisurf.fio.fluorescence.tttr.spc2hdf(
            spc_files,
            routine_name=filetype,
            filename=output
        )
        h5.close()

    def test_photons(self):
        import glob
        import chisurf.fio
        directory = './test/data/tttr/'
        test_data = [
            {
                "routine": "bh132",
                "files": glob.glob(directory + '/BH/132/*.spc'),
                "n_tac": 4096,
                "measurement_time": 62.3288052934344,
                "n_photons": 183657,
                "mt_clk": 13.5e-09,
                "dt": 3.2967032967032967e-09
            },
            # {
            #     "routine": "hdf",
            #     "files": glob.glob(directory + '/HDF_TP/*.h5'),
            #     "n_tac": 4095,
            #     "measurement_time": 62.328307194000004,
            #     "n_photons": 183656,
            #     "mt_clk": 1.35e-05,
            #     "dt": 3.2967032967032967e-09
            # },
        ]
        for d in test_data:
            photons = chisurf.fio.fluorescence.photons.Photons(
                d["files"],
                reading_routine=d["routine"]
            )
            self.assertListEqual(
                photons.filenames,
                d["files"]
            )
            self.assertAlmostEqual(
                photons.measurement_time,
                d["measurement_time"]
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

    def test_read_tcspc_csv_jordi(self):
        import chisurf.fio.fluorescence
        filename = './test/data/tcspc/Jordi/H2O_8-0 ps_2048 ch.dat'
        dt = 0.032

        ref_vv = np.array(
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 1., 4., 2., 10., 6., 14., 13., 15., 8.,
             12., 7., 21., 16., 13., 15., 24., 17., 25., 21., 12.,
             22., 20., 29., 23., 20., 29., 25., 36., 47., 37., 48.,
             60., 44., 59., 86., 83., 106., 88., 122., 96., 126., 139.,
             181., 179., 214., 233., 243., 284., 341., 374., 401., 460., 509.,
             561.]
        )

        ref_vh = np.array(
            [
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 3., 5., 6., 10., 11., 10., 8., 15., 12., 13., 14., 17.,
                14., 12., 15., 23., 16., 23., 22., 23., 27.
            ]
        )

        # read vv from jordi
        decay_data_curve_vv = chisurf.fio.fluorescence.tcspc.read_tcspc_csv(
            filename=filename,
            skiprows=0,
            dt=dt,
            is_jordi=True,
            polarization='vv'
        )
        self.assertEqual(
            np.allclose(
                decay_data_curve_vv.y[200:300],
                ref_vv
            ),
            True
        )

        # read vh from jordi
        decay_data_curve_vh = chisurf.fio.fluorescence.tcspc.read_tcspc_csv(
            filename=filename,
            skiprows=0,
            dt=dt,
            is_jordi=True,
            polarization='vh'
        )

        self.assertEqual(
            np.allclose(
                decay_data_curve_vh.y[200:300],
                ref_vh
            ),
            True
        )

        # combines vv and vh and adjusts the noise of the combined curve (not poisson anymore)
        g_factor = 1.5
        decay_data_curve_vm = chisurf.fio.fluorescence.tcspc.read_tcspc_csv(
            filename=filename,
            skiprows=0,
            dt=dt,
            is_jordi=True,
            polarization='vm',
            g_factor=1.5
        )
        ref_vm = ref_vv + ref_vh * 2.0 * g_factor
        self.assertEqual(
            np.allclose(
                decay_data_curve_vm.y[200:300],
                ref_vm
            ),
            True
        )

        # reads vv/vh in a group
        decay_data_curve_vv_vh = chisurf.fio.fluorescence.tcspc.read_tcspc_csv(
            filename=filename,
            skiprows=0,
            dt=dt,
            is_jordi=True,
            polarization='vv/vh'
        )
        self.assertEqual(
            len(decay_data_curve_vv_vh),
            2
        )
        self.assertEqual(
            np.allclose(
                decay_data_curve_vv_vh[0].y[200:300],
                ref_vv
            ),
            True
        )
        self.assertEqual(
            np.allclose(
                decay_data_curve_vv_vh[1].y[200:300],
                ref_vh
            ),
            True
        )

    def test_read_fcs_kristine(self):
        import chisurf.fio.fluorescence
        ref_fcs = np.array(
            [4.21595667, 3.87717445, 3.67014087, 1.62239048, 3.6475554,
             3.91858119, 4.01645162, 3.95622368, 3.85835326, 3.97504493,
             3.92987397, 3.83953205, 3.92610974, 3.95622372, 4.04280141,
             3.89976003, 4.01645171, 3.74542593, 3.93740254, 3.85458912,
             3.92987408, 3.69649077, 3.99198419, 3.82635732, 3.80094868,
             3.82729844, 3.90258342, 3.89787814, 3.93740276, 3.77459909,
             3.8837623, 3.84423773, 3.82306389, 3.85506005, 3.80518383,
             3.86823503, 3.76612987, 3.72707586, 3.80236087, 3.7411919]
        )

        filename = './test/data/fcs/kristine/Kristine_with_error.cor'
        fcs_data_curve_1 = chisurf.fio.fluorescence.fcs.read_fcs(
            reader_name='kristine',
            filename=filename
        )
        self.assertEqual(
            np.allclose(
                fcs_data_curve_1.y[0:40],
                ref_fcs
            ),
            True
        )

        filename = './test/data/fcs/kristine/Kristine_without_error.cor'
        fcs_data_curve_2 = chisurf.fio.fluorescence.fcs.read_fcs(
            reader_name='kristine',
            filename=filename
        )
        self.assertEqual(
            np.allclose(
                fcs_data_curve_2.y[0:40],
                ref_fcs
            ),
            True
        )


    # Removed for now because mmcif does not exist for Windows
    # def test_mmcif_read(self):
    #     import mmcif.fio.PdbxReader
    #     import scikit_fluorescence.io.zipped
    #     filename = "./test/data/atomic_coordinates/mmcif/1ffk.cif.gz"
    #
    #     data = []
    #     with scikit_fluorescence.io.zipped.open_maybe_zipped(
    #             filename=filename,
    #             mode='r'
    #     ) as fp:
    #         reader = mmcif.fio.PdbxReader.PdbxReader(fp)
    #         reader.read(data)
    #     #chisurf.fio.structure.coordinates.keys
    #     #atoms = data[0]['atom_site']
    #


if __name__ == '__main__':
    unittest.main()

