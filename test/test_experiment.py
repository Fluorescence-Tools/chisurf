import utils
import os
import unittest
import numpy as np
import tempfile

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)

import chisurf.experiments
import chisurf.models
import chisurf.fio


class Tests(unittest.TestCase):

    def test_experiment(self):
        experiment = chisurf.experiments.experiment.Experiment(
            name="AAA"
        )
        self.assertEqual(
            experiment.name,
            "AAA"
        )
        experiment.add_model_class(
            chisurf.models.model.Model
        )
        self.assertEqual(
            chisurf.models.model.Model in experiment.model_classes,
            True
        )
        experiment.add_model_classes(
            [
                chisurf.models.model.Model
            ]
        )

        # Models are unique
        experiment.add_model_class(
            chisurf.models.model.Model
        )
        self.assertEqual(
            len(experiment.model_classes),
            1
        )
        self.assertListEqual(
            experiment.model_names,
            ['Model name not available']
        )

        experiment_reader = chisurf.experiments.reader.ExperimentReader(
            name="ExperimentReaderName_A",
            experiment=experiment
        )
        experiment.add_reader(
            experiment_reader
        )

        experiment.add_readers(
            [
                (experiment_reader, None)
            ]
        )

        self.assertListEqual(
            experiment.get_reader_names(),
            ["ExperimentReaderName_A"]
        )

    def test_experimental_data(self):
        experiment = chisurf.experiments.experiment.Experiment(
            name="Experiment Type"
        )
        data_reader = chisurf.experiments.reader.ExperimentReader(
            experiment=experiment
        )
        experiment.add_reader(
            data_reader
        )
        a = np.arange(100)
        experimental_data = chisurf.experiments.data.ExperimentalData(
            experiment=experiment,
            data_reader=data_reader,
            embed_data=True,
            data=bytes(a)
        )
        self.assertEqual(
            experimental_data.filename,
            'None'
        )
        self.assertEqual(
            experimental_data.experiment,
            experiment
        )
        self.assertEqual(
            experimental_data.data_reader,
            data_reader
        )
        self.assertListEqual(
            experiment.reader_names,
            [data_reader.name]
        )
        #file = tempfile.NamedTemporaryFile(
        #    suffix='.npy'
        #)
        #filename = file.name
        _, filename = tempfile.mkstemp(
            suffix='.npy'
        )
        np.save(
            file=filename,
            arr=a
        )
        experimental_data.filename = filename
        self.assertEqual(
            experimental_data.filename,
            filename
        )
        # TODO: test to_dict and to_json

    def test_ExperimentReaderController(self):
        experiment = chisurf.experiments.experiment.Experiment(
            name="TestExperiment"
        )
        experiment_reader = chisurf.experiments.reader.ExperimentReader(
            experiment=experiment
        )
        ec = chisurf.experiments.reader.ExperimentReaderController(
            experiment_reader=experiment_reader
        )
        ec.add_call(
            'read',
            experiment_reader.read,  # this calls chisurf.base.load
            {
                'filename': None
            }
        )
        ec.call('read')

    def test_TCSPCReader(self):
        filename = "./test/data/tcspc/ibh_sample/Decay_577D.txt"
        ex = chisurf.experiments.experiment.Experiment(
            'TCSPC'
        )
        dt = 0.0141
        g1 = chisurf.experiments.tcspc.TCSPCReader(
            experiment=ex,
            skiprows=8,
            rebin=(1, 8),
            dt=dt
        )
        g2 = chisurf.experiments.tcspc.TCSPCReader(
            experiment=ex
        )
        g2.from_dict(
            g1.to_dict()
        )
        self.assertDictEqual(
            g1.to_dict(),
            g2.to_dict()
        )

        # Test binning
        d1 = g1.read(
            filename=filename,
        )
        self.assertEqual(
            len(d1.x),
            512
        )

        g1.rebin = (1, 1)
        d2 = g1.read(
            filename=filename
        )
        self.assertEqual(
            len(d2.x),
            4096
        )

    def test_FCS_Reader(self):
        import chisurf.experiments
        filename = './test/data/fcs/Kristine/Kristine_with_error.cor'
        ex = chisurf.experiments.experiment.Experiment(
            'FCS'
        )
        g1 = chisurf.experiments.fcs.FCS(
            experiment=ex,
            experiment_reader='Kristine'
        )
        fcs_curve = g1.read(
            filename=filename
        )
        self.assertEqual(
            fcs_curve.name,
            filename
        )
        # there is one FCS curve in the Kristine file
        self.assertEqual(
            len(fcs_curve),
            1
        )
        self.assertEqual(
            len(fcs_curve.x),
            207
        )

    def test_DataCurve(self):
        x = np.linspace(0, np.pi * 2.0)
        y = np.sin(x)
        ex = np.zeros_like(x)
        ey = np.ones_like(y)
        data = np.vstack([x, y, ex, ey])
        csv_io = chisurf.fio.ascii.Csv(
            use_header=False
        )
        #file = tempfile.NamedTemporaryFile(
        #    suffix='.txt'
        #)
        #filename = file.name
        _, filename = tempfile.mkstemp(
            suffix='.txt'
        )

        csv_io.save(
            data=data,
            filename=filename
        )
        d = chisurf.experiments.data.DataCurve(
            *data
        )

        #file = tempfile.NamedTemporaryFile(
        #    suffix='.txt'
        #)
        #filename = file.name
        _, filename = tempfile.mkstemp(
            suffix='.txt'
        )

        d.save(
            filename=filename,
            file_type='txt'
        )
        self.assertEqual(
            d.filename,
            filename
        )

        reference_string = """
length  : 50
x	y	error-x	error-y
0.000e+00   	0.000e+00   	0.000e+00   	1.000e+00   	
1.282e-01   	1.279e-01   	0.000e+00   	1.000e+00   	
....
6.027e+00   	-2.537e-01  	0.000e+00   	1.000e+00  """
        self.assertEqual(
            reference_string in d.__str__(),
            True
        )

        d2 = chisurf.experiments.data.DataCurve()
        d2.load(
            filename=filename,
            skiprows=0
        )
        self.assertEqual(
            np.allclose(
                np.hstack(d[:]),
                np.hstack(d2[:]),
            ),
            True
        )

        d3 = chisurf.experiments.data.DataCurve()
        d3.set_data(*d2.data)
        self.assertEqual(
            np.allclose(
                np.hstack(d[:]),
                np.hstack(d3[:]),
            ),
            True
        )

        d4 = chisurf.experiments.data.DataCurve()
        d4.data = d3.data
        self.assertEqual(
            np.allclose(
                np.hstack(d[:]),
                np.hstack(d4[:]),
            ),
            True
        )

        # d5 = experiments.data.DataCurve(
        #     filename=file.name
        # )
        # self.assertEqual(
        #     np.allclose(
        #         np.hstack(d[:]),
        #         np.hstack(d5[:]),
        #     ),
        #     True
        # )
        #

