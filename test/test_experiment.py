import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import numpy as np
import tempfile
import mfm.experiments
import mfm.models
import mfm.io


class Tests(unittest.TestCase):

    def test_experiment(self):
        experiment = mfm.experiments.experiment.Experiment(
            name="AAA"
        )
        self.assertEqual(
            experiment.name,
            "AAA"
        )
        experiment.add_model_class(
            mfm.models.model.Model
        )
        self.assertEqual(
            mfm.models.model.Model in experiment.model_classes,
            True
        )
        experiment.add_model_classes(
            [mfm.models.model.Model]
        )

        # Models are unique
        experiment.add_model_class(
            mfm.models.model.Model
        )
        self.assertEqual(
            len(experiment.model_classes),
            1
        )
        self.assertListEqual(
            experiment.model_names,
            ['Model name not available']
        )

        experiment_reader = mfm.experiments.reader.ExperimentReader(
            name="ExperimentReaderName_A",
            experiment=experiment
        )
        experiment.add_reader(
            experiment_reader
        )

        experiment.add_readers(
            [experiment_reader]
        )

        self.assertListEqual(
            experiment.get_reader_names(),
            ["ExperimentReaderName_A"]
        )

    def test_experimental_data(self):
        experiment = mfm.experiments.experiment.Experiment(
            name="Experiment Type"
        )
        data_reader = mfm.experiments.reader.ExperimentReader(
            experiment=experiment
        )
        experiment.add_reader(
            data_reader
        )
        a = np.arange(100)
        experimental_data = mfm.experiments.data.ExperimentalData(
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
        file = tempfile.NamedTemporaryFile(
            suffix='.npy'
        )
        np.save(
            file=file.name,
            arr=a
        )
        experimental_data.filename = file.name
        self.assertEqual(
            experimental_data.filename,
            file.name
        )
        # TODO: test to_dict and to_json

    def test_ExperimentReaderController(self):
        import mfm.experiments
        experiment = mfm.experiments.experiment.Experiment(
            name="TestExperiment"
        )
        experiment_reader = mfm.experiments.reader.ExperimentReader(
            experiment=experiment
        )
        ec = mfm.experiments.reader.ExperimentReaderController(
            experiment_reader=experiment_reader
        )
        ec.add_call(
            'read',
            experiment_reader.read,  # this calls mfm.Base.load
            {
                'filename': None
            }
        )
        ec.call('read')

    def test_CsvTCSPC(self):
        g1 = mfm.experiments.tcspc.tcspc.CsvTCSPC()
        g2 = mfm.experiments.tcspc.tcspc.CsvTCSPC()
        g2.from_dict(
            g1.to_dict()
        )
        self.assertDictEqual(
            g1.to_dict(),
            g2.to_dict()
        )


    def test_DataCurve(self):
        x = np.linspace(0, np.pi * 2.0)
        y = np.sin(x)
        ex = np.zeros_like(x)
        ey = np.ones_like(y)
        data = np.vstack([x, y, ex, ey])
        csv_io = mfm.io.ascii.Csv(
            use_header=False
        )
        file = tempfile.NamedTemporaryFile(
            suffix='.txt'
        )
        csv_io.save(
            data=data,
            filename=file.name
        )
        d = mfm.experiments.data.DataCurve(
            *data
        )
        file = tempfile.NamedTemporaryFile(
            suffix='.txt'
        )
        d.save(
            filename=file.name,
            file_type='txt'
        )
        self.assertEqual(
            d.filename,
            file.name
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

        d2 = mfm.experiments.data.DataCurve()
        d2.load(
            filename=file.name,
            skiprows=0
        )
        self.assertEqual(
            np.allclose(
                np.hstack(d[:]),
                np.hstack(d2[:]),
            ),
            True
        )

        d3 = mfm.experiments.data.DataCurve()
        d3.set_data(*d2.data)
        self.assertEqual(
            np.allclose(
                np.hstack(d[:]),
                np.hstack(d3[:]),
            ),
            True
        )

        d4 = mfm.experiments.data.DataCurve()
        d4.data = d3.data
        self.assertEqual(
            np.allclose(
                np.hstack(d[:]),
                np.hstack(d4[:]),
            ),
            True
        )

        # d5 = mfm.experiments.data.DataCurve(
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

