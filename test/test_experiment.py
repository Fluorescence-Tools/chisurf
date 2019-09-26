import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import numpy as np
import tempfile
import mfm


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
            experiment.get_setup_names(),
            ["ExperimentReaderName_A"]
        )

    def test_experimental_data(self):
        experiment = mfm.experiments.experiment.Experiment(
            name="Experiment Type"
        )
        data_reader = mfm.experiments.reader.ExperimentReader(
            experiment=experiment
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
