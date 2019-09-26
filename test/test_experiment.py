import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import numpy as np
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
            name="ExperimentReaderName_A"
        )
        experiment.add_reader(
            experiment_reader
        )

