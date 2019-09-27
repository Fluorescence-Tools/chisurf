import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import numpy as np
import tempfile
import copy

import mfm
import mfm.base
import mfm.experiments


def get_data_values(
        c_value: float = 3.1,
        a_value: float = 1.2,
        n_points: int = 32
):
    x_data = np.linspace(0, 32, n_points)
    y_data = c_value + a_value * x_data ** 2.0
    return x_data, y_data


class Tests(unittest.TestCase):

    def test_data(self):
        # write some random data
        a = np.random.random(1000)
        file = tempfile.NamedTemporaryFile(
            suffix='.npy'
        )
        np.save(
            file=file.name,
            arr=a
        )
        d = mfm.base.Data(
            filename=file.name,
            embed_data=True
        )
        self.assertEqual(
            len(d.data),
            8128
        )
        self.assertEqual(
            d.embed_data,
            True
        )

        d = mfm.base.Data(
            filename=file.name,
            embed_data=False
        )
        self.assertEqual(
            len(d.data),
            0
        )

        d.embed_data = False
        self.assertEqual(
            d.embed_data,
            False
        )
        self.assertEqual(
            d.data,
            None
        )

    def test_data_group(self):
        a_value = 1.2
        c_value = 3.1
        x_data, y_data = get_data_values(
            a_value=a_value,
            c_value=c_value
        )
        data = mfm.experiments.data.DataCurve(
            x=x_data,
            y=y_data,
            ey=np.ones_like(y_data)
        )
        data2 = copy.copy(data)
        self.assertEqual(
            np.allclose(
                data2.y,
                data.y
            ),
            True
        )
        self.assertEqual(
            np.allclose(
                data2.x,
                data.x
            ),
            True
        )
        data_group = mfm.experiments.data.DataGroup(
            [data, data2]
        )
        data_group.current_dataset = 0
        self.assertIs(
            data_group.current_dataset,
            data
        )
        data_group.current_dataset = 1
        self.assertIs(
            data_group.current_dataset,
            data2
        )

