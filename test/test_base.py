import utils
import os
import unittest

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
utils.set_search_paths(TOPDIR)

import tempfile
import copy
import numpy as np

import chisurf.base
import chisurf.experiments
import chisurf.data


def get_data_values(
        c_value: float = 3.1,
        a_value: float = 1.2,
        n_points: int = 32
):
    x_data = np.linspace(0, 32, n_points)
    y_data = c_value + a_value * x_data ** 2.0
    return x_data, y_data


class Tests(unittest.TestCase):

    def test_base_init(self):
        b1 = chisurf.base.Base()
        self.assertEqual(b1.name, 'Base')
        b2 = chisurf.base.Base(name='B')
        self.assertEqual(b2.name, 'B')
        b3 = chisurf.base.Base(
            name='B',
            test_parameter='aa'
        )
        self.assertEqual(
            b3.test_parameter,
            'aa'
        )
        test_name = "tes"
        b3.name = test_name
        self.assertEqual(
            b3.name,
            test_name
        )

    def test_base_copy(self):
        b1 = chisurf.base.Base(name='B')
        b2 = copy.copy(b1)
        self.assertNotEqual(
            b1.unique_identifier,
            b2.unique_identifier,
        )
        b1.unique_identifier = b2.unique_identifier
        self.assertDictEqual(
            b1.to_dict(),
            b2.to_dict()
        )

    def test_base_dict(self):
        d = {
            'name': 'B',
            'test_parameter': 'aa',
            'verbose': False,
            'unique_identifier': 'e7f0eb02-cbab-4aa3-abf2-799aebe96a09'
        }
        b1 = chisurf.base.Base(**d)
        self.assertEqual(b1.to_dict(), d)
        b2 = chisurf.base.Base()
        b2.from_dict(d)
        self.assertEqual(b1.to_dict(), b2.to_dict())

    def test_base_uuid(self):
        b1 = chisurf.base.Base(value=2.0)
        b2 = chisurf.base.Base(value=2.0)
        self.assertIsNot(
            b1.unique_identifier,
            b2.unique_identifier
        )
        b2.from_dict(b1.to_dict())

        self.assertEqual(
            b1.unique_identifier,
            b2.unique_identifier
        )

    def test_base_yaml(self):
        yaml_string = 'name: B\ntest_parameter: aa\nunique_identifier: e7f0eb02-cbab-4aa3-abf2-799aebe96a09\nverbose: false\n'
        b1 = chisurf.base.Base()
        b1.from_yaml(yaml_string, verbose=False)

        b2 = chisurf.base.Base()
        b2.from_yaml(yaml_string)
        self.assertEqual(b1.to_dict(), b2.to_dict())

        b3 = chisurf.base.Base()
        b3.from_yaml(b2.to_yaml())
        self.assertEqual(b2.to_dict(), b3.to_dict())
        # test from_yaml
        b4 = chisurf.base.Base()
        b4.from_yaml(
            yaml_string=b2.to_yaml(),
            verbose=True
        )
        self.assertEqual(
            b4.to_dict(),
            b2.to_dict()
        )

    def test_json(self):
        json_string = '{\n    "name": "B",\n    "test_parameter": "aa",\n    "unique_identifier": "e7f0eb02-cbab-4aa3-abf2-799aebe96a09",\n    "verbose": false\n}'
        b1 = chisurf.base.Base()
        b1.from_json(json_string, verbose=False)

        b2 = chisurf.base.Base()
        b2.from_json(json_string)
        self.assertEqual(b1.to_dict(), b2.to_dict())

        b3 = chisurf.base.Base()
        b3.from_json(b2.to_json())
        self.assertEqual(b2.to_dict(), b3.to_dict())
        # test from_json
        b4 = chisurf.base.Base()
        b4.from_json(
            json_string=b2.to_json(),
            verbose=True
        )
        self.assertEqual(
            b4.to_dict(),
            b2.to_dict()
        )

    def test_base_save_load(self):
        import tempfile
        d = {
            'name': 'B',
            'test_parameter': 'aa',
            'verbose': False,
            'unique_identifier': 'e7f0eb02-cbab-4aa3-abf2-799aebe96a09'
        }

        # JSON File
        #file = tempfile.NamedTemporaryFile(
        #    suffix='.json'
        #)
        #filename = file.name
        _, filename = tempfile.mkstemp(
            suffix='.json'
        )

        b1 = chisurf.base.Base(**d)
        b1.save(
            filename=filename,
            file_type='json',
            verbose=True
        )
        b2 = chisurf.base.Base()
        b2.load(
            filename=filename,
            file_type='json'
        )
        self.assertEqual(
            b2.to_dict(),
            b1.to_dict()
        )

        # YAML File
        #file = tempfile.NamedTemporaryFile(
        #    suffix='.yaml'
        #)
        #filename = file.name
        _, filename = tempfile.mkstemp(
            suffix='.yaml'
        )

        b1 = chisurf.base.Base(**d)
        b1.save(
            filename=filename,
            file_type='yaml'
        )
        b2 = chisurf.base.Base()
        b2.load(
            filename=filename,
            file_type='yaml'
        )
        self.assertEqual(
            b2.to_dict(),
            b1.to_dict()
        )
        b2.load(
            filename="not a file",
            file_type='yaml'
        )

    def test_clean_string(self):
        s1 = "ldldöö_ddd   dd**"
        s2 = "ldldoo_ddd_dd"
        self.assertEqual(
            chisurf.base.clean_string(s1),
            s2
        )

    def test_data(self):
        # write some random data
        a = np.random.random(1000)
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
        d = chisurf.base.Data(
            filename=filename,
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

        d = chisurf.base.Data(
            filename=filename,
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
        data = chisurf.data.DataCurve(
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
        data_group = chisurf.data.DataGroup(
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


if __name__ == '__main__':
    unittest.main()
