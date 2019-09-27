import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mfm
import mfm.base


class Tests(unittest.TestCase):

    def test_base_init(self):
        b1 = mfm.base.Base()
        self.assertEqual(b1.name, 'Base')

        b2 = mfm.base.Base(name='B')
        self.assertEqual(b2.name, 'B')

        b3 = mfm.base.Base(
            name='B',
            test_parameter='aa'
        )
        self.assertEqual(b3.test_parameter, 'aa')

    def test_copy(self):
        import copy
        b1 = mfm.base.Base(name='B')
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
        b1 = mfm.base.Base(**d)
        self.assertEqual(b1.to_dict(), d)
        b2 = mfm.base.Base()
        b2.from_dict(d)
        self.assertEqual(b1.to_dict(), b2.to_dict())

    def test_uuid(self):
        b1 = mfm.base.Base(value=2.0)
        b2 = mfm.base.Base(value=2.0)
        self.assertIsNot(
            b1.unique_identifier,
            b2.unique_identifier
        )
        b2.from_dict(b1.to_dict())

        self.assertEqual(
            b1.unique_identifier,
            b2.unique_identifier
        )

    def test_yaml(self):
        yaml_string = 'name: B\ntest_parameter: aa\nunique_identifier: e7f0eb02-cbab-4aa3-abf2-799aebe96a09\nverbose: false\n'
        b1 = mfm.base.Base()
        b1.from_yaml(yaml_string, verbose=False)

        b2 = mfm.base.Base()
        b2.from_yaml(yaml_string)
        self.assertEqual(b1.to_dict(), b2.to_dict())

        b3 = mfm.base.Base()
        b3.from_yaml(b2.to_yaml())
        self.assertEqual(b2.to_dict(), b3.to_dict())

    def test_json(self):
        json_string = '{\n    "name": "B",\n    "test_parameter": "aa",\n    "unique_identifier": "e7f0eb02-cbab-4aa3-abf2-799aebe96a09",\n    "verbose": false\n}'
        b1 = mfm.base.Base()
        b1.from_json(json_string, verbose=False)

        b2 = mfm.base.Base()
        b2.from_json(json_string)
        self.assertEqual(b1.to_dict(), b2.to_dict())

        b3 = mfm.base.Base()
        b3.from_json(b2.to_json())
        self.assertEqual(b2.to_dict(), b3.to_dict())

    def test_save_load(self):
        import tempfile
        d = {
            'name': 'B',
            'test_parameter': 'aa',
            'verbose': False,
            'unique_identifier': 'e7f0eb02-cbab-4aa3-abf2-799aebe96a09'
        }

        # JSON File
        file = tempfile.NamedTemporaryFile(
            suffix='.json'
        )
        filename = file.name
        b1 = mfm.base.Base(**d)
        b1.save(
            filename=filename,
            file_type='json'
        )
        b2 = mfm.base.Base()
        b2.load(
            filename=filename,
            file_type='json'
        )
        self.assertEqual(b2.to_dict(), b1.to_dict())

        # YAML File
        file = tempfile.NamedTemporaryFile(
            suffix='.yaml'
        )
        filename = file.name
        b1 = mfm.base.Base(**d)
        b1.save(
            filename=filename,
            file_type='yaml'
        )
        b2 = mfm.base.Base()
        b2.load(
            filename=filename,
            file_type='yaml'
        )
        self.assertEqual(b2.to_dict(), b1.to_dict())

    def test_clean_string(self):
        s1 = "ldldöö_ddd   dd**"
        s2 = "ldldoo_ddd_dd"
        self.assertEqual(
            mfm.base.clean_string(s1),
            s2
        )


if __name__ == '__main__':
    unittest.main()
