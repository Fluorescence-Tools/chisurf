import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mfm


class Tests(unittest.TestCase):

    def test_get_instances(self):
        p1 = mfm.parameter.Parameter()
        self.assertEqual(
            len(list(p1.get_instances())), 1
        )
        self.assertEqual(p1 in p1.get_instances(), True)

        p2 = mfm.parameter.Parameter()
        self.assertEqual(
            len(list(p1.get_instances())), 2
        )
        self.assertEqual(p2 in p1.get_instances(), True)

    def test_create(self):
        p1 = mfm.parameter.Parameter()
        p1.value = 2.0
        self.assertEqual(p1.value, 2.0)

        p2 = mfm.parameter.Parameter(value=2.0)
        self.assertEqual(p2.value, 2.0)

    def test_equality(self):
        p1 = mfm.parameter.Parameter(value=2.0)
        p2 = mfm.parameter.Parameter(value=2.0)
        self.assertEqual(p1, p2)
        self.assertIsNot(p1, p2)

    def test_arithmetics(self):
        p1 = mfm.parameter.Parameter(value=2.0)
        p2 = mfm.parameter.Parameter(value=3.0)

        p3 = p1 + p2
        self.assertEqual(p3.value, 5.0)

        p3 = p1 - p2
        self.assertEqual(p3.value, -1.0)

        p3 = p1 * p2
        self.assertEqual(p3.value, 6.0)

        p3 = p1 / p2
        self.assertEqual(p3.value, 2. / 3.)

        p3 = p1 // p2
        self.assertEqual(p3.value, 2. // 3.)

        p3 = p1 % p2
        self.assertEqual(p3.value, 2. % 3.)

        p3 = p1 ** p2
        self.assertEqual(p3.value, 2. ** 3.)

    def test_linking(self):
        p1 = mfm.parameter.Parameter(value=2.0)
        p2 = mfm.parameter.Parameter(value=3.0)
        self.assertEqual(p1.value, 2.0)
        self.assertEqual(p2.value, 3.0)

        self.assertEqual(p2.is_linked, False)

        p2.link = p1
        self.assertEqual(p2.value, 2.0)
        self.assertEqual(p2.is_linked, True)

        # The original value is overwritten with the
        # linked value once the parameters are unlinked
        p2.link = None
        self.assertEqual(p2.value, 2.0)
        self.assertEqual(p2.is_linked, False)

        p2.value = 3
        self.assertEqual(p2.value, 3.0)

    def test_bounds(self):
        p1 = mfm.parameter.Parameter(
            value=2.0,
            bounds_on=True,
            lb=1.,
            ub=2.5
        )
        self.assertEqual(p1.value, 2.0)
        p1.value = 5.0
        self.assertEqual(p1.value, 2.5)

        p1.bounds_on = False
        self.assertEqual(p1.value, 5.0)

    def test_rep_str(self):
        p1 = mfm.parameter.Parameter(22)
        p2 = mfm.parameter.Parameter(11)

    def test_dict(self):
        d1 = {
            'value': 2.0,
            'bounds_on': True,
            'lb': 1.,
            'ub': 2.5,
            'unique_identifier': 'b671b0b3-3009-42df-824a-6d690c2b3e54'
        }
        p1 = mfm.parameter.Parameter(**d1)
        d3 = {
            'name': 'Parameter',
            'verbose': False,
            'unique_identifier': 'b671b0b3-3009-42df-824a-6d690c2b3e54',
            '_bounds_on': True,
            '_link': None,
            '_value': 2.0,
            '_lb': 1.0,
            '_ub': 2.5
        }
        self.assertEqual(
            p1.to_dict(),
            d3
        )

    def test_save_load(self):

        import tempfile

        file = tempfile.NamedTemporaryFile(
            suffix='.json'
        )
        filename = file.name
        p1 = mfm.parameter.Parameter(
            value=2.0,
            bounds_on=True,
            lb=1.,
            ub=2.5
        )
        p1.save(
            filename,
            file_type='json'
        )

        p2 = mfm.parameter.Parameter()
        p2.load(
            filename=filename,
            file_type='json'
        )

    def test_parameter_group(self):
        p1 = mfm.parameter.Parameter(22)
        p2 = mfm.parameter.Parameter(11)
        group_name = 'Parameter Gruppe'
        pg = mfm.parameter.ParameterGroup(
            name=group_name
        )
        self.assertEqual(
            pg.name,
            group_name
        )
        pg.append(p1)
        pg.append(p2)
        self.assertEqual(
            pg.values,
            [22, 11]
        )


if __name__ == '__main__':
    unittest.main()
