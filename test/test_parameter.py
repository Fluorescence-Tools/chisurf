import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mfm


class Tests(unittest.TestCase):

    def test_get_instances(self):
        p1 = mfm.parameter.Parameter()
        instances = list(p1.get_instances())
        self.assertEqual(len(instances), 1)
        self.assertEqual(p1 in instances, True)

        p2 = mfm.parameter.Parameter()
        self.assertEqual(len(instances), 2)
        self.assertEqual(p2 in instances, True)

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

        p2.link = None
        self.assertEqual(p2.value, 3.0)
        self.assertEqual(p2.is_linked, False)

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


if __name__ == '__main__':
    unittest.main()
