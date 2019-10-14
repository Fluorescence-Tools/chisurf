import utils
import os
import unittest

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../chisurf/')
)
utils.set_search_paths(TOPDIR)

import mfm.decorators


class Tests(unittest.TestCase):

    def test_register(self):
        @mfm.decorators.register
        class A1():
            pass

        @mfm.decorators.register
        class B():
            pass

        @mfm.decorators.register
        class A2(A1):
            pass

        class A3(A1):
            pass

        a1_1 = A1()
        a1_2 = A1()
        a2_1 = A2()
        a3_1 = A3()
        b = B()
        self.assertEqual(
            a1_2 in a1_1.get_instances(),
            True
        )
        self.assertEqual(
            a2_1 in a1_1.get_instances(),
            False
        )
        self.assertEqual(
            a3_1 in a1_1.get_instances(),
            True
        )
        self.assertEqual(
            b in a1_1.get_instances(),
            False
        )

    def test_set_module(self):
        name = 'test_module_name'
        @mfm.decorators.set_module(name)
        def example():
            pass
        self.assertEqual(
            example.__module__,
            name
        )
