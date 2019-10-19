import utils
import os
import unittest

TOPDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../chisurf/')
)
utils.set_search_paths(TOPDIR)


class Tests(unittest.TestCase):

    def test_1(self):
        import chisurf

    def test_2(self):
        import chisurf.experiments

if __name__ == '__main__':
    unittest.main()
