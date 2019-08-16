import utils
import os
import unittest
import sys
import json

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)


class Tests(unittest.TestCase):

    def test_mfm(self):
        import mfm


if __name__ == '__main__':
    unittest.main()