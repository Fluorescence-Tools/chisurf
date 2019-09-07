import utils
import os
import unittest
import sys
import json

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mfm


class Tests(unittest.TestCase):

    def test_create(self):
        p1 = mfm.parameter.Parameter()
        p1.value = 2.0


if __name__ == '__main__':
    unittest.main()
