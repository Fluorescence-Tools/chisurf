import utils
import os
import unittest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)


class Tests(unittest.TestCase):

    def test_mfm_import(self):
        import mfm
