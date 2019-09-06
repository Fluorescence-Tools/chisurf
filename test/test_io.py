import utils
import os
import unittest
import sys
import json

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mfm.io


class Tests(unittest.TestCase):

    def test_fetch_pdb(self):
        pdb_id = "148L"
        mfm.io.pdb.fetch_pdb_string(pdb_id)

    def test_parse_string_pdb(self):
        pdb_id = "148L"
        s = mfm.io.pdb.fetch_pdb_string(pdb_id)
        mfm.io.pdb.parse_string_pdb(s)


if __name__ == '__main__':
    unittest.main()

