import utils
import os
import doctest

TOPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
utils.set_search_paths(TOPDIR)

import mfm.structure.structure


def test_structure_structure_module():
    assert doctest.testmod(mfm.structure.structure, raise_on_error=True)


if __name__ == '__main__':
    unittest.main()
