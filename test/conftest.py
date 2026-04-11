import sys
import pathlib

# Add the project root to sys.path so 'chisurf' can be imported in all tests
TOPDIR = pathlib.Path(__file__).parent.parent
if str(TOPDIR) not in sys.path:
    sys.path.insert(0, str(TOPDIR))

# Add 'test' directory to sys.path so 'utils' can be imported by tests in subfolders
TESTDIR = TOPDIR / "test"
if str(TESTDIR) not in sys.path:
    sys.path.insert(0, str(TESTDIR))

# Import utils and setup paths (backward compatibility for tests that still use it)
try:
    import utils
    utils.set_search_paths(TOPDIR)
except ImportError:
    pass
