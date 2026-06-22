import os
import sys
import pathlib
import tempfile

import pytest

# Add the project root to sys.path so 'chisurf' can be imported in all tests
TOPDIR = pathlib.Path(__file__).parent.parent
if str(TOPDIR) not in sys.path:
    sys.path.insert(0, str(TOPDIR))

# Add 'test' directory to sys.path so 'utils' can be imported by tests in subfolders
TESTDIR = TOPDIR / "test"
if str(TESTDIR) not in sys.path:
    sys.path.insert(0, str(TESTDIR))

# ---------------------------------------------------------------------------
# Hermetic test harness (PRD-18)
#
# Redirect chisurf's per-user state directory to a temporary location for the
# whole test session so no test can read or write the real ``~/.chisurf`` — in
# particular the sample database at ``~/.chisurf/flr/sample_management.db`` and
# the object store. This is set *before* any chisurf import resolves a path, via
# the ``CHISURF_SETTINGS_DIR`` override honoured by ``settings.path_utils.get_path``.
# ---------------------------------------------------------------------------

_REAL_SETTINGS_DIR = pathlib.Path.home() / ".chisurf"
_HERMETIC_SETTINGS_DIR = pathlib.Path(tempfile.mkdtemp(prefix="chisurf-test-settings-"))
os.environ["CHISURF_SETTINGS_DIR"] = str(_HERMETIC_SETTINGS_DIR)


@pytest.fixture(scope="session", autouse=True)
def _hermetic_settings_dir():
    """Ensure every test uses the temp settings dir, never the real ~/.chisurf."""
    os.environ["CHISURF_SETTINGS_DIR"] = str(_HERMETIC_SETTINGS_DIR)
    yield _HERMETIC_SETTINGS_DIR


@pytest.fixture(autouse=True)
def _guard_real_user_db():
    """Fail loudly if a test resolves chisurf state to the real ~/.chisurf."""
    from chisurf.core.settings.path_utils import get_path
    from chisurf.core.mfdb.database_resolver import (
        object_store_root,
        user_database_path,
    )

    real = _REAL_SETTINGS_DIR.resolve()
    assert get_path("settings").resolve() != real, (
        "Test resolved the REAL settings dir; CHISURF_SETTINGS_DIR not in effect."
    )
    assert real not in user_database_path().resolve().parents, (
        f"Test would use the REAL user database at {user_database_path()}."
    )
    assert real not in object_store_root().resolve().parents and object_store_root().resolve() != real, (
        f"Test would use the REAL object store at {object_store_root()}."
    )
    yield


# Import utils and setup paths (backward compatibility for tests that still use it)
try:
    import utils
    utils.set_search_paths(TOPDIR)
except ImportError:
    pass
