"""Hermetic test harness for the standalone MFDB package.

These tests import only :mod:`mfdb` — never ChiSurf. Isolation is achieved
through MFDB's own runtime config: ``MFDB_SETTINGS_DIR`` redirects the per-user
state directory (sample database + object store) to a throwaway location for the
whole session, so no test can read or write the real ``~/.chisurf``.

Run standalone from the package root:

    PYTHONPATH=src pytest            # or: pixi run test-mfdb
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import pytest

# Redirect MFDB state *before* any mfdb import resolves a path.
_REAL_SETTINGS_DIR = pathlib.Path.home() / ".chisurf"
_HERMETIC_SETTINGS_DIR = pathlib.Path(tempfile.mkdtemp(prefix="mfdb-test-settings-"))
os.environ["MFDB_SETTINGS_DIR"] = str(_HERMETIC_SETTINGS_DIR)


@pytest.fixture(scope="session", autouse=True)
def _hermetic_settings_dir():
    """Point MFDB at the temp settings dir and seed a fresh, current-schema DB.

    Pre-creating the user database means ``resolve_database_path`` finds it and
    does not copy any shipped curated source DB (older schema / demo data).
    """
    os.environ["MFDB_SETTINGS_DIR"] = str(_HERMETIC_SETTINGS_DIR)
    try:
        from mfdb.database_resolver import user_database_path
        from mfdb.repository import MFDatabase

        user_db = user_database_path()
        user_db.parent.mkdir(parents=True, exist_ok=True)
        if not user_db.exists():
            MFDatabase(str(user_db)).close()
    except Exception:
        pass
    yield _HERMETIC_SETTINGS_DIR


@pytest.fixture(autouse=True)
def _guard_real_user_db():
    """Fail loudly if a test resolves MFDB state to the real ~/.chisurf."""
    from mfdb.database_resolver import object_store_root, user_database_path

    real = _REAL_SETTINGS_DIR.resolve()
    assert real not in user_database_path().resolve().parents, (
        f"Test would use the REAL user database at {user_database_path()}."
    )
    assert (
        real not in object_store_root().resolve().parents
        and object_store_root().resolve() != real
    ), f"Test would use the REAL object store at {object_store_root()}."
    yield


@pytest.fixture(autouse=True)
def _reset_global_db():
    """Reset the process-global active DB between tests.

    ``result_registry`` caches the active database in a module global; a test
    that sets it must not leak that connection into later tests.
    """
    try:
        from mfdb.result_registry import set_global_db
    except Exception:
        set_global_db = None
    if set_global_db is not None:
        set_global_db(None)
    yield
    if set_global_db is not None:
        set_global_db(None)
