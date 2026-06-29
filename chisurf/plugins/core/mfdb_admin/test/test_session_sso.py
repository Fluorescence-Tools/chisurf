"""Process-wide MFDB session cache — single sign-on across surfaces.

The first login caches a token; re-opening mfdb-admin or the scraper's
"Add to MFDB" reuses it without asking for a password again.
"""
from __future__ import annotations

import pytest

from chisurf.plugins.core.mfdb_admin.gui import session as S


@pytest.fixture(autouse=True)
def _clean():
    S.clear_cached_session()
    yield
    S.clear_cached_session()


def test_cache_and_reuse_token_for_matching_endpoint():
    assert S.cached_token("127.0.0.1", 8765, 8766) is None
    S.cache_session("admin", "tok-123", "127.0.0.1", 8765, 8766)
    assert S.cached_user() == "admin"
    # reused for the same endpoint …
    assert S.cached_token("127.0.0.1", 8765, 8766) == "tok-123"
    # … not for a different host/port
    assert S.cached_token("10.0.0.9", 8765, 8766) is None
    assert S.cached_token("127.0.0.1", 9000, 9001) is None


def test_clear_session():
    S.cache_session("admin", "tok", "127.0.0.1", 8765, 8766)
    S.clear_cached_session()
    assert S.cached_token("127.0.0.1", 8765, 8766) is None
    assert S.cached_user() is None
