"""PRD-17: one canonical identity/session resolver.

Writes (which stamp ownership) and reads (which scope "mine") must resolve the
same user. These tests lock in that the consolidated resolver is the single
source and that the legacy wrappers delegate to it.
"""

from __future__ import annotations

from chisurf.core.mfdb.security.session import (
    DEFAULT_USER_ID,
    SessionContext,
    configured_default_user_id,
    resolve_active_user_id,
    resolve_session,
)


def test_no_auth_resolves_configured_default():
    """With no auth payload, the resolver returns the configured default user."""
    assert resolve_active_user_id() == configured_default_user_id()


def test_default_when_unconfigured(monkeypatch):
    """Falls back to DEFAULT_USER_ID when no default is configured."""
    import chisurf.core.settings as settings

    monkeypatch.setitem(settings.cs_settings, "mfdb", {})
    assert configured_default_user_id() == DEFAULT_USER_ID
    assert resolve_active_user_id() == DEFAULT_USER_ID


def test_configured_default_is_honoured(monkeypatch):
    """A configured mfdb.default_user_id is what reads and writes both use."""
    import chisurf.core.settings as settings

    monkeypatch.setitem(settings.cs_settings, "mfdb", {"default_user_id": "alice"})
    assert configured_default_user_id() == "alice"
    assert resolve_active_user_id() == "alice"


def test_resolve_session_carries_user_id(monkeypatch):
    """resolve_session builds a context whose user_id is the canonical user."""
    import chisurf.core.settings as settings

    monkeypatch.setitem(settings.cs_settings, "mfdb", {"default_user_id": "bob"})
    ctx = resolve_session()
    assert isinstance(ctx, SessionContext)
    assert ctx.user_id == "bob"


def test_result_registry_wrapper_delegates(monkeypatch):
    """The legacy result_registry resolver returns the canonical identity."""
    import chisurf.core.settings as settings
    from chisurf.core.mfdb.provenance.result_registry import _resolve_active_user_id

    monkeypatch.setitem(settings.cs_settings, "mfdb", {"default_user_id": "carol"})
    assert _resolve_active_user_id() == resolve_active_user_id() == "carol"


def test_injected_session_stamps_owner(tmp_path):
    """A SessionContext threaded into register_* stamps that user as owner,
    regardless of the configured default — the PRD-17 injection path."""
    import os

    from chisurf.core.mfdb.repository import MFDatabase
    from chisurf.core.mfdb.provenance.result_registry import register_raw_measurement, set_global_db

    db = MFDatabase(os.path.join(tmp_path, "owner.db"))
    try:
        f = tmp_path / "m.ptu"
        f.write_bytes(b"\x00\x01\x02")
        ctx = SessionContext(user_id="dave", db=db)
        artifact_id = register_raw_measurement(str(f), db=db, session=ctx)
        assert artifact_id
        owners = db.list_artifact_owners(artifact_id)
        assert "dave" in owners, f"expected dave in owners, got {owners}"
    finally:
        set_global_db(None)
        db.close()
