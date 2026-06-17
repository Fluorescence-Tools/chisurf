from __future__ import annotations

import types


def test_keyring_session_token_round_trip(monkeypatch) -> None:
    """Session tokens round-trip through the optional keyring backend."""
    from chisurf.core.mfdb import credentials

    stored = {}

    def set_password(service, account, token):
        stored[(service, account)] = token

    def get_password(service, account):
        return stored.get((service, account))

    def delete_password(service, account):
        stored.pop((service, account), None)

    fake_keyring = types.SimpleNamespace(
        set_password=set_password,
        get_password=get_password,
        delete_password=delete_password,
    )
    monkeypatch.setattr(credentials, "_get_keyring", lambda: fake_keyring)

    assert credentials.store_session_token("127.0.0.1", 8765, "admin", "secret-token") is True
    assert credentials.load_session_token("127.0.0.1", 8765, "admin") == "secret-token"
    assert credentials.delete_session_token("127.0.0.1", 8765, "admin") is True
    assert credentials.load_session_token("127.0.0.1", 8765, "admin") is None


def test_runtime_session_token_round_trip() -> None:
    """Runtime session tokens are available without persistent storage."""
    from chisurf.core.mfdb import credentials

    credentials.delete_runtime_session_token("127.0.0.1", 8765, "admin")

    assert credentials.store_runtime_session_token("127.0.0.1", 8765, "admin", "runtime-token") is True
    assert credentials.load_runtime_session_token("127.0.0.1", 8765, "admin") == "runtime-token"
    assert credentials.delete_runtime_session_token("127.0.0.1", 8765, "admin") is True
    assert credentials.load_runtime_session_token("127.0.0.1", 8765, "admin") is None


def test_session_token_rename_moves_runtime_and_keyring_tokens(monkeypatch) -> None:
    """Renaming a user moves runtime and persisted session token keys."""
    from chisurf.core.mfdb import credentials

    stored = {}

    def set_password(service, account, token):
        stored[(service, account)] = token

    def get_password(service, account):
        return stored.get((service, account))

    def delete_password(service, account):
        stored.pop((service, account), None)

    fake_keyring = types.SimpleNamespace(
        set_password=set_password,
        get_password=get_password,
        delete_password=delete_password,
    )
    monkeypatch.setattr(credentials, "_get_keyring", lambda: fake_keyring)

    credentials.store_runtime_session_token("127.0.0.1", 8765, "old_user", "runtime-token")
    credentials.store_session_token("127.0.0.1", 8765, "old_user", "stored-token")

    assert credentials.rename_runtime_session_token("127.0.0.1", 8765, "old_user", "new_user") is True
    assert credentials.rename_session_token("127.0.0.1", 8765, "old_user", "new_user") is True
    assert credentials.load_runtime_session_token("127.0.0.1", 8765, "old_user") is None
    assert credentials.load_runtime_session_token("127.0.0.1", 8765, "new_user") == "runtime-token"
    assert credentials.load_session_token("127.0.0.1", 8765, "old_user") is None
    assert credentials.load_session_token("127.0.0.1", 8765, "new_user") == "stored-token"


def test_missing_credential_store_does_not_persist_plaintext(monkeypatch) -> None:
    """No plaintext fallback is used when no credential store is available."""
    from chisurf.core.mfdb import credentials

    monkeypatch.setattr(credentials, "_get_keyring", lambda: None)
    monkeypatch.setattr(credentials, "_macos_security_available", lambda: False)

    assert credentials.store_session_token("127.0.0.1", 8765, "admin", "secret-token") is False
    assert credentials.load_session_token("127.0.0.1", 8765, "admin") is None
