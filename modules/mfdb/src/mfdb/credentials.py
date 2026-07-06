"""Secure client-side storage for MFDB session tokens."""

from __future__ import annotations

import shutil
import subprocess
import sys

SERVICE_NAME = "ChiSurf MFDB"
_RUNTIME_SESSION_TOKENS: dict[str, str] = {}


def session_token_registry() -> dict[str, str]:
    """Return the live in-process session-token registry.

    Public accessor for the runtime session-token map so callers do not depend on
    the private ``_RUNTIME_SESSION_TOKENS`` name. The returned dict is the live
    object, not a copy — mutations are visible to the store.

    Returns
    -------
    dict of str to str
        Mapping of session account key to session token.
    """
    return _RUNTIME_SESSION_TOKENS


def credential_account(server_host: str, server_port: int, user_id: str) -> str:
    """Return the credential-store account key for an MFDB session.

    Parameters
    ----------
    server_host : str
        MFDB server host.
    server_port : int
        MFDB command port.
    user_id : str
        MFDB user identifier.

    Returns
    -------
    str
        Stable account key for the OS credential store.
    """
    return f"{server_host}:{int(server_port)}:{user_id}"


def store_runtime_session_token(server_host: str, server_port: int, user_id: str, token: str) -> bool:
    """Store an MFDB session token for the current process only.

    Parameters
    ----------
    server_host : str
        MFDB server host.
    server_port : int
        MFDB command port.
    user_id : str
        MFDB user identifier.
    token : str
        Session token returned by the MFDB backend.

    Returns
    -------
    bool
        ``True`` when a token was stored in memory.
    """
    if not token:
        return False
    _RUNTIME_SESSION_TOKENS[credential_account(server_host, server_port, user_id)] = token
    return True


def load_runtime_session_token(server_host: str, server_port: int, user_id: str) -> str | None:
    """Load an MFDB session token kept for the current process.

    Parameters
    ----------
    server_host : str
        MFDB server host.
    server_port : int
        MFDB command port.
    user_id : str
        MFDB user identifier.

    Returns
    -------
    str or None
        Runtime session token, or ``None`` when no token is available.
    """
    return _RUNTIME_SESSION_TOKENS.get(credential_account(server_host, server_port, user_id))


def delete_runtime_session_token(server_host: str, server_port: int, user_id: str) -> bool:
    """Delete an MFDB session token kept for the current process.

    Parameters
    ----------
    server_host : str
        MFDB server host.
    server_port : int
        MFDB command port.
    user_id : str
        MFDB user identifier.

    Returns
    -------
    bool
        ``True`` after the runtime token entry is absent.
    """
    _RUNTIME_SESSION_TOKENS.pop(credential_account(server_host, server_port, user_id), None)
    return True


def rename_runtime_session_token(
    server_host: str,
    server_port: int,
    old_user_id: str,
    new_user_id: str,
) -> bool:
    """Move a runtime session token to a renamed user account.

    Parameters
    ----------
    server_host : str
        MFDB server host.
    server_port : int
        MFDB command port.
    old_user_id : str
        Previous MFDB user identifier.
    new_user_id : str
        New MFDB user identifier.

    Returns
    -------
    bool
        ``True`` when the runtime token was moved or no old token existed.
    """
    old_account = credential_account(server_host, server_port, old_user_id)
    token = _RUNTIME_SESSION_TOKENS.pop(old_account, None)
    if token:
        _RUNTIME_SESSION_TOKENS[credential_account(server_host, server_port, new_user_id)] = token
    return True


def rename_session_token(server_host: str, server_port: int, old_user_id: str, new_user_id: str) -> bool:
    """Move a persisted session token to a renamed user account.

    Parameters
    ----------
    server_host : str
        MFDB server host.
    server_port : int
        MFDB command port.
    old_user_id : str
        Previous MFDB user identifier.
    new_user_id : str
        New MFDB user identifier.

    Returns
    -------
    bool
        ``True`` when no token existed or the token was moved.
    """
    token = load_session_token(server_host, server_port, old_user_id)
    if not token:
        return True
    if not store_session_token(server_host, server_port, new_user_id, token):
        return False
    delete_session_token(server_host, server_port, old_user_id)
    return True


def store_session_token(server_host: str, server_port: int, user_id: str, token: str) -> bool:
    """Store an MFDB session token in the OS credential store.

    Parameters
    ----------
    server_host : str
        MFDB server host.
    server_port : int
        MFDB command port.
    user_id : str
        MFDB user identifier.
    token : str
        Session token returned by the MFDB backend.

    Returns
    -------
    bool
        ``True`` when the token was stored.
    """
    if not token:
        return False
    account = credential_account(server_host, server_port, user_id)
    keyring = _get_keyring()
    if keyring is not None:
        try:
            keyring.set_password(SERVICE_NAME, account, token)
            return True
        except Exception:
            return False
    return _store_macos_keychain(account, token)


def load_session_token(server_host: str, server_port: int, user_id: str) -> str | None:
    """Load an MFDB session token from the OS credential store.

    Parameters
    ----------
    server_host : str
        MFDB server host.
    server_port : int
        MFDB command port.
    user_id : str
        MFDB user identifier.

    Returns
    -------
    str or None
        Stored session token, or ``None`` when unavailable.
    """
    account = credential_account(server_host, server_port, user_id)
    keyring = _get_keyring()
    if keyring is not None:
        try:
            return keyring.get_password(SERVICE_NAME, account)
        except Exception:
            return None
    return _load_macos_keychain(account)


def delete_session_token(server_host: str, server_port: int, user_id: str) -> bool:
    """Delete an MFDB session token from the OS credential store.

    Parameters
    ----------
    server_host : str
        MFDB server host.
    server_port : int
        MFDB command port.
    user_id : str
        MFDB user identifier.

    Returns
    -------
    bool
        ``True`` when the token was deleted or no token existed.
    """
    account = credential_account(server_host, server_port, user_id)
    keyring = _get_keyring()
    if keyring is not None:
        try:
            keyring.delete_password(SERVICE_NAME, account)
            return True
        except Exception:
            return False
    return _delete_macos_keychain(account)


def credential_store_available() -> bool:
    """Return whether a supported credential store is available.

    Returns
    -------
    bool
        ``True`` if keyring or the macOS Keychain command is available.
    """
    return _get_keyring() is not None or _macos_security_available()


def _get_keyring():
    """Return the optional keyring module when installed.

    Returns
    -------
    module or None
        Imported keyring module, or ``None`` when unavailable.
    """
    try:
        import keyring
    except Exception:
        return None
    return keyring


def _macos_security_available() -> bool:
    """Return whether the macOS ``security`` command can be used.

    Returns
    -------
    bool
        ``True`` on macOS when ``security`` is on ``PATH``.
    """
    return sys.platform == "darwin" and shutil.which("security") is not None


def _store_macos_keychain(account: str, token: str) -> bool:
    """Store a token using macOS Keychain.

    Parameters
    ----------
    account : str
        Credential account key.
    token : str
        Session token to store.

    Returns
    -------
    bool
        ``True`` when the token was stored.
    """
    if not _macos_security_available():
        return False
    result = subprocess.run(
        [
            "security",
            "add-generic-password",
            "-a",
            account,
            "-s",
            SERVICE_NAME,
            "-w",
            token,
            "-U",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _load_macos_keychain(account: str) -> str | None:
    """Load a token using macOS Keychain.

    Parameters
    ----------
    account : str
        Credential account key.

    Returns
    -------
    str or None
        Stored token, or ``None`` when unavailable.
    """
    if not _macos_security_available():
        return None
    result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-a",
            account,
            "-s",
            SERVICE_NAME,
            "-w",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _delete_macos_keychain(account: str) -> bool:
    """Delete a token using macOS Keychain.

    Parameters
    ----------
    account : str
        Credential account key.

    Returns
    -------
    bool
        ``True`` when the item was deleted or did not exist.
    """
    if not _macos_security_available():
        return False
    result = subprocess.run(
        [
            "security",
            "delete-generic-password",
            "-a",
            account,
            "-s",
            SERVICE_NAME,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode in (0, 44)
