"""RPC method constants and service envelopes for TTTR Image Browser."""

from __future__ import annotations

from typing import Any

METHOD_LIST_FILES = "tttr_image_browser.files.list"
METHOD_GET_METADATA = "tttr_image_browser.metadata.get"
METHOD_SET_METADATA = "tttr_image_browser.metadata.set"
METHOD_LOAD_IMAGE = "tttr_image_browser.images.load"
METHOD_EXPORT_TIFF = "tttr_image_browser.export.tiff"
METHOD_CONTRACT = "tttr_image_browser.contract.describe"

LEGACY_METHODS: dict[str, str] = {}


def service_success(result: Any) -> dict[str, Any]:
    """Return a successful RPC envelope.

    Parameters
    ----------
    result : Any
        The operation result to envelope.

    Returns
    -------
    dict[str, Any]
        The enveloped success response.
    """
    return {"ok": True, "result": result, "error": None}


def service_error(message: str, code: str = "ERROR") -> dict[str, Any]:
    """Return a failed RPC envelope.

    Parameters
    ----------
    message : str
        The error message.
    code : str, optional
        The error code, by default "ERROR".

    Returns
    -------
    dict[str, Any]
        The enveloped error response.
    """
    return {"ok": False, "result": None, "error": {"message": message, "code": code}}


def unwrap(result: dict[str, Any] | None) -> Any:
    """Unwrap a successful RPC envelope.

    Parameters
    ----------
    result : dict[str, Any] or None
        The RPC response dict.

    Returns
    -------
    Any
        The unwrapped result, or None.
    """
    if result and result.get("ok"):
        return result.get("result")
    return None


def contract_descriptor() -> dict[str, Any]:
    """Return the TTTR Image Browser RPC contract.

    Returns
    -------
    dict[str, Any]
        A descriptor of the supported RPC methods.
    """
    return {
        "methods": [
            METHOD_LIST_FILES,
            METHOD_GET_METADATA,
            METHOD_SET_METADATA,
            METHOD_LOAD_IMAGE,
            METHOD_EXPORT_TIFF,
            METHOD_CONTRACT,
        ],
        "legacy_aliases": LEGACY_METHODS,
    }
