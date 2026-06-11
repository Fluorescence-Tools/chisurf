from __future__ import annotations

import logging
from typing import Any

from chisurf.server.services import INVALID_INPUT, ServiceResult, service_error
from chisurf.server.session import SessionState


_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def write_log(
    state: SessionState,
    level: str = "info",
    message: str = "",
    logger_name: str = "chisurf.rpc",
    extra: dict[str, Any] | None = None,
) -> ServiceResult:
    """Write one log record through the server logging system.

    Parameters
    ----------
    state : SessionState
        Server-side session state. Included for service signature consistency.
    level : str
        Logging level name.
    message : str
        Log message.
    logger_name : str
        Name of the Python logger to write to.
    extra : dict, optional
        Optional structured fields appended to the message.

    Returns
    -------
    ServiceResult
        ``{"ok": True}`` on success.

    """
    del state
    normalized_level = str(level or "info").lower()
    if normalized_level not in _LEVELS:
        return service_error(
            f"invalid log level: {level}",
            error_code=INVALID_INPUT,
        )

    text = str(message)
    if extra:
        text = f"{text} | {extra}"
    logging.getLogger(str(logger_name or "chisurf.rpc")).log(_LEVELS[normalized_level], text)
    return {"ok": True}
