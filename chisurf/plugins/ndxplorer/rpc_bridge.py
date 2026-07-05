"""Build an in-process ChiSurf RPC client for ndXplorer (PRD-56).

In the GUI, ndXplorer runs in-process with ChiSurf, so it does not need a socket: it can
be handed an :class:`~chisurf.core.plugin.client.InProcessClient` wired to a
``ServiceDispatcher`` that has the phasor and FRET-line services registered. That client
satisfies ndXplorer's chisurf-free ``RpcClient`` contract (``call(method, params)``), so
``NDXplorer(chisurf_rpc=...)`` gains the ``phasor.*`` / ``fret_line.*`` methods with no
server process and no configuration.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Plugin service entrypoints to register on the in-process dispatcher.
_SERVICE_REGISTRARS = (
    "chisurf.plugins.microscopy.img_pixel_phasor.backend.services:register_services",
    "chisurf.plugins.fret_line.backend.services:register_services",
)


def _load(path: str):
    module_name, attr = path.split(":", 1)
    import importlib

    return getattr(importlib.import_module(module_name), attr)


def make_inprocess_chisurf_client() -> Any | None:
    """Return an ``InProcessClient`` exposing the phasor + FRET-line RPC methods.

    Best-effort: returns ``None`` if the RPC/plugin machinery is unavailable, so callers
    can degrade gracefully (ndXplorer then runs without ChiSurf features).
    """
    try:
        from chisurf.core.plugin.client import InProcessClient
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState
    except Exception as exc:  # pragma: no cover - optional server stack
        logger.warning("ChiSurf RPC stack unavailable: %s", exc)
        return None

    dispatcher = ServiceDispatcher(SessionState())
    for path in _SERVICE_REGISTRARS:
        try:
            _load(path)(dispatcher)
        except Exception:
            logger.warning("Could not register services %s", path, exc_info=True)
    return InProcessClient(dispatcher)


def make_ndxplorer(**kwargs: Any):
    """Construct an ``NDXplorer`` with the in-process ChiSurf client injected.

    The single entry point every in-GUI launcher should use so the window always
    gets the phasor / FRET-line features (the "ChiSurf Phasor" toolbar). Any
    caller-supplied ``chisurf_rpc`` is respected; otherwise an in-process client is
    built and injected. Falls back to a plain ``NDXplorer`` if the client cannot be
    built, so ndXplorer still opens when the RPC stack is unavailable.
    """
    import ndxplorer

    if kwargs.get("chisurf_rpc") is None:
        client = make_inprocess_chisurf_client()
        if client is not None:
            kwargs["chisurf_rpc"] = client
    return ndxplorer.NDXplorer(**kwargs)
