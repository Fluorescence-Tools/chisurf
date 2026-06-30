"""ServiceDispatcher-compatible RPC handlers for FRET docking.

Thin adapters: accept JSON-compatible params, delegate to
:mod:`...api.operations`, return JSON-safe results. Registered with the
server/in-process ``ServiceDispatcher`` via the plugin manifest's
``entrypoints.services``.
"""

from __future__ import annotations

from typing import Any, Dict

from ..api import operations as ops


def register_services(dispatcher: Any) -> None:
    """Register the ``fret.*`` RPC methods with a ServiceDispatcher.

    Parameters
    ----------
    dispatcher : ServiceDispatcher
        The server's (or in-process) service dispatcher.
    """
    dispatcher.register("fret.info_backends", lambda params: ops.backend_info())
    dispatcher.register("fret.dock", lambda params: ops.dock(params or {}))
    dispatcher.register("fret.refine", lambda params: ops.refine(params or {}))
    dispatcher.register("fret.score", lambda params: ops.score(params or {}))
    dispatcher.register("fret.screen", lambda params: ops.screen(params or {}))
    dispatcher.register("fret.estimate_errors", lambda params: ops.estimate_errors(params or {}))


def list_methods() -> Dict[str, str]:
    """Return the RPC method catalogue."""
    return {
        "fret.info_backends": "Report IMP/IMP.bff backend availability.",
        "fret.dock": "FRET-restrained Monte-Carlo rigid-body docking.",
        "fret.refine": "Conjugate-gradient local refinement of a pose.",
        "fret.score": "Score a single structure against FRET restraints.",
        "fret.screen": "Score and rank a structure library.",
        "fret.estimate_errors": "Repeated-trial docking error estimation.",
    }
