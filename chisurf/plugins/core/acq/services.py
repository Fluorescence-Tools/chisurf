"""RPC service registration for the Acquisition plugin.

Exposes the tttrlib ``Sim*`` photon simulator over the ChiSurf JSON-RPC
``ServiceDispatcher`` (ZMQ). The method contract + per-parameter help live in
``manifest.json`` under ``rpc_methods`` (``acq.simulation.run``) — the single
source of truth shared by the GUI form (tooltips), CLI, API and this handler.
The plugin registry calls :func:`register` via the manifest ``services``
entrypoint at server start.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict

#: RPC method name; mirrors ``rpc_methods[0].name`` in ``manifest.json``.
METHOD_SIMULATION_RUN = "acq.simulation.run"


def simulation_run(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run one photon-stream simulation and (optionally) write SPC output.

    Parameters
    ----------
    params : dict
        The ``acq.simulation.run`` parameters — see the ``params_schema`` in
        ``manifest.json`` for every knob and its documentation.

    Returns
    -------
    dict
        ``{"n_photons": int, "output_path": str | None}``.
    """
    from .tcspc_devices.simulation.core.algorithms import (
        build_engine,
        encode_records,
        tttrlib_available,
    )

    if not tttrlib_available():
        raise RuntimeError(
            "tttrlib with the Sim* photon simulator is required "
            "(install/upgrade tttrlib)."
        )

    params = dict(params or {})
    engine = build_engine(params)
    engine.run()
    result: Dict[str, Any] = {"n_photons": int(engine.n_photons()), "output_path": None}

    out_path = params.get("spc_output_path") or params.get("output_path")
    if out_path:
        words = encode_records(engine, params)
        p = pathlib.Path(out_path)
        if p.is_dir() or out_path.endswith(("/", "\\")):
            p = p / "simulation.spc"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(words.tobytes())
        result["output_path"] = str(p)

    return result


def register(dispatcher: Any) -> None:
    """Register acquisition RPC methods on the server's ``ServiceDispatcher``."""
    dispatcher.register(METHOD_SIMULATION_RUN, simulation_run)
