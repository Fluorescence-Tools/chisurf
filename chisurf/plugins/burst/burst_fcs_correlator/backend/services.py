"""Backend RPC services for the burst-wise FCS correlator.

Exposes the Qt-free core (:mod:`..core.algorithms`) as transport-agnostic RPC
handlers under the ``burst_fcs.*`` namespace.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List

import numpy as np

from ..core.algorithms import (
    BurstFcsSettings,
    PairConfig,
    correlate_burst_file,
    fit_curve,
    fit_diffusion_time,
    fit_simple_diffusion,
    parse_bst_file,
    parse_bur_file,
)


def _ok(result: Any) -> Dict[str, Any]:
    return {"ok": True, "result": result}


def parse_bst_handler(path: str) -> Dict[str, Any]:
    tttr_path, ranges = parse_bst_file(pathlib.Path(path))
    return _ok({
        "tttr_path": str(tttr_path) if tttr_path is not None else None,
        "ranges": [[int(a), int(b)] for a, b in ranges],
    })


def parse_bur_handler(path: str, analysis_root: str = None) -> Dict[str, Any]:
    root = pathlib.Path(analysis_root) if analysis_root else pathlib.Path(path).parent
    tttr_path, ranges = parse_bur_file(pathlib.Path(path), root)
    return _ok({
        "tttr_path": str(tttr_path) if tttr_path is not None else None,
        "ranges": [[int(a), int(b)] for a, b in ranges],
    })


def fit_curve_handler(tau, g, settings: Dict[str, Any] = None) -> Dict[str, Any]:
    s = BurstFcsSettings.from_dict(settings or {})
    return _ok(fit_curve(np.asarray(tau, dtype=float), np.asarray(g, dtype=float), s))


def fit_diffusion_handler(tau, g) -> Dict[str, Any]:
    td_mean, td_peak = fit_diffusion_time(np.asarray(tau, dtype=float), np.asarray(g, dtype=float))
    return _ok({"td_mean": td_mean, "td_peak": td_peak})


def fit_simple_handler(tau, g) -> Dict[str, Any]:
    td, tau_used, g_used, g_fit = fit_simple_diffusion(
        np.asarray(tau, dtype=float), np.asarray(g, dtype=float)
    )
    return _ok({
        "td": float(td),
        "tau": np.asarray(tau_used, dtype=float).tolist(),
        "g": np.asarray(g_used, dtype=float).tolist(),
        "g_fit": np.asarray(g_fit, dtype=float).tolist(),
    })


def correlate_file_handler(
    tttr_path: str,
    ranges: List,
    pairs: List[Dict[str, Any]],
    settings: Dict[str, Any] = None,
    filetype=None,
) -> Dict[str, Any]:
    s = BurstFcsSettings.from_dict(settings or {})
    pair_cfgs = [PairConfig.from_dict(p) for p in (pairs or [])]
    rng = [(int(a), int(b)) for a, b in ranges]
    curves = correlate_burst_file(pathlib.Path(tttr_path), rng, pair_cfgs, s, filetype=filetype)
    return _ok({"curves": curves})


def register_services(dispatcher: Any) -> None:
    """Register the burst-FCS RPC handlers with a ServiceDispatcher."""
    dispatcher.register("burst_fcs.parse_bst", lambda p: parse_bst_handler(**p))
    dispatcher.register("burst_fcs.parse_bur", lambda p: parse_bur_handler(**p))
    dispatcher.register("burst_fcs.fit_curve", lambda p: fit_curve_handler(**p))
    dispatcher.register("burst_fcs.fit_diffusion", lambda p: fit_diffusion_handler(**p))
    dispatcher.register("burst_fcs.fit_simple", lambda p: fit_simple_handler(**p))
    dispatcher.register("burst_fcs.correlate_file", lambda p: correlate_file_handler(**p))
