"""Backend RPC services for the FCS filter calculator.

Wraps the Qt-free :mod:`..api` (which itself builds on
``chisurf.core.fluorescence.fcs.filtered``) as transport-agnostic RPC handlers
under the ``fcs_filter.*`` namespace.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .. import api


def _ok(result: Any) -> Dict[str, Any]:
    return {"ok": True, "result": result}


def _err(exc: Exception) -> Dict[str, Any]:
    return {"ok": False, "error": str(exc)}


def compute_handler(total_decay, species_decays, metadata=None) -> Dict[str, Any]:
    try:
        total = np.asarray(total_decay, dtype=float)
        species = [np.asarray(s, dtype=float) for s in species_decays]
        res = api.compute_filters(total, species, metadata=metadata)
        return _ok(res.to_dict())
    except Exception as exc:
        return _err(exc)


def compute_from_files_handler(total_path: str, species_paths: List[str]) -> Dict[str, Any]:
    try:
        res = api.compute_filters_from_files(total_path, species_paths)
        return _ok(res.to_dict())
    except Exception as exc:
        return _err(exc)


def compute_mfd_handler(
    total_decay_par, total_decay_perp, species_decays_par, species_decays_perp,
    metadata=None,
) -> Dict[str, Any]:
    try:
        res = api.compute_filters_mfd(
            np.asarray(total_decay_par, dtype=float),
            np.asarray(total_decay_perp, dtype=float),
            [np.asarray(s, dtype=float) for s in species_decays_par],
            [np.asarray(s, dtype=float) for s in species_decays_perp],
            metadata=metadata,
        )
        return _ok(res.to_dict())
    except Exception as exc:
        return _err(exc)


def compute_mfd_from_files_handler(
    total_par_path: str, total_perp_path: str,
    species_par_paths: List[str], species_perp_paths: List[str],
) -> Dict[str, Any]:
    try:
        res = api.compute_filters_mfd_from_files(
            total_par_path, total_perp_path, species_par_paths, species_perp_paths,
        )
        return _ok(res.to_dict())
    except Exception as exc:
        return _err(exc)


def register_services(dispatcher: Any) -> None:
    """Register the fcs_filter RPC handlers with a ServiceDispatcher."""
    dispatcher.register("fcs_filter.compute", lambda p: compute_handler(**p))
    dispatcher.register("fcs_filter.compute_from_files", lambda p: compute_from_files_handler(**p))
    dispatcher.register("fcs_filter.compute_mfd", lambda p: compute_mfd_handler(**p))
    dispatcher.register(
        "fcs_filter.compute_mfd_from_files", lambda p: compute_mfd_from_files_handler(**p)
    )
