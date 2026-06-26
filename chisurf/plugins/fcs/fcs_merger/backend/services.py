"""Backend RPC services for the FCS-Merger plugin (``fcs_merger.*``)."""

from __future__ import annotations

from typing import Any, Dict, List

from ..core import compute_average_correlations, merge_folder, parse_correlation_folder


def merge_folder_handler(folder: str, output: str = None) -> Dict[str, Any]:
    try:
        return {"ok": True, "result": merge_folder(folder, output)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def average_handler(correlations: List[dict]) -> Dict[str, Any]:
    try:
        import numpy as np
        m = compute_average_correlations(correlations)
        return {"ok": True, "result": {
            "x": np.asarray(m["x"], dtype=float).tolist(),
            "y": np.asarray(m["y"], dtype=float).tolist(),
            "ey": np.asarray(m["ey"], dtype=float).tolist(),
            "duration": float(m["duration"]),
            "count_rate": float(m["count_rate"]),
        }}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def parse_folder_handler(folder: str) -> Dict[str, Any]:
    try:
        return {"ok": True, "result": {"correlations": parse_correlation_folder(folder)}}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def register_services(dispatcher: Any) -> None:
    """Register the fcs_merger RPC handlers with a ServiceDispatcher."""
    dispatcher.register("fcs_merger.merge_folder", lambda p: merge_folder_handler(**p))
    dispatcher.register("fcs_merger.average", lambda p: average_handler(**p))
    dispatcher.register("fcs_merger.parse_folder", lambda p: parse_folder_handler(**p))
