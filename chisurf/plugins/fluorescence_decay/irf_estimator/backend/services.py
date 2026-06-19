from __future__ import annotations

from typing import Any

import numpy as np

from ..api.contract import (
    METHOD_ESTIMATE_IRF,
    METHOD_LOAD_DECAY,
    METHOD_LOAD_DATASET,
    METHOD_SAVE_IRF,
    METHOD_TRANSFER_IRF,
    METHOD_DESCRIBE_CONTRACT,
    contract_descriptor,
    service_success,
)
from ..api.models import IRFEstimationSettings
from ..api.serialization import settings_from_dict
from ..core.estimation import estimate_irf as _estimate_irf


def register_services(dispatcher: Any) -> None:
    """Register IRF Estimator RPC handlers with a ServiceDispatcher."""
    dispatcher.register(
        METHOD_ESTIMATE_IRF,
        lambda params: estimate_handler(**params),
    )
    dispatcher.register(
        METHOD_LOAD_DECAY,
        lambda params: load_decay_handler(**params),
    )
    dispatcher.register(
        METHOD_LOAD_DATASET,
        lambda params: load_dataset_handler(**params),
    )
    dispatcher.register(
        METHOD_SAVE_IRF,
        lambda params: save_irf_handler(**params),
    )
    dispatcher.register(
        METHOD_TRANSFER_IRF,
        lambda params: transfer_irf_handler(**params),
    )
    dispatcher.register(
        METHOD_DESCRIBE_CONTRACT,
        lambda params: contract_handler(**(params or {})),
    )


def estimate_handler(
    intensity: list[float],
    dt: float = 1.0,
    settings: dict[str, Any] | None = None,
    time_axis: list[float] | None = None,
) -> dict[str, Any]:
    """Run IRF estimation."""
    try:
        est_settings = settings_from_dict(settings)
        channel_axis = np.array(time_axis) if time_axis is not None else None
        result = _estimate_irf(
            intensity=np.array(intensity, dtype=np.float32),
            dt=dt,
            settings=est_settings,
            channel_axis=channel_axis,
        )
        return service_success({
            "irf": result.irf,
            "params": result.params,
            "time_axis": result.time_axis,
            "dt": result.dt,
            "lifetime_ns": result.lifetime_ns,
            "decay_rate_ns": result.decay_rate_ns,
            "amplitude": result.amplitude,
            "offset": result.offset,
        })
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def load_decay_handler(path: str) -> dict[str, Any]:
    """Load a Jordi format decay file and return its data."""
    try:
        from chisurf.core.fio import read_jordi
        data, metadata = read_jordi(path, return_metadata=True)
        data = np.asarray(data, dtype=np.float32)
        # First half is VV, second half is VH in legacy format
        if len(data) % 2 == 0:
            vv = data[:len(data) // 2]
        else:
            vv = data
        dt = float(metadata.get("dt", 1.0))
        return service_success({
            "intensity": vv.tolist(),
            "dt": dt,
            "filename": path,
        })
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def load_dataset_handler() -> dict[str, Any]:
    """Load decay data from a ChiSurf dataset (stub for RPC)."""
    try:
        import chisurf as cs
        from chisurf.core.data import get_data
        all_curves = get_data(
            data_set=getattr(cs, "imported_datasets", []),
            curve_type="experiment",
        )
        datasets = []
        for ds in all_curves:
            if hasattr(ds, "y") and hasattr(ds, "x"):
                x = np.asarray(ds.x, dtype=np.float32)
                y = np.asarray(ds.y, dtype=np.float32)
                dt = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
                datasets.append({
                    "name": getattr(ds, "name", "Unnamed"),
                    "experiment": getattr(getattr(ds, "experiment", None), "name", "Uncategorized"),
                    "intensity": y.tolist(),
                    "time_axis": x.tolist(),
                    "dt": dt,
                })
        return service_success({"datasets": datasets})
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def save_irf_handler(path: str, irf_data: list[float], dt: float = 1.0) -> dict[str, Any]:
    """Save IRF data to a Jordi file."""
    try:
        from chisurf.core.fio import write_jordi
        import os
        if not path.lower().endswith(".dat"):
            path += ".dat"
        irf_array = np.asarray(irf_data, dtype=float).flatten()
        write_jordi(
            path,
            data=np.column_stack((np.arange(len(irf_array)) * dt, irf_array)),
            metadata={"dt": dt},
        )
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError("Failed to save IRF file")
        return service_success({"path": path})
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def transfer_irf_handler() -> dict[str, Any]:
    """Transfer estimated IRF to ChiSurf (stub — actual transfer is GUI-driven)."""
    return service_success({"transferred": False, "message": "Use GUI for transfer"})


def contract_handler() -> dict[str, Any]:
    """Return the IRF Estimator workflow contract descriptor."""
    return service_success(contract_descriptor())
