"""Serialization helpers for MaxEnt MEM API payloads."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from .models import MEMRequest, MEMResult, MEMSettings


def to_jsonable(value: Any) -> Any:
    """Return a JSON-compatible representation of ``value``."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def settings_from_dict(value: dict[str, Any] | MEMSettings | None) -> MEMSettings:
    """Build :class:`MEMSettings` from a JSON-compatible dictionary."""
    if value is None:
        return MEMSettings()
    if isinstance(value, MEMSettings):
        return value
    data = dict(value or {})
    extra = data.pop("extra", None)
    settings = MEMSettings(**data)
    if extra is not None:
        settings.extra = dict(extra)
    return settings


def request_from_dict(value: dict[str, Any] | MEMRequest) -> MEMRequest:
    """Build :class:`MEMRequest` from a JSON-compatible dictionary."""
    if isinstance(value, MEMRequest):
        return value
    data = dict(value or {})
    settings = settings_from_dict(data.get("settings"))
    fitrange = data.get("fitrange")
    if isinstance(fitrange, (list, tuple)) and len(fitrange) == 2:
        fitrange = (int(fitrange[0]), int(fitrange[1]))
    elif fitrange is not None:
        fitrange = None
    return MEMRequest(
        decay=[float(x) for x in data.get("decay", [])],
        irf=[float(x) for x in data.get("irf", [])],
        dt=float(data.get("dt", 1.0)),
        fitrange=fitrange,
        settings=settings,
        prior=data.get("prior"),
        donly=data.get("donly"),
    )


def request_to_payload(request: MEMRequest) -> dict[str, Any]:
    """Return a JSON-compatible request payload."""
    return to_jsonable(request)


def result_to_payload(result: MEMResult) -> dict[str, Any]:
    """Return a JSON-compatible result payload."""
    return to_jsonable(result)
