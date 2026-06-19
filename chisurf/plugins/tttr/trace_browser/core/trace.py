"""Trace loading helpers for Trace Browser."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import numpy as np

from chisurf.plugins.tttr.intensity_trace.__init__ import IntensityTrace
from chisurf.plugins.tttr.trace_browser.api.models import TraceLoadResult


def human_size(size_bytes: int) -> str:
    """Return a human-readable file size."""
    try:
        return f"{float(size_bytes) / (1024.0 * 1024.0):.1f} MB"
    except Exception:
        return "? MB"


def cache_dir_for(folder: pathlib.Path, file_path: pathlib.Path) -> pathlib.Path:
    """Return the trace-cache directory for *file_path*."""
    base = folder or file_path.parent
    path = base / ".tttr_trace_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def trace_signature(file_path: pathlib.Path, window_ms: float, setup_settings: dict[str, Any] | None, selected_channels: list[int] | None) -> str:
    """Return a stable cache signature for a trace load."""
    try:
        stat = file_path.stat()
        size = int(getattr(stat, "st_size", 0))
        mtime = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
    except Exception:
        size = 0
        mtime = 0
    mode: dict[str, Any]
    if isinstance(setup_settings, dict) and "detectors" in setup_settings:
        detectors = setup_settings.get("detectors") or {}
        mode = {
            key: {
                "chs": list(value.get("chs", [])),
                "micro_time_ranges": list(value.get("micro_time_ranges", [])),
            }
            for key, value in detectors.items()
        }
    else:
        mode = {"chs": list(selected_channels) if selected_channels else None}
    payload = {
        "v": 1,
        "path": str(file_path.resolve()),
        "size": size,
        "mtime": mtime,
        "win_ms": float(window_ms),
        "mode": mode,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def cache_file(file_path: pathlib.Path, folder: pathlib.Path, window_ms: float, setup_settings: dict[str, Any] | None, selected_channels: list[int] | None) -> pathlib.Path:
    """Return the on-disk cache file for a trace."""
    sig = trace_signature(file_path, window_ms, setup_settings, selected_channels)
    return cache_dir_for(folder, file_path) / f"{file_path.stem}_{sig}.npz"


def load_cached(file_path: pathlib.Path, folder: pathlib.Path, window_ms: float, setup_settings: dict[str, Any] | None, selected_channels: list[int] | None) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Load a cached trace if present."""
    try:
        path = cache_file(file_path, folder, window_ms, setup_settings, selected_channels)
        if not path.exists():
            return None
        with np.load(str(path), allow_pickle=True) as data:
            time_axis = data["time_axis"]
            counts = data["padded"]
            labels = list(data["labels"]) if "labels" in data else []
            return time_axis, counts, labels
    except Exception:
        return None


def save_cached(file_path: pathlib.Path, folder: pathlib.Path, window_ms: float, setup_settings: dict[str, Any] | None, selected_channels: list[int] | None, time_axis: np.ndarray, counts: np.ndarray, labels: list[str]) -> None:
    """Save a computed trace to disk cache."""
    try:
        path = cache_file(file_path, folder, window_ms, setup_settings, selected_channels)
        np.savez_compressed(str(path), time_axis=time_axis, padded=counts, labels=np.array(labels, dtype=object))
    except Exception:
        pass


def load_trace(
    path: str,
    time_window_ms: float = 10.0,
    setup_settings: dict[str, Any] | None = None,
    selected_channels: list[int] | None = None,
    cache_folder: str | None = None,
) -> dict[str, Any]:
    """Load and bin a TTTR trace into JSON-safe arrays."""
    file_path = pathlib.Path(path)
    folder = pathlib.Path(cache_folder) if cache_folder else file_path.parent
    window_ms = float(time_window_ms)
    cached = load_cached(file_path, folder, window_ms, setup_settings, selected_channels)
    if cached is not None:
        time_axis, counts, labels = cached
    else:
        time_window_s = window_ms / 1000.0
        if isinstance(setup_settings, dict) and "detectors" in setup_settings:
            time_axis, counts, labels = IntensityTrace().process_ptu(
                file_path,
                time_window_s,
                selected_detectors=setup_settings.get("detectors") or {},
            )
        else:
            channels = selected_channels
            if channels is None:
                try:
                    import tttrlib

                    tttr_obj = tttrlib.TTTR(str(file_path))
                    channels = sorted(tttr_obj.get_used_routing_channels())
                except Exception:
                    channels = []
            time_axis, counts, channels = IntensityTrace().process_ptu(file_path, time_window_s, channels)
            labels = [str(channel) for channel in channels]
        save_cached(file_path, folder, window_ms, setup_settings, selected_channels, time_axis, counts, labels)
    result = TraceLoadResult(
        path=str(file_path),
        time_axis=np.asarray(time_axis).tolist(),
        counts=np.asarray(counts).tolist(),
        labels=[str(label) for label in labels],
        time_window_ms=window_ms,
    )
    return {
        "time_axis": result.time_axis,
        "counts": result.counts,
        "labels": result.labels,
        "time_window_ms": result.time_window_ms,
    }
