"""IO helpers for Trace Browser."""

from __future__ import annotations

import csv
import pathlib
from collections.abc import Iterable
from typing import Any

import numpy as np

from chisurf.plugins.tttr.trace_browser.core.metadata import load_meta
from chisurf.plugins.tttr.trace_browser.core.trace import human_size, load_trace

DEFAULT_EXTENSIONS = {".ptu", ".phu", ".ht2", ".ht3", ".pt3", ".t3r"}


def iter_trace_files(folder: pathlib.Path, recursive: bool = False, extensions: Iterable[str] | None = None) -> list[pathlib.Path]:
    """Return trace files under *folder*."""
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in (extensions or DEFAULT_EXTENSIONS)}
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    files: list[pathlib.Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() in exts or path.name.lower().endswith(tuple(ext.lower() + ".gz" for ext in exts)):
            files.append(path)
    return sorted(files, key=lambda item: item.name.lower())


def list_files(folder: str, recursive: bool = False, setup_settings: dict[str, Any] | None = None, selected_channels: list[int] | None = None) -> list[dict[str, Any]]:
    """List trace files with metadata summaries."""
    root = pathlib.Path(folder)
    meta = load_meta(root)
    rows: list[dict[str, Any]] = []
    for path in iter_trace_files(root, recursive=recursive):
        file_meta = meta.get(str(path)) or meta.get(path.name) or {}
        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "size": path.stat().st_size,
                "size_text": human_size(path.stat().st_size),
                "rating": int(file_meta.get("rating", 0)),
                "annotation": str(file_meta.get("annotation", "")),
                "channels": list(selected_channels or []),
            }
        )
    return rows


def export_csv(paths: list[str], output_dir: str, time_window_ms: float = 10.0, setup_settings: dict[str, Any] | None = None, selected_channels: list[int] | None = None) -> dict[str, Any]:
    """Export loaded traces to CSV files."""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for path in paths:
        data = load_trace(path, time_window_ms=time_window_ms, setup_settings=setup_settings, selected_channels=selected_channels, cache_folder=str(out))
        time_axis = np.asarray(data["time_axis"], dtype=float)
        counts = np.asarray(data["counts"], dtype=float)
        labels = data.get("labels") or [str(idx) for idx in range(counts.shape[1])]
        csv_path = out / f"{pathlib.Path(path).stem}_trace.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_ms", *labels])
            for idx, time_value in enumerate(time_axis):
                writer.writerow([time_value, *counts[idx].tolist()])
        written.append(str(csv_path))
    return {"paths": written}
