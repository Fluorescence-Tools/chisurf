"""Metadata helpers for Trace Browser."""

from __future__ import annotations

import json
import pathlib

META_FILENAME = ".trace_browser_meta.json"


def meta_path(folder: pathlib.Path) -> pathlib.Path:
    """Return the metadata file path for *folder*."""
    return folder / META_FILENAME


def load_meta(folder: pathlib.Path) -> dict[str, dict]:
    """Load trace-browser metadata from *folder*."""
    path = meta_path(folder)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_meta(folder: pathlib.Path, data: dict[str, dict]) -> None:
    """Atomically save trace-browser metadata."""
    path = meta_path(folder)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if path.exists():
        path.unlink()
    tmp.replace(path)
