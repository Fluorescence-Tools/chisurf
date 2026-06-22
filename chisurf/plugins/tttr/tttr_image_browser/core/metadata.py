"""Metadata helpers for TTTR Image Browser."""

from __future__ import annotations

import json
import pathlib

META_FILENAME = ".image_browser_meta.json"


def meta_path(folder: pathlib.Path) -> pathlib.Path:
    """Return the metadata file path for *folder*.

    Parameters
    ----------
    folder : pathlib.Path
        The folder containing TTTR images.

    Returns
    -------
    pathlib.Path
        The file path to the metadata file.
    """
    return folder / META_FILENAME


def load_meta(folder: pathlib.Path) -> dict[str, dict]:
    """Load image-browser metadata from *folder*.

    Parameters
    ----------
    folder : pathlib.Path
        The folder containing TTTR images.

    Returns
    -------
    dict[str, dict]
        The metadata dict mapping relative paths/filenames to record dicts.
    """
    path = meta_path(folder)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_meta(folder: pathlib.Path, data: dict[str, dict]) -> None:
    """Atomically save image-browser metadata.

    Parameters
    ----------
    folder : pathlib.Path
        The folder containing TTTR images.
    data : dict[str, dict]
        The metadata dict to save.
    """
    path = meta_path(folder)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if path.exists():
        path.unlink()
    tmp.replace(path)
