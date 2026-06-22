"""I/O helpers for TTTR Image Browser."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable
from typing import Any

from chisurf.plugins.tttr.tttr_image_browser.core.metadata import load_meta

try:
    import tttrlib
except Exception:
    tttrlib = None


def get_tttr_supported_exts() -> list[str]:
    """Get supported TTTR extensions from tttrlib.

    Returns
    -------
    list[str]
        Dotted lowercase extensions.
    """
    exts: list[str] = []
    try:
        if tttrlib is not None and hasattr(tttrlib, "get_supported_filetypes"):
            exts = list(tttrlib.get_supported_filetypes())
    except Exception:
        exts = []

    norm: list[str] = []
    for e in exts:
        s = str(e).strip().lower()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        if s not in norm:
            norm.append(s)
    return norm


def allowed_exts_for_setup(setup_settings: dict[str, Any] | None) -> set[str]:
    """Get allowed extensions for a given setup definition.

    Parameters
    ----------
    setup_settings : dict, optional
        Detector setup settings.

    Returns
    -------
    set[str]
        Set of dotted lowercase extensions.
    """
    try:
        reading = setup_settings.get("tttr_reading", {}) if isinstance(setup_settings, dict) else {}
        filetype = reading.get("file_type") or None
    except Exception:
        filetype = None
    all_exts = set(get_tttr_supported_exts())
    if not filetype or str(filetype).strip().lower() == "auto":
        return all_exts
    ft = str(filetype).strip().upper()
    mapping = {
        "PTU": {".ptu"},
        "PT3": {".pt3"},
        "HT3": {".ht3"},
        "PT2": {".pt2"},
        "PT5": {".pt5"},
        "SPC-130": {".spc"},
        "SPC-600": {".spc"},
        "SPC-830": {".spc"},
        "PHU": {".phu"},
        "PHOTON_HDF5": {".h5", ".hdf5", ".photon.hdf5"},
        "HDF5": {".h5", ".hdf5"},
    }
    exts = mapping.get(ft)
    if exts:
        return {e for e in exts if (not all_exts or e in all_exts)} or exts
    return all_exts


def human_size(size_bytes: int) -> str:
    """Return a human-readable file size string.

    Parameters
    ----------
    size_bytes : int
        Size in bytes.

    Returns
    -------
    str
        Size formatted in MB.
    """
    try:
        return f"{float(size_bytes) / (1024.0 * 1024.0):.1f} MB"
    except Exception:
        return "? MB"


def iter_image_files(folder: pathlib.Path, recursive: bool = False, extensions: Iterable[str] | None = None) -> list[pathlib.Path]:
    """Return matching image files under folder.

    Parameters
    ----------
    folder : pathlib.Path
        The root folder to list.
    recursive : bool, optional
        Whether to search subfolders recursively.
    extensions : Iterable of str, optional
        Filtering extensions.

    Returns
    -------
    list of pathlib.Path
        Sorted list of matching file paths.
    """
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in (extensions or get_tttr_supported_exts())}
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    files: list[pathlib.Path] = []
    for path in iterator:
        try:
            if not path.is_file():
                continue
            # Exclude anything under a hidden .trash within the folder
            try:
                relp = path.resolve().relative_to(folder.resolve())
                if any(part == ".trash" for part in relp.parts):
                    continue
            except Exception:
                pass
            if path.suffix.lower() in exts:
                files.append(path)
        except Exception:
            continue
    return sorted(files, key=lambda item: item.name.lower())


def list_files(folder: str, recursive: bool = False, setup_settings: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """List TTTR files in folder with metadata summaries.

    Parameters
    ----------
    folder : str
        The root folder.
    recursive : bool, optional
        Whether to list recursively.
    setup_settings : dict, optional
        Setup settings.

    Returns
    -------
    list of dict
        File details.
    """
    root = pathlib.Path(folder)
    meta = load_meta(root)
    allowed_exts = allowed_exts_for_setup(setup_settings)
    rows: list[dict[str, Any]] = []
    for path in iter_image_files(root, recursive=recursive, extensions=allowed_exts):
        # We need a stable key. Since metadata keys can be relative to the folder or name, check both:
        try:
            key = str(path.resolve().relative_to(root.resolve()))
        except Exception:
            key = path.name
        file_meta = meta.get(key) or meta.get(path.name) or {}
        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "size": path.stat().st_size,
                "size_text": human_size(path.stat().st_size),
                "rating": int(file_meta.get("rating", 0)),
                "annotation": str(file_meta.get("annotation", "")),
            }
        )
    return rows
