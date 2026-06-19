"""Load vocabulary data from JSON files.

This module provides functions to load vocabulary constants from JSON data files,
separating vocabulary data from code for easier maintenance and updates.

The vocabulary data files are stored in chisurf/core/mfdb/data/*.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

# Path to the data directory
_DATA_DIR = Path(__file__).resolve().parent / "data"

# Cache for loaded vocabulary data
_vocabulary_cache: dict[str, Tuple[str, ...]] = {}


def _load_vocabulary(filename: str, use_cache: bool = True) -> Tuple[str, ...]:
    """Load vocabulary from a JSON file and return as a tuple.

    Parameters
    ----------
    filename : str
        Name of the JSON file in the data directory (without .json extension).
    use_cache : bool, optional
        Whether to use cached values. Default is True.

    Returns
    -------
    tuple of str
        Vocabulary entries as a tuple.

    Raises
    ------
    FileNotFoundError
        If the vocabulary file does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    ValueError
        If the file does not contain a JSON array.

    """
    if use_cache and filename in _vocabulary_cache:
        return _vocabulary_cache[filename]

    filepath = _DATA_DIR / f"{filename}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Vocabulary file {filename} must contain a JSON array")

    # Convert to tuple and cache
    vocabulary = tuple(data)
    if use_cache:
        _vocabulary_cache[filename] = vocabulary
    return vocabulary


def get_entity_types() -> Tuple[str, ...]:
    """Return the ENTITY_TYPES vocabulary.

    Returns
    -------
    tuple of str
        Entity type vocabulary (protein, dna, rna, etc.).

    """
    return _load_vocabulary("entity_types")


def get_probe_names() -> Tuple[str, ...]:
    """Return the COMMON_PROBE_NAMES vocabulary.

    Returns
    -------
    tuple of str
        Common probe/fluorophore name vocabulary.

    """
    return _load_vocabulary("probe_names")


def get_buffer_components() -> Tuple[str, ...]:
    """Return the BUFFER_COMPONENTS vocabulary.

    Returns
    -------
    tuple of str
        Buffer component vocabulary.

    """
    return _load_vocabulary("buffer_components")


def get_sample_condition_fields() -> Tuple[str, ...]:
    """Return the SAMPLE_CONDITION_FIELDS vocabulary.

    Returns
    -------
    tuple of str
        Sample condition field vocabulary.

    """
    return _load_vocabulary("sample_condition_fields")


def reload_vocabulary() -> None:
    """Clear the vocabulary cache, forcing a reload on next access.

    This is useful for testing or when vocabulary files are updated at runtime.

    """
    global _vocabulary_cache
    _vocabulary_cache.clear()


def get_all_vocabulary_names() -> list[str]:
    """Return list of available vocabulary names.

    Returns
    -------
    list of str
        Names of all available vocabulary files (without .json extension).

    """
    if not _DATA_DIR.exists():
        return []

    return [
        f.stem for f in _DATA_DIR.glob("*.json")
        if f.is_file() and not f.name.startswith("_")
    ]
