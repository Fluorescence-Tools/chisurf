"""Parse mmCIF_PDBX v5 dictionary for metadata key suggestions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DICT_PATH = Path(__file__).resolve().parent / "data" / "mmcif_pdbx_v50.dic"
_cache_keys: Optional[List[str]] = None
_cache_descriptions: Optional[Dict[str, str]] = None


def _parse_save_category_name(line: str) -> Optional[str]:
    """Extract category name from a ``save_<name>`` line."""
    rest = line[5:].strip()
    if rest and not rest.startswith("_"):
        return rest
    return None


def _parse_item_name(line: str) -> Optional[str]:
    """Extract ``_category.attribute`` from a ``_item.name`` value."""
    idx = line.find("_item.name")
    if idx < 0:
        return None
    rest = line[idx + 10:].strip().strip("'").strip('"')
    if rest.startswith("_"):
        return rest
    return None


def _parse_item_description(line: str) -> Optional[str]:
    """Extract description text from ``_item.description`` value."""
    idx = line.find("_item.description")
    if idx < 0:
        return None
    rest = line[idx + 17:].strip().strip("'").strip('"')
    return rest if rest else None


def parse_pdbx_keys(path: Optional[Path] = None) -> List[str]:
    """Return all ``_category.attribute`` keys from the PDBx dictionary.

    Parameters
    ----------
    path : Path, optional
        Path to the mmCIF dictionary file.  Defaults to the bundled
        ``mmcif_pdbx_v50.dic``.

    Returns
    -------
    list of str
        Sorted unique attribute names.
    """
    keys, _ = _parse_pdbx(path=path)
    return keys


def parse_pdbx_descriptions(path: Optional[Path] = None) -> Dict[str, str]:
    """Return a mapping of ``_category.attribute`` keys to descriptions.

    Parameters
    ----------
    path : Path, optional
        Path to the mmCIF dictionary file.

    Returns
    -------
    dict
        Key -> description text.
    """
    _, descriptions = _parse_pdbx(path=path)
    return descriptions


def _parse_pdbx(path: Optional[Path] = None) -> Tuple[List[str], Dict[str, str]]:
    """Parse all keys and descriptions from the PDBx dictionary."""
    p = path or DICT_PATH
    if not p.exists():
        return [], {}
    keys: Set[str] = set()
    descriptions: Dict[str, str] = {}
    current_save: Optional[str] = None
    pending_desc: Optional[str] = None
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("save_"):
                name = _parse_save_category_name(line)
                current_save = name if name else None
                if not name:
                    pending_desc = None
                continue
            if line.startswith("_item.description"):
                pending_desc = _parse_item_description(line)
                continue
            if line.startswith("_item.name"):
                full = _parse_item_name(line)
                if full:
                    keys.add(full)
                    if pending_desc:
                        descriptions[full] = pending_desc
                pending_desc = None
    return sorted(keys), descriptions


def get_pdbx_metadata_keys() -> List[str]:
    """Return cached PDBx metadata keys.

    Results are cached after the first parse.
    """
    global _cache_keys, _cache_descriptions
    if _cache_keys is None:
        keys, descs = _parse_pdbx()
        _cache_keys = keys
        _cache_descriptions = descs
    return list(_cache_keys)


def get_pdbx_metadata_descriptions() -> Dict[str, str]:
    """Return cached PDBx key -> description mapping."""
    global _cache_keys, _cache_descriptions
    if _cache_descriptions is None:
        keys, descs = _parse_pdbx()
        _cache_keys = keys
        _cache_descriptions = descs
    return dict(_cache_descriptions)


def pdbx_category_names() -> List[str]:
    """Return unique category names (without attributes)."""
    keys = get_pdbx_metadata_keys()
    cats: Set[str] = set()
    for k in keys:
        parts = k.split(".", 1)
        if len(parts) == 2:
            cats.add(parts[0])
    return sorted(cats)
