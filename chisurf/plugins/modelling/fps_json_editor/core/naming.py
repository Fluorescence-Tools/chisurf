"""Naming helpers for FPS labeling positions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def default_label_name(chain: Any, residue: Any) -> str:
    """Return the default FPS label name for a chain/residue pair.

    Parameters
    ----------
    chain : Any
        Chain identifier.
    residue : Any
        Residue sequence number.

    Returns
    -------
    str
        Compact label name such as ``A132`` or ``E5``. If either component is
        missing, the available component is returned.
    """
    chain_text = str(chain).strip() if chain is not None else ""
    residue_text = str(residue).strip() if residue is not None else ""
    if chain_text and residue_text:
        return f"{chain_text}{residue_text}"
    return chain_text or residue_text


def unique_label_name(base_name: str, existing_names: Iterable[str]) -> str:
    """Return a unique label name by appending a numeric suffix if needed.

    Parameters
    ----------
    base_name : str
        Preferred label name.
    existing_names : Iterable[str]
        Names that are already used.

    Returns
    -------
    str
        ``base_name`` or ``base_name_N`` when the base name is already used.
    """
    base = str(base_name).strip()
    existing = {str(name).strip() for name in existing_names if str(name).strip()}
    if not base or base not in existing:
        return base

    counter = 2
    candidate = f"{base}_{counter}"
    while candidate in existing:
        counter += 1
        candidate = f"{base}_{counter}"
    return candidate


__all__ = ["default_label_name", "unique_label_name"]
