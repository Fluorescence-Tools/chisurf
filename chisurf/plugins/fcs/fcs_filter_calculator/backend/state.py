"""Persisted state schema for the FCS filter calculator plugin."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List


@dataclasses.dataclass
class FilterCalcState:
    """Last-used file selections."""

    total_path: str = ""
    species_paths: List[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FilterCalcState":
        d = d or {}
        return cls(
            total_path=str(d.get("total_path", "")),
            species_paths=list(d.get("species_paths", []) or []),
        )
