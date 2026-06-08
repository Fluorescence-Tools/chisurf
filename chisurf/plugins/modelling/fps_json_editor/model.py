"""Data model for fps.json documents.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import chisurf.core.fio as io


_RESERVED = {"Distances", "Positions", "χ²"}
_FLEXFIT_KEYS = {"Flexible residues", "Bonds"}


def _int_or(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


class FpsJsonModel:
    """Central data model for a fps.json document.

    Owns positions, distances, score_sets, and extra_sections.
    Provides a single fps_json_payload getter/setter for embedding
    the editor inside other widgets without file I/O.
    """

    def __init__(self) -> None:
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.distances: Dict[str, Dict[str, Any]] = {}
        self.score_sets: Dict[str, Dict[str, Any]] = {}
        self.extra_sections: Dict[str, Any] = {}

    @property
    def fps_json_payload(self) -> dict:
        """Return full fps.json payload as a Python dict (no file I/O)."""
        p = dict(self.extra_sections)
        p["Distances"] = self.distances
        p["Positions"] = self.positions
        if self.score_sets:
            p["χ²"] = self.score_sets
        return p

    @fps_json_payload.setter
    def fps_json_payload(self, payload: dict) -> None:
        """Load a full fps.json payload dict into the model."""
        if not isinstance(payload, dict):
            payload = {}
        self.positions = dict(payload.get("Positions", {}) or {})
        self.distances = dict(payload.get("Distances", {}) or {})
        self.score_sets = dict(payload.get("χ²", {}) or {})
        self.extra_sections = {
            k: v for k, v in payload.items() if k not in _RESERVED
        }

    def load_file(self, path: str) -> None:
        """Read an fps.json file into the model."""
        with io.zipped.open_maybe_zipped(filename=path, mode='r') as fp:
            payload = json.load(fp)
        self.fps_json_payload = payload

    def save_file(self, path: str) -> None:
        """Write the model to an fps.json file."""
        payload = self.fps_json_payload
        with open(path, "w") as fp:
            json.dump(payload, fp, sort_keys=True, indent=4, separators=(',', ': '))

    def add_position(self, name: str, params: Dict[str, Any]) -> None:
        """Add or update a labeling position in the model."""
        self.positions[name] = params

    def remove_position(self, name: str) -> None:
        """Remove a position and all distances referencing it."""
        if name in self.positions:
            del self.positions[name]
        
        # Remove any distance referencing this position
        to_remove = self.distances_referencing(name)
        for dist_name in to_remove:
            self.remove_distance(dist_name)

    def add_distance(self, name: str, params: Dict[str, Any], score_set: Optional[str] = None) -> None:
        """Add or update a distance restraint in the model."""
        self.distances[name] = params
        if score_set and score_set in self.score_sets:
            group = self.score_sets[score_set]
            if isinstance(group, dict):
                group.setdefault("distances", [])
                if name not in group["distances"]:
                    group["distances"].append(name)

    def remove_distance(self, name: str) -> None:
        """Remove a distance restraint and clean it up from score sets."""
        if name in self.distances:
            del self.distances[name]
        self._cleanup_score_sets(name)

    def add_score_set(self, name: str) -> None:
        """Add a new score set (scoring group)."""
        if name not in self.score_sets:
            self.score_sets[name] = {"distances": []}

    def remove_score_set(self, name: str) -> None:
        """Remove a score set from the model."""
        if name in self.score_sets:
            del self.score_sets[name]

    def distances_referencing(self, position_name: str) -> List[str]:
        """Return distance keys whose position1_name or position2_name matches."""
        return [
            dn for dn, d in self.distances.items()
            if d.get("position1_name") == position_name
            or d.get("position2_name") == position_name
        ]

    def _cleanup_score_sets(self, distance_name: str) -> None:
        """Remove distance_name from every score group's distance list."""
        for group in self.score_sets.values():
            if isinstance(group, dict) and "distances" in group:
                group["distances"] = [
                    d for d in group["distances"] if d != distance_name
                ]
