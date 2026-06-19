"""Core fps.json payload inspection helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .model import FpsJsonModel


def normalize_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a canonical fps.json payload through the editor model.

    Parameters
    ----------
    payload : Mapping or None
        Input fps.json-like payload.

    Returns
    -------
    dict
        Canonical payload with standard top-level sections.
    """
    model = FpsJsonModel()
    model.fps_json_payload = dict(payload or {})
    return model.fps_json_payload


def summarize_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return counts and reference diagnostics for an fps.json payload.

    Parameters
    ----------
    payload : Mapping or None
        Input fps.json-like payload.

    Returns
    -------
    dict
        JSON-serializable summary of positions, distances, score sets, and
        unresolved references.
    """
    model = FpsJsonModel()
    model.fps_json_payload = dict(payload or {})

    position_names = set(model.positions)
    used_positions = set()
    dangling_distances = []

    for distance_name, distance in model.distances.items():
        if not isinstance(distance, Mapping):
            dangling_distances.append(
                {
                    "distance": distance_name,
                    "missing_positions": ["<invalid-distance-payload>"],
                }
            )
            continue

        missing = []
        for field in ("position1_name", "position2_name"):
            position_name = distance.get(field)
            if not position_name:
                missing.append(field)
            elif position_name in position_names:
                used_positions.add(str(position_name))
            else:
                missing.append(str(position_name))
        if missing:
            dangling_distances.append(
                {"distance": distance_name, "missing_positions": missing}
            )

    missing_score_set_distances = []
    for score_set_name, score_set in model.score_sets.items():
        if not isinstance(score_set, Mapping):
            missing_score_set_distances.append(
                {
                    "score_set": score_set_name,
                    "missing_distances": ["<invalid-score-set-payload>"],
                }
            )
            continue
        missing = [
            str(distance_name)
            for distance_name in score_set.get("distances", [])
            if distance_name not in model.distances
        ]
        if missing:
            missing_score_set_distances.append(
                {"score_set": score_set_name, "missing_distances": missing}
            )

    return {
        "format_version": model.format_version,
        "n_positions": len(model.positions),
        "n_distances": len(model.distances),
        "n_score_sets": len(model.score_sets),
        "n_extra_sections": len(model.extra_sections),
        "position_names": sorted(model.positions),
        "distance_names": sorted(model.distances),
        "score_set_names": sorted(model.score_sets),
        "extra_section_names": sorted(model.extra_sections),
        "used_positions": sorted(used_positions),
        "unused_positions": sorted(position_names - used_positions),
        "dangling_distances": dangling_distances,
        "missing_score_set_distances": missing_score_set_distances,
    }


def validate_payload(payload: Any) -> dict[str, Any]:
    """Validate an fps.json payload enough for editor/RPC workflows.

    Parameters
    ----------
    payload : Any
        Input object to validate.

    Returns
    -------
    dict
        Validation result with ``valid``, ``errors``, ``warnings``, and
        ``summary`` keys.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(payload, Mapping):
        return {
            "valid": False,
            "errors": ["payload must be a JSON object"],
            "warnings": [],
            "summary": summarize_payload({}),
        }

    positions = payload.get("Positions", {})
    distances = payload.get("Distances", {})
    score_sets = payload.get("χ²", {})

    if not isinstance(positions, Mapping):
        errors.append("Positions must be an object")
        positions = {}
    if not isinstance(distances, Mapping):
        errors.append("Distances must be an object")
        distances = {}
    if score_sets and not isinstance(score_sets, Mapping):
        errors.append("χ² must be an object when present")
        score_sets = {}

    normalized = dict(payload)
    normalized["Positions"] = dict(positions)
    normalized["Distances"] = dict(distances)
    if score_sets:
        normalized["χ²"] = dict(score_sets)
    summary = summarize_payload(normalized)

    for item in summary["dangling_distances"]:
        missing = ", ".join(item["missing_positions"])
        errors.append(f"Distance '{item['distance']}' references missing position(s): {missing}")

    for item in summary["missing_score_set_distances"]:
        missing = ", ".join(item["missing_distances"])
        errors.append(f"Score set '{item['score_set']}' references missing distance(s): {missing}")

    if summary["unused_positions"]:
        warnings.append(
            "Unused position(s): " + ", ".join(summary["unused_positions"])
        )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


__all__ = ["normalize_payload", "summarize_payload", "validate_payload"]
