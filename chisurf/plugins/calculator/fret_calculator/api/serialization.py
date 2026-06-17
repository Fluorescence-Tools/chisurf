"""JSON serialization helpers for the FRET Calculator."""

from __future__ import annotations

from typing import Any

from .models import FretResult, FretSettings, HomoFretResult, HomoFretSettings


def fret_settings_from_dict(data: dict[str, Any]) -> FretSettings:
    """Deserialize FRET settings from a JSON-compatible dict."""
    return FretSettings.from_dict(data)


def homo_fret_settings_from_dict(data: dict[str, Any]) -> HomoFretSettings:
    """Deserialize homoFRET settings from a JSON-compatible dict."""
    return HomoFretSettings.from_dict(data)


def fret_result_from_dict(data: dict[str, Any]) -> FretResult:
    """Deserialize FRET result from a JSON-compatible dict."""
    return FretResult.from_dict(data)


def homo_fret_result_from_dict(data: dict[str, Any]) -> HomoFretResult:
    """Deserialize homoFRET result from a JSON-compatible dict."""
    return HomoFretResult.from_dict(data)
