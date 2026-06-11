"""Serialization helpers for burst-selection settings and results."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from .models import AnalysisSettings

T = TypeVar("T")


def to_jsonable(value: Any) -> Any:
    """Convert settings/results to JSON-serializable objects."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _convert_value(field_type: type[Any], value: Any) -> Any:
    """Convert one JSON-compatible value according to its dataclass field type."""
    origin = get_origin(field_type)
    if value is None:
        if origin is list:
            return []
        if origin is dict:
            return {}
        return None
    if is_dataclass(field_type):
        return from_jsonable(field_type, value)
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return field_type(value)
    if origin is list:
        inner = get_args(field_type)[0]
        return [_convert_value(inner, item) for item in value]
    if origin is tuple:
        args = get_args(field_type)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert_value(args[0], item) for item in value)
        return tuple(
            _convert_value(item_type, item)
            for item_type, item in zip(args, value, strict=False)
        )
    if origin is dict:
        key_type, value_type = get_args(field_type)
        return {
            key_type(key): _convert_value(value_type, item)
            for key, item in value.items()
        }
    return value


def from_jsonable(cls: type[T], data: dict[str, Any]) -> T:
    """Build a dataclass instance from a JSON-compatible dictionary."""
    values: dict[str, Any] = {}
    hints = get_type_hints(cls)
    for field in fields(cls):
        if field.name not in data:
            continue
        values[field.name] = _convert_value(hints[field.name], data[field.name])
    return cls(**values)


def settings_from_dict(data: dict[str, Any]) -> AnalysisSettings:
    """Create analysis settings from a JSON-compatible dictionary."""
    return from_jsonable(AnalysisSettings, data)
