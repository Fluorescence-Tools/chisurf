"""Color helpers for fps.json position metadata."""

from __future__ import annotations

from typing import Any

DEFAULT_AV_COLOR: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.35)


def normalize_rgba(
    value: Any,
    default: tuple[float, float, float, float] = DEFAULT_AV_COLOR,
) -> tuple[float, float, float, float]:
    """Return a normalized RGBA tuple with float channels in ``[0, 1]``.

    Parameters
    ----------
    value : object
        Sequence of three or four color channels. Values greater than ``1`` are
        interpreted as 8-bit channels.
    default : tuple of float, optional
        Color returned when *value* cannot be interpreted.

    Returns
    -------
    tuple of float
        Normalized ``(red, green, blue, alpha)``.
    """
    try:
        channels = list(value)
    except TypeError:
        return default
    if len(channels) not in (3, 4):
        return default
    try:
        rgba = [float(v) for v in channels[:4]]
    except (TypeError, ValueError):
        return default
    if len(rgba) == 3:
        rgba.append(default[3])
    if any(v > 1.0 for v in rgba):
        rgba = [v / 255.0 if v > 1.0 else v for v in rgba]
    return tuple(max(0.0, min(1.0, v)) for v in rgba)  # type: ignore[return-value]


def rgba_to_json(value: Any) -> list[float]:
    """Return an fps.json-safe RGBA list.

    Parameters
    ----------
    value : object
        Color representation accepted by :func:`normalize_rgba`.

    Returns
    -------
    list of float
        Four float channels rounded for stable JSON output.
    """
    return [round(v, 4) for v in normalize_rgba(value)]


__all__ = ["DEFAULT_AV_COLOR", "normalize_rgba", "rgba_to_json"]
