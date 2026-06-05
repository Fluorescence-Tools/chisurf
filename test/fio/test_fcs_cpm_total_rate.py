from __future__ import annotations

import pytest


try:
    from chisurf.core.models.fcs.fcs import ParseFCSWidget
except Exception as exc:  # pragma: no cover - environment-dependent import guard
    pytestmark = pytest.mark.skip(reason=f"chisurf FCS model import unavailable: {exc}")
    ParseFCSWidget = None  # type: ignore[assignment]


def _widget() -> ParseFCSWidget:
    # Bypass full Qt/model initialization; tests target metadata resolution only.
    assert ParseFCSWidget is not None
    return ParseFCSWidget.__new__(ParseFCSWidget)


def test_resolve_total_rate_prefers_explicit_total() -> None:
    w = _widget()
    meta = {
        "mean_count_rate_total": 120.0,
        "mean_count_rate": 60.0,
        "mean_count_rate_semantics": "per_detector_mean",
        "detector_count": 2,
    }
    assert w._resolve_total_mean_count_rate(meta) == pytest.approx(120.0)


def test_resolve_total_rate_from_per_detector_semantics() -> None:
    w = _widget()
    meta = {
        "mean_count_rate": 55.0,
        "mean_count_rate_semantics": "per_detector_mean",
        "detector_count": 2,
    }
    assert w._resolve_total_mean_count_rate(meta) == pytest.approx(110.0)


def test_resolve_total_rate_falls_back_to_mean_rate() -> None:
    w = _widget()
    meta = {
        "mean_count_rate": 42.0,
    }
    assert w._resolve_total_mean_count_rate(meta) == pytest.approx(42.0)


def test_resolve_total_rate_returns_none_for_invalid_meta() -> None:
    w = _widget()
    assert w._resolve_total_mean_count_rate({}) is None
    assert w._resolve_total_mean_count_rate({"mean_count_rate": "nan"}) is None
    assert w._resolve_total_mean_count_rate(None) is None
