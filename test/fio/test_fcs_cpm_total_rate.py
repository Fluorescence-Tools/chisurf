from __future__ import annotations

import pytest

from chisurf.core.fluorescence.fcs import resolve_total_mean_count_rate


def test_resolve_total_rate_prefers_explicit_total() -> None:
    meta = {
        "mean_count_rate_total": 120.0,
        "mean_count_rate": 60.0,
        "mean_count_rate_semantics": "per_detector_mean",
        "detector_count": 2,
    }
    assert resolve_total_mean_count_rate(meta) == pytest.approx(120.0)


def test_resolve_total_rate_from_per_detector_semantics() -> None:
    meta = {
        "mean_count_rate": 55.0,
        "mean_count_rate_semantics": "per_detector_mean",
        "detector_count": 2,
    }
    assert resolve_total_mean_count_rate(meta) == pytest.approx(110.0)


def test_resolve_total_rate_falls_back_to_mean_rate() -> None:
    meta = {
        "mean_count_rate": 42.0,
    }
    assert resolve_total_mean_count_rate(meta) == pytest.approx(42.0)


def test_resolve_total_rate_returns_none_for_invalid_meta() -> None:
    assert resolve_total_mean_count_rate({}) is None
    assert resolve_total_mean_count_rate({"mean_count_rate": "nan"}) is None
    assert resolve_total_mean_count_rate(None) is None
