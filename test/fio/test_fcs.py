from __future__ import annotations

# Consolidated test file: test_fcs.py


# --- FROM test_fcs_setup.py ---
# Test script to verify that changing to an FCS setup doesn't crash
# This script should be run in the chisurf console

# # First, set the experiment to FCS
# cs.current_experiment = 'FCS'
# 
# # Then, try to set a setup
# # This should not crash
# cs.current_setup = 'CSV'
# 
# # Print a message to confirm that the script completed successfully
# print("FCS setup changed successfully")
# --- FROM test_fcs_cpm_total_rate.py ---

import pytest


try:
    from chisurf.models.fcs.fcs import ParseFCSWidget
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

# --- FROM test_fio_fcs_sin.py ---

import pathlib

import pytest

import chisurf
import chisurf.fio.fluorescence.fcs as fcs_io
import chisurf.fio.fluorescence.fcs.sin_correlator as sin_reader


DATA_ROOT = pathlib.Path(__file__).parent / "data" / "fcs"
if DATA_ROOT.is_dir():
    SIN_FILES = list(DATA_ROOT.rglob("*.sin"))
else:
    SIN_FILES = []

# If no correlator.com SIN example files are available in the test data
# directory, leave the SIN reader effectively untested but do not fail the
# test suite.
pytestmark = pytest.mark.skipif(
    len(SIN_FILES) == 0,
    reason="No correlator.com .sin test files found; SIN reader left untested.",
)


def _assert_fcs_dataset_dict(ds: dict) -> None:
    """Lightweight structural checks for a FCSDataset-style dict.

    This intentionally does not check numerical values, only that required
    keys are present and array-like fields have consistent lengths.
    """

    required_keys = [
        "filename",
        "measurement_id",
        "acquisition_time",
        "mean_count_rate",
        "correlation_times",
        "correlation_amplitudes",
        "correlation_amplitude_weights",
    ]
    for key in required_keys:
        assert key in ds, f"missing key '{key}' in FCSDataset"

    t = ds["correlation_times"]
    g = ds["correlation_amplitudes"]
    w = ds["correlation_amplitude_weights"]
    assert len(t) == len(g) == len(w)


def test_read_sin_low_level() -> None:
    """Low-level reader should return a non-empty list of FCSDataset dicts."""

    fn = SIN_FILES[0]
    ds_list = sin_reader.read_sin(str(fn), verbose=False)

    assert isinstance(ds_list, list)
    assert ds_list, "read_sin should return at least one dataset"

    for ds in ds_list:
        _assert_fcs_dataset_dict(ds)


def test_read_sin_via_read_fcs_dispatcher() -> None:
    """High-level read_fcs dispatcher should wrap SIN datasets into DataCurves."""

    fn = SIN_FILES[0]
    group = fcs_io.read_fcs(filename=str(fn), reader_name="sin")

    # Expect some curves in the ExperimentDataCurveGroup.
    data = getattr(group, "data", None)
    assert data is not None
    assert len(data) > 0

    for curve in data:
        # DataCurve should at least have x, y, ey arrays of matching length.
        assert hasattr(curve, "x")
        assert hasattr(curve, "y")
        assert hasattr(curve, "ey")
        n = curve.x.size
        assert n > 0
        assert curve.y.size == n
        assert curve.ey.size == n
