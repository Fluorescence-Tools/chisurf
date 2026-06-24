"""Tests for the pure shift logic in api/shift.py."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tttrlib

from chisurf.plugins.tttr.tttr_microtime_shifter.api.shift import (
    _apply_shifts,
    _compute_effective_shifts,
    _load_tttr,
    load_file_metadata,
    safe_tttr_path,
    shift_file,
)


def _find_test_ptu() -> str | None:
    """Return the path of a small PTU file for testing."""
    candidates = [
        "/Users/tpeulen/dev/chisurf/test/data/auto_nanosecond_after_1.ptu",
        "/Users/tpeulen/dev/chisurf/test/data/auto_nanosecond_prompt_1.ptu",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


@pytest.mark.skipif(not _find_test_ptu(), reason="No test PTU file available")
def test_shift_file_roundtrip() -> None:
    """Read->shift->write->read produces expected micro-time values."""
    src = _find_test_ptu()
    tt_orig = tttrlib.TTTR(src)
    n_mt = tt_orig.header.get_effective_number_of_micro_time_channels()
    routing = tt_orig.routing_channels
    used = sorted(set(int(c) for c in routing))

    global_shift = 3
    channel_shifts: dict[int, int] = {}
    for i, ch in enumerate(used):
        channel_shifts[ch] = i + 1

    with tempfile.TemporaryDirectory() as tmp:
        out_path, applied = shift_file(
            src,
            global_shift=global_shift,
            channel_shifts=channel_shifts,
            output_dir=tmp,
        )
        assert os.path.isfile(out_path)

        tt_shifted = tttrlib.TTTR(out_path)
        new_mt = tt_shifted.micro_times
        new_routing = tt_shifted.routing_channels

        for ch in used:
            mask_orig = routing == ch
            mask_new = new_routing == ch
            expected_shift = (global_shift + channel_shifts.get(ch, 0)) % n_mt
            if expected_shift == 0:
                assert np.array_equal(
                    new_mt[mask_new],
                    tt_orig.micro_times[mask_orig],
                ), f"Channel {ch} should be unshifted"
            else:
                shifted_orig = (tt_orig.micro_times[mask_orig] + expected_shift) % n_mt
                assert np.array_equal(
                    new_mt[mask_new],
                    shifted_orig,
                ), f"Channel {ch} shift mismatch"

        assert applied == {int(k): (global_shift + channel_shifts.get(int(k), 0)) % n_mt for k in used}


def test_compute_effective_shifts() -> None:
    """Effective shifts are computed correctly."""
    routing = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    effective = _compute_effective_shifts(2, {0: 1, 1: -1}, routing, 10)
    assert effective == {0: 3, 1: 1, 2: 2}


def test_compute_effective_shifts_no_per_channel() -> None:
    """Without per-channel shifts, only global shift is applied."""
    routing = np.array([0, 1], dtype=np.int32)
    effective = _compute_effective_shifts(5, {}, routing, 10)
    assert effective == {0: 5, 1: 5}


def test_compute_effective_shifts_modulo() -> None:
    """Shifts wrap around at n_mt."""
    routing = np.array([0], dtype=np.int32)
    effective = _compute_effective_shifts(10, {}, routing, 8)
    assert effective == {0: 2}


@pytest.mark.skipif(sys.platform == "win32", reason="Safe path test for non-Windows")
def test_safe_tttr_path_ascii() -> None:
    """ASCII paths pass through unchanged on non-Windows."""
    path = "/tmp/test/ptu/file.ptu"
    result = safe_tttr_path(path)
    assert result == path


def test_load_file_metadata_returns_dict() -> None:
    """load_file_metadata returns a dict with expected keys for a real file."""
    src = _find_test_ptu()
    if not src:
        pytest.skip("No test PTU file available")
    meta = load_file_metadata(src)
    assert isinstance(meta, dict)
    assert "routing_channels" in meta
    assert "n_mt" in meta
    assert "n_photons" in meta
    assert isinstance(meta["routing_channels"], list)
    assert meta["n_mt"] > 0
    assert meta["n_photons"] > 0
