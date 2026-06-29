"""Offline tests for the ATTO (Wayback) value parsing.

The ATTO pages report properties with German decimal commas, an εmax exponent
notation ("9,0×104") and the quantum yield as a percent — these must normalise
to canonical numbers. No network.
"""
from __future__ import annotations

from chisurf.plugins.spectra_downloader.download.atto import _clean_property


def test_clean_german_decimal():
    assert _clean_property("τfl", "4,1") == "4.1"
    assert _clean_property("CF260", "0,22") == "0.22"


def test_clean_epsilon_exponent():
    # 9,0×104 → 9.0 × 10^4 = 90000
    assert float(_clean_property("εmax", "9,0×104")) == 90000.0
    assert float(_clean_property("εmax", "1,2×105")) == 120000.0


def test_clean_quantum_yield_percent():
    # ηfl is a percentage → fraction
    assert _clean_property("ηfl", "80") == "0.8"
    assert _clean_property("ηfl", "75") == "0.75"


def test_plain_value_unchanged():
    assert _clean_property("λabs", "500") == "500"
    assert _clean_property("λfl", "520") == "520"
