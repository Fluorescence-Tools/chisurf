"""Data models for the BVA API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BvaSettings:
    """Burst Variance Analysis settings.

    Attributes
    ----------
    donor_channels : list of int
        TCSPC routing channel numbers for the donor.
    donor_micro_time_ranges : list of tuple[int, int]
        Micro time ranges for the donor.
    acceptor_channels : list of int
        TCSPC routing channel numbers for the acceptor.
    acceptor_micro_time_ranges : list of tuple[int, int]
        Micro time ranges for the acceptor.
    minimum_window_length : float
        Minimum burst slicing window in seconds.
    number_of_photons_per_slice : int
        Fixed photon count per slice (negative = use time windows).
    file_type : str
        tttrlib container name (e.g. ``"SPC-130"``).

    """

    donor_channels: list[int] = field(default_factory=lambda: [0, 8])
    donor_micro_time_ranges: list[tuple[int, int]] = field(default_factory=lambda: [(0, 32768)])
    acceptor_channels: list[int] = field(default_factory=lambda: [1, 9])
    acceptor_micro_time_ranges: list[tuple[int, int]] = field(default_factory=lambda: [(0, 32768)])
    minimum_window_length: float = 0.01
    number_of_photons_per_slice: int = 10
    file_type: str = "SPC-130"


@dataclass
class BvaResult:
    """Result of a BVA analysis.

    Attributes
    ----------
    files : list of str
        Original TTTR file paths.
    n_bursts_total : int
        Total number of bursts processed.
    n_bursts_valid : int
        Bursts with Std > 0.
    output_paths : dict
        Paths written by the analysis.
    settings_applied : dict
        The settings that were used.

    """

    files: list[str]
    n_bursts_total: int = 0
    n_bursts_valid: int = 0
    output_paths: dict[str, str] = field(default_factory=dict)
    settings_applied: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
