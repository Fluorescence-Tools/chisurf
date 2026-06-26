"""Programmatic API for the FCS-Merger plugin (Qt-free).

Thin facade over the shared core merge primitives in
``chisurf.core.fluorescence.fcs.merge``.
"""

from __future__ import annotations

from chisurf.core.fluorescence.fcs.merge import (
    compute_average_correlations,
    merge_folder,
    parse_correlation_folder,
    save_mean_correlation,
)

__all__ = [
    "compute_average_correlations",
    "merge_folder",
    "parse_correlation_folder",
    "save_mean_correlation",
]
