"""Qt-free core for the FCS-Merger plugin (re-exports the shared merge core)."""

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
