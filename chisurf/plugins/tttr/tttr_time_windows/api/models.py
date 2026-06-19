"""Data models for the Time Window Bins API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TimeWindowRequest:
    """Workflow input for TTTR→time-window BIDs analysis.

    Attributes
    ----------
    files : list of str
        TTTR file paths to process.
    time_window_ms : float
        Duration of each time window in milliseconds.
    output_dir : str, optional
        Directory for output ``.bst`` files. If omitted the API derives
        one from the first file path.
    """

    files: list[str]
    time_window_ms: float = 10.0
    output_dir: str | None = None


@dataclass
class TimeWindowResult:
    """Workflow output returned by time-window BIDs analysis.

    Attributes
    ----------
    files : list of str
        Files included in the analysis.
    n_windows : dict
        Per-file window count keyed by file path.
    output_paths : dict
        Paths written by the analysis, keyed by file path.
    metadata : dict
        Counts and additional run metadata.
    """

    files: list[str]
    n_windows: dict[str, int] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)
