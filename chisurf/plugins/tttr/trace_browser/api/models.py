"""Data models for the Trace Browser plugin."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceBrowserRequest:
    """Request parameters for listing and loading trace-browser data."""

    folder: str
    recursive: bool = False
    time_window_ms: float = 10.0
    setup_settings: dict[str, Any] | None = None
    selected_channels: list[int] | None = None
    paths: list[str] = field(default_factory=list)
    output_dir: str | None = None


@dataclass
class TraceFileSummary:
    """Summary of one TTTR file shown in the trace browser."""

    path: str
    name: str
    size: int
    rating: int = 0
    annotation: str = ""
    channels: list[int] = field(default_factory=list)


@dataclass
class TraceLoadResult:
    """JSON-safe trace data returned by the RPC layer."""

    path: str
    time_axis: list[float]
    counts: list[list[float]]
    labels: list[str]
    time_window_ms: float


def request_from_dict(data: dict[str, Any]) -> TraceBrowserRequest:
    """Build a request dataclass from a JSON payload."""
    return TraceBrowserRequest(
        folder=str(data.get("folder", "")),
        recursive=bool(data.get("recursive", False)),
        time_window_ms=float(data.get("time_window_ms", 10.0)),
        setup_settings=data.get("setup_settings"),
        selected_channels=data.get("selected_channels"),
        paths=[str(p) for p in data.get("paths", [])],
        output_dir=data.get("output_dir"),
    )


def request_to_dict(request: TraceBrowserRequest) -> dict[str, Any]:
    """Serialize a request dataclass to a JSON payload."""
    return asdict(request)


def summary_to_dict(summary: TraceFileSummary) -> dict[str, Any]:
    """Serialize a file summary to a JSON payload."""
    return asdict(summary)


def trace_result_to_dict(result: TraceLoadResult) -> dict[str, Any]:
    """Serialize trace data to a JSON payload."""
    return asdict(result)
