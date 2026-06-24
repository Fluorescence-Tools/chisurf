"""Data models for the Micro-time Shifter API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MFDBContext:
    """MFDB archival context for a micro-time shift request.

    Attributes
    ----------
    enabled : bool
        If ``False``, skip MFDB registration for this request.
    sample_id : str
        Optional existing MFDB sample identifier.
    source_artifact_ids : dict
        Mapping from input file path to an existing raw artifact ID.
    register_missing_inputs : bool
        If ``True``, archive input files that do not already have a
        source artifact ID.
    setup_id : str
        Optional MFDB setup identifier.
    setup_version : int, optional
        Optional setup version.

    """

    enabled: bool = True
    sample_id: str = ""
    source_artifact_ids: dict[str, str] = field(default_factory=dict)
    register_missing_inputs: bool = True
    setup_id: str = ""
    setup_version: int | None = None


@dataclass
class ShiftRequest:
    """Workflow input object for micro-time shifting.

    Attributes
    ----------
    files : list of str
        TTTR files to shift.
    global_shift : int
        Global micro-time shift applied to all routing channels.
    channel_shifts : dict
        Per-channel micro-time shifts keyed by routing channel number,
        e.g. ``{0: 1, 1: -2}``.
    filetype : str, optional
        Explicit tttrlib file type. ``None`` lets tttrlib infer.
    output_dir : str, optional
        Directory for shifted output files.
    mfdb : MFDBContext
        Optional MFDB archival context.

    """

    files: list[str]
    global_shift: int = 0
    channel_shifts: dict[int, int] = field(default_factory=dict)
    filetype: str | None = None
    output_dir: str | None = None
    mfdb: MFDBContext = field(default_factory=MFDBContext)


@dataclass
class ShiftResult:
    """Workflow output object produced by micro-time shifting.

    Attributes
    ----------
    output_paths_by_file : dict
        Mapping from input file path to shifted output file path.
    applied_shifts_by_file : dict
        Mapping from input file path to dict with ``global_shift`` and
        ``channel_shifts`` keys showing the exact shifts applied.
    mfdb_artifacts : dict
        MFDB artifact IDs returned by the archival layer.
    warnings : list
        Non-fatal warnings from the shift or MFDB registration.

    """

    output_paths_by_file: dict[str, str] = field(default_factory=dict)
    applied_shifts_by_file: dict[str, dict[str, Any]] = field(default_factory=dict)
    mfdb_artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)
