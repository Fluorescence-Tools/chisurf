"""Pure dataclasses for molecule-wise MLE analysis — no Qt, no tttrlib at import time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MoleculeMleSettings:
    """Parameters controlling the molecule-wise MLE analysis."""

    detector_chs: List[int] = field(default_factory=lambda: [2, 0])
    micro_time_range: Tuple[int, int] = (0, 256)
    micro_time_binning: int = 32
    normalize_counts: int = 0
    threshold: float = -1.0
    minlength: int = -1
    shift_sp: float = 0.0
    shift_ss: float = 0.0
    irf_threshold_fraction: float = 0.08
    fit_initial_values: Tuple[float, float, float, float] = (1.0, 0.0, 0.38, 1.0)
    fit_fixed_flags: Tuple[int, int, int, int] = (0, 0, 1, 0)
    l1: float = 0.04
    l2: float = 0.04
    twoi_star: bool = True
    bifl_scatter: bool = False
    seg_sigma: float = 1.0
    seg_threshold: float = -1.0
    peak_footprint_size: int = 6


@dataclass
class MoleculeMleRequest:
    """Encapsulates a full analysis request."""

    files: List[str]
    irf_file: str
    output_dir: str = ""
    settings: MoleculeMleSettings = field(default_factory=MoleculeMleSettings)


@dataclass
class MoleculeMleResult:
    """Result produced by a molecule-wise MLE analysis run."""

    processed_files: List[str]
    output_paths: List[str]
    joint_tsv: str = ""
    warnings: List[str] = field(default_factory=list)
