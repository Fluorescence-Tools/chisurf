"""Pure dataclasses for pixel-wise MLE analysis — no Qt, no tttrlib at import time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PixelMleSettings:
    """Parameters controlling the pixel-wise MLE analysis."""

    # Detector / timing
    detector_chs_p: List[int] = field(default_factory=lambda: [0])
    detector_chs_s: List[int] = field(default_factory=lambda: [2])
    micro_time_start: int = 0
    micro_time_stop: int = 256
    micro_time_binning: int = 1

    # IRF preparation
    irf_threshold: float = 0.02
    irf_threshold_vv: Optional[float] = None
    irf_threshold_vh: Optional[float] = None
    shift_sp: float = 0.0
    shift_ss: float = 0.0
    irf_shift: int = 0

    # Fit initial / fixed
    tau: float = 1.0
    gamma: float = 0.0
    r0: float = 0.38
    rho: float = 1.0
    fix_tau: bool = False
    fix_gamma: bool = False
    fix_r0: bool = True
    fix_rho: bool = False

    # Fit flags
    twoi_star: bool = True
    bifl_scatter: bool = False

    # Background
    bg_p: float = 0.0
    bg_s: float = 0.0
    use_bg: bool = False
    min_photons: int = 10

    # Output
    output_format: str = "hdf"  # "hdf" or "csv"


@dataclass
class PixelMleRequest:
    """Encapsulates a full pixel-wise MLE analysis request."""

    files: List[str]
    irf_file: str
    output_dir: str = ""
    settings: PixelMleSettings = field(default_factory=PixelMleSettings)


@dataclass
class PixelMleResult:
    """Result produced by a pixel-wise MLE analysis run."""

    processed_files: List[str]
    output_paths: List[str]
    warnings: List[str] = field(default_factory=list)
