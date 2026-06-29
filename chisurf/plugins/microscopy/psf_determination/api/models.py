"""Pure dataclasses for PSF determination — no Qt at import time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PsfSettings:
    """Parameters for 3D PSF fitting."""

    pixel_size_nm: float = 100.0
    z_step_nm: float = 200.0
    roi_xy: int = 15
    roi_z: int = 15
    pixels_per_frame: int = 20
    min_distance: float = 5.0


@dataclass
class PsfFitResult:
    """Result of a 3D Gaussian PSF fit for one bead."""

    index: int
    x_px: int
    y_px: int
    z_slice: int
    sigma_x_px: float
    sigma_y_px: float
    sigma_z_px: float
    fwhm_x_nm: float
    fwhm_y_nm: float
    fwhm_z_nm: float
    fwhm_xy_nm: float
    sigma_xy_nm: float
    sigma_z_nm: float
    axial_ratio: float
    success: bool
    cost: float = 0.0
    error: Optional[str] = None
