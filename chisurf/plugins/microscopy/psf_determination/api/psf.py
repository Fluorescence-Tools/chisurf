"""Pure PSF computation functions extracted from PSFDeterminationWidget.

All functions are free of Qt dependencies and can be called from headless
contexts (CLI, tests, RPC services).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

if TYPE_CHECKING:
    from .models import PsfFitResult, PsfSettings


def gaussian_3d(coords: np.ndarray, params: np.ndarray | list) -> np.ndarray:
    """Evaluate a 3-D Gaussian at *coords*.

    Parameters
    ----------
    coords:
        Array of shape ``(N, 3)`` with columns ``[z, y, x]``.
    params:
        Sequence ``[z_c, y_c, x_c, sigma_z, sigma_y, sigma_x, amplitude, offset]``.

    Returns
    -------
    numpy.ndarray
        Shape ``(N,)`` — Gaussian values at each coordinate.
    """
    z_c, y_c, x_c, sigma_z, sigma_y, sigma_x, amplitude, offset = params
    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
    exponent = -0.5 * (
        ((z - z_c) / sigma_z) ** 2
        + ((y - y_c) / sigma_y) ** 2
        + ((x - x_c) / sigma_x) ** 2
    )
    return amplitude * np.exp(exponent) + offset


def extract_roi(
    stack: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
    roi_xy: int,
    roi_z: int,
) -> tuple[np.ndarray | None, tuple[int, ...] | None]:
    """Extract a 3-D ROI centred at ``(z0, y0, x0)``.

    Parameters
    ----------
    stack:
        3-D image stack with axes ``(z, y, x)``.
    z0, y0, x0:
        Centre position in stack coordinates.
    roi_xy:
        Half-size of the ROI in x and y (total = ``2 * roi_xy + 1``).
    roi_z:
        Half-size of the ROI in z (total = ``2 * roi_z + 1``).

    Returns
    -------
    roi:
        Extracted 3-D sub-array, or ``None`` if out of bounds.
    bounds:
        ``(z_min, z_max, y_min, y_max, x_min, x_max)`` in stack coordinates,
        or ``None`` if out of bounds.
    """
    nz, ny, nx = stack.shape
    half_xy = roi_xy // 2
    half_z = roi_z // 2

    z_min, z_max = z0 - half_z, z0 + half_z + 1
    y_min, y_max = y0 - half_xy, y0 + half_xy + 1
    x_min, x_max = x0 - half_xy, x0 + half_xy + 1

    if z_min < 0 or z_max > nz or y_min < 0 or y_max > ny or x_min < 0 or x_max > nx:
        return None, None

    roi = stack[z_min:z_max, y_min:y_max, x_min:x_max].copy()
    return roi, (z_min, z_max, y_min, y_max, x_min, x_max)


def fit_3d_gaussian(roi: np.ndarray) -> dict:
    """Fit a 3-D Gaussian to *roi*.

    Parameters
    ----------
    roi:
        3-D array ``(nz, ny, nx)``.

    Returns
    -------
    dict
        Keys: ``params``, ``success``, ``cost``.
        ``params`` is ``[z_c, y_c, x_c, sigma_z, sigma_y, sigma_x, amplitude, offset]``.
    """
    nz, ny, nx = roi.shape
    offset_init = float(np.min(roi))
    amplitude_init = float(np.max(roi)) - offset_init

    idx_max = np.unravel_index(np.argmax(roi), roi.shape)
    z_c_init, y_c_init, x_c_init = idx_max

    p0 = [
        z_c_init, y_c_init, x_c_init,
        nz / 5.0, ny / 5.0, nx / 5.0,
        amplitude_init, offset_init,
    ]

    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij"
    )
    coords = np.stack([z_grid.ravel(), y_grid.ravel(), x_grid.ravel()], axis=1)
    data_flat = roi.ravel()

    def residuals(params: np.ndarray) -> np.ndarray:
        return gaussian_3d(coords, params) - data_flat

    result = least_squares(
        residuals,
        p0,
        bounds=(
            [0, 0, 0, 0.5, 0.5, 0.5, 0, -np.inf],
            [nz, ny, nx, nz, ny, nx, np.inf, np.inf],
        ),
        max_nfev=500,
    )

    return {"params": result.x, "success": bool(result.success), "cost": float(result.cost)}


def detect_beads(
    stack: np.ndarray,
    roi_xy: int = 15,
    roi_z: int = 15,
    pixels_per_frame: int = 20,
    min_distance: float = 5.0,
) -> list[tuple[int, int, int]]:
    """Detect candidate bead positions in a 3-D image stack.

    Uses an adaptive quantile threshold per z-slice and filters by minimum
    lateral distance between candidates.

    Parameters
    ----------
    stack:
        3-D array ``(nz, ny, nx)``.
    roi_xy:
        Half-size of the fitting ROI in x/y (used for boundary exclusion).
    roi_z:
        Half-size of the fitting ROI in z (used for boundary exclusion and step).
    pixels_per_frame:
        Expected number of bright pixels per frame (used to derive quantile).
    min_distance:
        Minimum lateral distance (pixels) between accepted candidates.

    Returns
    -------
    list of (z, y, x) tuples
        Detected bead positions.
    """
    nz, ny, nx = stack.shape
    half_xy = roi_xy // 2
    half_z = roi_z // 2
    total_px = float(ny * nx)
    q_level = max(0.0, min(1.0, 1.0 - pixels_per_frame / total_px)) if total_px > 0 else 0.99
    step_z = max(4, roi_z // 4)

    beads: list[tuple[int, int, int]] = []
    z_start = half_z + 2
    z_end = nz - half_z - 2

    for z in range(z_start, max(z_start, z_end), step_z):
        frame = stack[z]
        if not np.any(np.isfinite(frame)):
            continue

        q = np.quantile(frame, q_level)
        mask = frame >= q
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue

        intensities = frame[ys, xs]
        order = np.argsort(intensities)[::-1]
        ys, xs = ys[order], xs[order]

        accepted: list[tuple[int, int]] = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            if accepted:
                dy = np.array([y - ay for ay, _ in accepted], dtype=float)
                dx = np.array([x - ax for _, ax in accepted], dtype=float)
                if not np.all(dx * dx + dy * dy >= min_distance * min_distance):
                    continue
            accepted.append((y, x))

        for y, x in accepted:
            if x < half_xy or x >= nx - half_xy:
                continue
            if y < half_xy or y >= ny - half_xy:
                continue
            if z < half_z or z >= nz - half_z:
                continue
            beads.append((z, int(y), int(x)))

    return beads


def fit_all_beads(
    stack: np.ndarray,
    detected_beads: list[tuple[int, int, int]],
    roi_xy: int,
    roi_z: int,
    pixel_nm: float,
    z_step_nm: float,
) -> list[dict]:
    """Fit 3-D Gaussian PSF to every detected bead.

    Parameters
    ----------
    stack:
        3-D image stack ``(nz, ny, nx)``.
    detected_beads:
        List of ``(z, y, x)`` positions from :func:`detect_beads`.
    roi_xy, roi_z:
        ROI half-sizes in xy and z.
    pixel_nm:
        Lateral pixel size in nm.
    z_step_nm:
        Axial step size in nm.

    Returns
    -------
    list of dict
        One dict per bead with keys matching :class:`~.models.PsfFitResult`.
    """
    from .models import PsfFitResult

    results = []
    for idx, (z0, y0, x0) in enumerate(detected_beads):
        roi, _ = extract_roi(stack, z0, y0, x0, roi_xy, roi_z)
        if roi is None:
            results.append({
                "index": idx, "x_px": x0, "y_px": y0, "z_slice": z0,
                "sigma_x_px": float("nan"), "sigma_y_px": float("nan"),
                "sigma_z_px": float("nan"), "fwhm_x_nm": float("nan"),
                "fwhm_y_nm": float("nan"), "fwhm_z_nm": float("nan"),
                "fwhm_xy_nm": float("nan"), "sigma_xy_nm": float("nan"),
                "sigma_z_nm": float("nan"), "axial_ratio": float("nan"),
                "success": False, "cost": float("nan"),
                "error": "ROI out of bounds",
            })
            continue

        try:
            fit = fit_3d_gaussian(roi)
        except Exception as exc:
            results.append({
                "index": idx, "x_px": x0, "y_px": y0, "z_slice": z0,
                "sigma_x_px": float("nan"), "sigma_y_px": float("nan"),
                "sigma_z_px": float("nan"), "fwhm_x_nm": float("nan"),
                "fwhm_y_nm": float("nan"), "fwhm_z_nm": float("nan"),
                "fwhm_xy_nm": float("nan"), "sigma_xy_nm": float("nan"),
                "sigma_z_nm": float("nan"), "axial_ratio": float("nan"),
                "success": False, "cost": float("nan"),
                "error": str(exc),
            })
            continue

        params = fit["params"]
        sigma_z, sigma_y, sigma_x = params[3], params[4], params[5]
        sigma_xy = (sigma_x + sigma_y) / 2.0
        fwhm_x_nm = 2.355 * sigma_x * pixel_nm
        fwhm_y_nm = 2.355 * sigma_y * pixel_nm
        fwhm_z_nm = 2.355 * sigma_z * z_step_nm
        fwhm_xy_nm = (fwhm_x_nm + fwhm_y_nm) / 2.0
        sigma_xy_nm = sigma_xy * pixel_nm
        sigma_z_nm = sigma_z * z_step_nm
        axial_ratio = sigma_z_nm / sigma_xy_nm if sigma_xy_nm > 0 else math.nan

        results.append({
            "index": idx, "x_px": x0, "y_px": y0, "z_slice": z0,
            "sigma_x_px": float(sigma_x), "sigma_y_px": float(sigma_y),
            "sigma_z_px": float(sigma_z),
            "fwhm_x_nm": fwhm_x_nm, "fwhm_y_nm": fwhm_y_nm,
            "fwhm_z_nm": fwhm_z_nm, "fwhm_xy_nm": fwhm_xy_nm,
            "sigma_xy_nm": sigma_xy_nm, "sigma_z_nm": sigma_z_nm,
            "axial_ratio": axial_ratio,
            "success": fit["success"], "cost": fit["cost"],
            "error": None,
        })

    return results
