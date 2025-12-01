from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def make_rect_mask(
    shape: Sequence[int],
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
) -> np.ndarray:
    """Return a rectangular ROI mask with given x/y index ranges.

    Parameters
    ----------
    shape:
        Image shape ``(ny, nx)``.
    x_range, y_range:
        Start/stop indices in x and y (Python slicing semantics).
    """

    ny, nx = int(shape[0]), int(shape[1])
    x0, x1 = int(x_range[0]), int(x_range[1])
    y0, y1 = int(y_range[0]), int(y_range[1])
    m = np.zeros((ny, nx), dtype=bool)
    x0 = max(0, min(nx, x0))
    x1 = max(0, min(nx, x1))
    y0 = max(0, min(ny, y0))
    y1 = max(0, min(ny, y1))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


def make_intensity_threshold_mask(
    images: np.ndarray,
    low: Optional[float] = None,
    high: Optional[float] = None,
    use_mean_over_frames: bool = True,
) -> np.ndarray:
    """Create a mask based on intensity thresholds.

    Parameters
    ----------
    images:
        Image stack with shape ``(n_frames, ny, nx)`` or ``(ny, nx)``.
    low, high:
        Optional lower/upper thresholds. If omitted, they are inferred from
        the data percentiles (5 % / 95 %).
    use_mean_over_frames:
        If *True*, thresholds are applied to the mean image over frames.
    """

    arr = np.asarray(images, dtype=float)
    if arr.ndim == 3 and use_mean_over_frames:
        img = arr.mean(axis=0)
    elif arr.ndim == 2:
        img = arr
    else:
        raise ValueError(f"Unsupported image shape for threshold mask: {arr.shape}")

    finite = np.isfinite(img)
    data = img[finite]
    if data.size == 0:
        return np.zeros_like(img, dtype=bool)

    if low is None:
        low = float(np.percentile(data, 5.0))
    if high is None:
        high = float(np.percentile(data, 95.0))

    m = (img >= low) & (img <= high)
    m &= finite
    return m


def combine_masks(*masks: np.ndarray, mode: str = "and") -> np.ndarray:
    """Combine multiple boolean masks.

    Parameters
    ----------
    masks:
        Sequence of masks with identical shapes.
    mode:
        "and" (intersection) or "or" (union).
    """

    if not masks:
        raise ValueError("At least one mask must be provided")

    base = np.asarray(masks[0], dtype=bool).copy()
    for m in masks[1:]:
        m_arr = np.asarray(m, dtype=bool)
        if m_arr.shape != base.shape:
            raise ValueError("All masks must share the same shape")
        if mode == "and":
            base &= m_arr
        elif mode == "or":
            base |= m_arr
        else:
            raise ValueError(f"Unsupported combination mode: {mode}")
    return base
