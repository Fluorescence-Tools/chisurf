from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # tttrlib is expected to be available in the chisurf dev environment
    import tttrlib  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    tttrlib = None  # type: ignore[var-annotated]

from .data import RicsData, RicsSettings


def _ensure_3d_stack(images: np.ndarray) -> np.ndarray:
    """Return images as (n_frames, ny, nx) stack.

    Accepts common shapes and normalizes them for ICS computation.
    """

    arr = np.asarray(images, dtype=float)
    if arr.ndim == 2:
        # Single frame
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        # Heuristic for multi-channel CLSMImage.intensity:
        # (n_channels, frames_per_channel, n_lines, n_pixel) -> sum over channels
        return arr.sum(axis=0)
    raise ValueError(f"Unsupported image array shape for RICS: {arr.shape}")


def compute_rics_from_images(
    images: np.ndarray,
    settings: Optional[RicsSettings] = None,
    mask: Optional[np.ndarray] = None,
    use_fftshift: bool = True,
    **kwargs: Any,
) -> RicsData:
    """Compute 2D RICS/ICS from an image stack using tttrlib.

    Parameters
    ----------
    images:
        Image stack with shape ``(n_frames, ny, nx)`` or compatible.
    settings:
        Optional :class:`RicsSettings` with ROI and ICS options.
    mask:
        Optional binary mask ``(ny, nx)``. When provided, pixels outside the
        mask are set to zero prior to ICS computation (simple masked RICS).
    use_fftshift:
        If *True*, apply ``np.fft.fftshift`` to the mean ICS and its standard
        deviation so that the zero-lag correlation appears in the center
        of the image (as in the tttrlib RICS examples).

    Returns
    -------
    RicsData
        Container with ICS stack, mean/std images, and lag index arrays.
    """

    if tttrlib is None:
        raise RuntimeError("tttrlib is not available; cannot compute RICS")

    if settings is None:
        settings = RicsSettings()

    stack = _ensure_3d_stack(images)
    n_frames, ny, nx = stack.shape

    # Apply simple binary mask if provided (Approach A from the plan).
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != (ny, nx):
            raise ValueError(
                f"Mask shape {m.shape} does not match image shape {(ny, nx)}"
            )
        stack = stack * m[None, ...]

    # Prepare default ROI if none specified: full image as in tttrlib examples.
    if settings.x_range is None:
        x_range: Sequence[int] = (0, -1)
    else:
        x_range = settings.x_range

    if settings.y_range is None:
        y_range: Sequence[int] = (0, -1)
    else:
        y_range = settings.y_range

    ics_kwargs: Dict[str, Any] = {
        "images": stack,
        "x_range": list(x_range),
        "y_range": list(y_range),
        "subtract_average": settings.subtract_average,
    }

    # Allow caller to override / extend low-level ICS arguments.
    ics_kwargs.update(kwargs)

    # Optional explicit frame pairs (ACF/CCF) as in tttrlib examples.
    if settings.frames_index_pairs is not None:
        ics_kwargs["frames_index_pairs"] = list(settings.frames_index_pairs)

    ics = tttrlib.CLSMImage.compute_ics(**ics_kwargs)  # type: ignore[call-arg]
    ics_stack = np.asarray(ics, dtype=float)

    if ics_stack.ndim == 2:
        ics_stack = ics_stack[None, ...]

    n_ics, ny_ics, nx_ics = ics_stack.shape

    # Mean and standard error of the mean, as in plot_imaging_ics_fit.py
    ics_mean_raw = ics_stack.mean(axis=0)
    ics_std_raw = ics_stack.std(axis=0) / max(1.0, np.sqrt(float(n_ics)))

    if use_fftshift:
        ics_mean = np.fft.fftshift(ics_mean_raw)
        ics_std = np.fft.fftshift(ics_std_raw)
    else:
        ics_mean = ics_mean_raw
        ics_std = ics_std_raw

    ny_out, nx_out = ics_mean.shape
    line_shift, pixel_shift = np.indices((ny_out, nx_out))
    line_shift = line_shift - ny_out // 2
    pixel_shift = pixel_shift - nx_out // 2

    meta: Dict[str, Any] = {
        "n_input_frames": int(n_frames),
        "n_ics_frames": int(n_ics),
        "fftshifted": bool(use_fftshift),
        "settings": {
            "x_range": tuple(x_range),
            "y_range": tuple(y_range),
            "subtract_average": settings.subtract_average,
            "frames_index_pairs": (
                list(settings.frames_index_pairs)
                if settings.frames_index_pairs is not None
                else None
            ),
            "pixel_duration_us": settings.pixel_duration_us,
            "line_duration_ms": settings.line_duration_ms,
            "pixel_size_nm": settings.pixel_size_nm,
        },
    }

    return RicsData(
        ics_stack=ics_stack,
        ics_mean=ics_mean,
        ics_std=ics_std,
        line_shift=line_shift,
        pixel_shift=pixel_shift,
        mask=np.asarray(mask, dtype=bool) if mask is not None else None,
        meta=meta,
    )
