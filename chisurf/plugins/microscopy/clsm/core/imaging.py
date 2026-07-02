"""CLSM image construction, representations and decay extraction (Qt-free).

``tttrlib`` is imported lazily inside the functions so this module stays
importable (and unit-testable) even where the compiled extension is absent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .frc import counting_noise, gaussian_kernel

#: Supported image-representation names.
IMAGE_TYPES = ("Intensity", "Mean micro time", "Intensity, Mean micro time")

#: Frame-reduction modes used by :func:`reduce_frames`.
FRAME_MODES = ("frame", "sum", "mean")


def brush_kernel(
    size: int = 7,
    width: float = 3.0,
    select: bool = True,
    peak: float = 30000.0,
) -> np.ndarray:
    """Build the paint kernel used to brush a pixel selection.

    Parameters
    ----------
    size : int
        Side length of the (square) Gaussian kernel.
    width : float
        Gaussian half-width in standard deviations.
    select : bool
        ``True`` paints (positive kernel), ``False`` erases (negated kernel).
    peak : float
        Value assigned to the kernel core so a single stroke saturates the
        selection mask.

    Returns
    -------
    numpy.ndarray
        The draw kernel.
    """
    kernel = gaussian_kernel(size, width)
    kernel[kernel > 0.001] = peak
    if not select:
        kernel = kernel * -1.0
    return kernel


def build_clsm_image(tttr: Any, setup: Any) -> Any:
    """Construct and fill a ``tttrlib.CLSMImage`` from a TTTR object.

    Parameters
    ----------
    tttr : tttrlib.TTTR
        The loaded TTTR dataset.
    setup : ClsmSetup-like
        Object/dataclass exposing ``frame_marker`` (list[int]),
        ``line_start_marker``, ``line_stop_marker``, ``event_type_marker``,
        ``pixel_per_line``, ``routine`` and ``channels``.

    Returns
    -------
    tttrlib.CLSMImage
        The filled CLSM image.
    """
    import tttrlib

    clsm_image = tttrlib.CLSMImage(
        tttr,
        list(setup.frame_marker),
        int(setup.line_start_marker),
        int(setup.line_stop_marker),
        int(setup.event_type_marker),
        int(setup.pixel_per_line),
        str(setup.routine),
    )
    clsm_image.fill(tttr_data=tttr, channels=list(setup.channels))
    return clsm_image


def representation(
    clsm_image: Any,
    tttr: Any,
    image_type: str = "Intensity",
    n_ph_min: int = 1,
) -> np.ndarray:
    """Compute an image representation as a ``(frames, lines, pixel)`` array.

    Parameters
    ----------
    clsm_image : tttrlib.CLSMImage
        A filled CLSM image.
    tttr : tttrlib.TTTR
        The TTTR dataset the image was built from (needed for micro-time stats).
    image_type : str
        One of :data:`IMAGE_TYPES`.
    n_ph_min : int
        Minimum photons per pixel for the mean-micro-time calculation.

    Returns
    -------
    numpy.ndarray
        3-D ``float64`` image stack.
    """
    if image_type == "Mean micro time":
        data = clsm_image.get_mean_micro_time(tttr, n_ph_min, False)
        return data.astype(np.float64)
    if image_type == "Intensity":
        return clsm_image.intensity.astype(np.float64)
    # default: intensity-weighted mean micro time
    mean_micro_time = clsm_image.get_mean_micro_time(tttr, n_ph_min, False)
    return mean_micro_time.astype(np.float64) * clsm_image.intensity.astype(np.float64)


def reduce_frames(
    image: np.ndarray,
    mode: str = "sum",
    frame_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a 3-D image stack to a display frame plus two FRC subsets.

    Parameters
    ----------
    image : numpy.ndarray
        3-D ``(frames, lines, pixel)`` stack.
    mode : str
        ``"sum"`` / ``"mean"`` collapse all frames (the FRC subsets are the
        even/odd frames); ``"frame"`` selects a single frame (the FRC subset is
        that frame against its neighbour).
    frame_idx : int
        Frame index used when *mode* is ``"frame"``.

    Returns
    -------
    current, subset_1, subset_2 : numpy.ndarray
        The 2-D image to display and the two 2-D subsets used for FRC.
    """
    if mode == "sum":
        return image.sum(axis=0), image[::2].sum(axis=0), image[1::2].sum(axis=0)
    if mode == "mean":
        return image.mean(axis=0), image[::2].mean(axis=0), image[1::2].mean(axis=0)
    # single frame
    ref_idx = max(0, frame_idx - 1)
    if ref_idx == 0 and frame_idx == 0:
        ref_idx = min(1, image.shape[0] - 1)
    return image[frame_idx], image[frame_idx], image[ref_idx]


def decay_of_selection(
    clsm_image: Any,
    tttr: Any,
    mask: np.ndarray,
    tac_coarsening: int = 1,
    stack_frames: bool = True,
    frame_idx: int = 0,
    trim_trailing_zeros: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a fluorescence decay histogram from a 2-D pixel selection.

    Parameters
    ----------
    clsm_image : tttrlib.CLSMImage
        The filled CLSM image.
    tttr : tttrlib.TTTR
        The TTTR dataset (for the micro-time resolution).
    mask : numpy.ndarray
        2-D selection mask (non-zero pixels are included); broadcast across all
        frames internally.
    tac_coarsening : int
        Micro-time (TAC) binning factor.
    stack_frames : bool
        If ``True`` sum the decay over all frames, otherwise return the decay of
        *frame_idx*.
    frame_idx : int
        Frame to return when *stack_frames* is ``False``.
    trim_trailing_zeros : bool
        Drop the empty tail of the histogram.

    Returns
    -------
    t : numpy.ndarray
        Time axis in nanoseconds.
    y : numpy.ndarray
        Photon counts per micro-time bin.
    ey : numpy.ndarray
        Poisson weights (see :func:`counting_noise`).
    """
    sel = np.copy(mask)
    sel[sel > 0] = 1
    sel[sel < 0] = 0
    sel = sel.astype(np.uint8)
    selection = np.ascontiguousarray(
        np.broadcast_to(
            sel,
            (clsm_image.n_frames, clsm_image.n_lines, clsm_image.n_pixel),
        )
    )
    decay = clsm_image.get_decay_of_pixels(
        tttr_data=tttr,
        mask=selection,
        tac_coarsening=tac_coarsening,
        stack_frames=stack_frames,
    )
    header = tttr.get_header()
    x = np.arange(decay.shape[1])
    t = x * header.micro_time_resolution * tac_coarsening * 1e9  # ns

    if stack_frames:
        y = decay.sum(axis=0)
    else:
        y = decay[frame_idx]
    y = y.astype(np.float64)

    if trim_trailing_zeros:
        y_pos = np.where(y > 0)[0]
        if len(y_pos) > 0:
            i_y_max = y_pos[-1]
            y = y[:i_y_max]
            t = t[:i_y_max]
    return t, y, counting_noise(y)
