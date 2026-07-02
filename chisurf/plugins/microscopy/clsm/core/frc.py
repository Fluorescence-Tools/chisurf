"""Fourier-Ring-Correlation and TCSPC noise helpers (Qt-free)."""

from __future__ import annotations

import numba as nb
import numpy as np
import scipy.stats as st


def counting_noise(
    decay: np.ndarray,
    treat_zeros: bool = True,
    zero_value: float = 1.0,
) -> np.ndarray:
    """Poisson counting noise (``sqrt`` of counts) for a TCSPC decay.

    Parameters
    ----------
    decay : numpy.ndarray
        The photon counts.
    treat_zeros : bool
        If ``True`` (default) bins that are ``<= 0`` are replaced by
        *zero_value* before taking the square root, so the returned weights
        never contain zeros.
    zero_value : float
        Value assigned to non-positive bins when *treat_zeros* is ``True``.

    Returns
    -------
    numpy.ndarray
        Per-bin noise estimate usable as data-analysis weights.
    """
    w = np.array(decay, dtype=np.float64)
    if treat_zeros:
        w[w <= 0.0] = zero_value
    return np.sqrt(w)


def gaussian_kernel(kernel_size: int = 21, nsig: float = 3.0) -> np.ndarray:
    """Return a normalised 2-D Gaussian kernel.

    Parameters
    ----------
    kernel_size : int
        Side length (in pixels) of the square kernel.
    nsig : float
        Half-width of the Gaussian in standard deviations across the kernel.

    Returns
    -------
    numpy.ndarray
        ``(kernel_size, kernel_size)`` kernel summing to one.
    """
    interval = (2.0 * nsig + 1.0) / kernel_size
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernel_size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    return kernel_raw / kernel_raw.sum()


@nb.jit(nopython=True)
def _frc_histogram(lx, rx, ly, ry, f1f2, f12, f22, n_bins, bin_width):
    """Accumulate the radial FRC shells (auxiliary for :func:`compute_frc`)."""
    wf1f2 = np.zeros(n_bins, np.float64)
    wf1 = np.zeros(n_bins, np.float64)
    wf2 = np.zeros(n_bins, np.float64)
    for xi in range(lx, rx):
        for yi in range(ly, ry):
            distance_bin = int(np.sqrt(xi**2 + yi**2) / bin_width)
            if distance_bin < n_bins:
                wf1f2[distance_bin] += f1f2[xi, yi]
                wf1[distance_bin] += f12[xi, yi]
                wf2[distance_bin] += f22[xi, yi]
    return wf1f2 / np.sqrt(wf1 * wf2)


def compute_frc(
    image_1: np.ndarray,
    image_2: np.ndarray,
    bin_width: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fourier Ring Correlation (FRC) between two images.

    The FRC is the normalised cross-correlation coefficient between the two
    images over corresponding shells in Fourier space and is commonly used to
    estimate the effective resolution of an image.

    Parameters
    ----------
    image_1, image_2 : numpy.ndarray
        The two 2-D images (typically the even/odd frame subsets).
    bin_width : float
        Ring width (in Fourier pixels) used when building the FRC histogram.

    Returns
    -------
    density : numpy.ndarray
        FRC value per ring.
    bins : numpy.ndarray
        Ring boundaries (spatial-frequency axis).
    """
    f1 = np.fft.fft2(image_1)
    f2 = np.fft.fft2(image_2)
    f1f2 = np.real(f1 * np.conjugate(f2))
    f12, f22 = np.abs(f1) ** 2, np.abs(f2) ** 2
    nx, ny = image_1.shape

    bins = np.arange(0, np.sqrt((nx // 2) ** 2 + (ny // 2) ** 2), bin_width)
    n_bins = int(bins.shape[0])
    lx, rx = int(-(nx // 2)), int(nx // 2)
    ly, ry = int(-(ny // 2)), int(ny // 2)
    density = _frc_histogram(lx, rx, ly, ry, f1f2, f12, f22, n_bins, bin_width)
    return density, bins
