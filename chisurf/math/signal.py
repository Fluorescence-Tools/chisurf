from __future__ import annotations
from chisurf import typing

import numpy as np
import numba as nb
import scipy.stats as st


window_function_types = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']


def window(
        data: np.array,
        window_len: int,
        window_function_type: str = 'bartlett'
) -> np.array:
    """
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    :param data: 1D numpy-array (data)
    :param window_len: the dimension of the smoothing window; should be an odd
    integer
    :param window_function_type: the type of window from 'flat', 'hanning',
    'hamming', 'bartlett', 'blackman' flat window will produce a moving average
    smoothing.
    :return: 1D numpy-array (smoothed data)

    Examples
    --------


    """
    if data.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if data.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return data
    if not window_function_type in [
        'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    ]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )
    s = np.r_[
        2 * data[0] - data[window_len:1:-1], data, 2 * data[-1] - data[-1:-window_len:-1]
    ]
    if window_function_type == 'flat': # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window_function_type + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


def shift_array(
        v: np.ndarray,
        shift: float,
        set_outside: bool = True,
        outside_value: float = 0.0
) -> np.array:
    """Calculates an array that is shifted by a float. For non-integer shifts
    the shifted array is interpolated.

    Parameters
    ----------
    v : 1D numpy-array
        The input numpy array that is shifted
    shift : float
        A floating point number by which the array is shifted
    set_outside : bool
        If True (default) the values outside of the array are set
        to the value defined by the parameter `outside_value`
    outside_value : float
        The value assigned to the vector that are outside. The
        values on the borders are assigned to this value (default
        set to zero).

    Returns
    -------
    numpy-array
        The shifted numpy array

    """
    ts = shift
    ts_i = int(ts)
    ts_f = ts - np.floor(ts)
    ysh = np.roll(v, ts_i) * (1.0 - ts_f) + np.roll(v, ts_i + 1) * ts_f
    if set_outside:
        if ts >= 0:
            b = int(np.ceil(ts))
            ysh[:b] = outside_value
        elif ts < 0:
            b = int(np.floor(ts))
            ysh[b:] = outside_value
    return ysh


def autocorr(
        x: np.ndarray,
        axis: int = 0,
        normalize: bool = True
) -> np.array:
    """
    Estimate the autocorrelation function of a time series using the FFT.

    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.

    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.

    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    """
    # if fast:
    #     # For computational efficiency, crop the chain to the largest power of two.
    #     x = np.atleast_1d(x)
    #     m = [slice(None), ] * len(x.shape)
    #     n = int(2**np.floor(np.log2(x.shape[axis])))
    #     m[axis] = slice(0, n)
    #     # Compute the FFT and then (from that) the auto-correlation function.
    #     f = np.fft.fft(x-np.mean(x, axis=axis), n=2*n, axis=axis)
    #     m[axis] = slice(0, n)
    #     acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
    #     m[axis] = 0
    #     if normalize:
    #         return acf / acf[m]
    #     else:
    #         return acf
    # else:
    if len(x) > 0:
        return xcorr_fft(
            in_1=x,
            in_2=x,
            axis=axis,
            normalize=normalize
        )
    else:
        return np.array([], dtype=x.dtype)


def xcorr_fft(
        in_1: np.ndarray,
        in_2: np.ndarray,
        axis: int = 0,
        normalize: bool = True
) -> np.ndarray:
    """Computes the cross-correlation function of two arrays using fast fourier transforms.
    """
    if len(in_1) > 0 and len(in_2) > 0:
        c = in_1
        d = in_2
        n = c.shape[axis]
        m = [slice(None), ] * len(c.shape)

        # Compute the FFT and then (from that) the auto-correlation function.
        f1 = np.fft.fft(c-np.mean(c, axis=axis), n=2*n, axis=axis)
        f2 = np.fft.fft(d-np.mean(d, axis=axis), n=2*n, axis=axis)

        m[axis] = slice(0, n)
        acf = np.fft.ifft(f1 * np.conjugate(f2), axis=axis)[m[axis]].real
        m[axis] = 0
        if normalize:
            return acf / acf[m[axis]]
        else:
            return acf
    else:
        return np.array([], dtype=in_1.dtype)


def calculate_fwhm(
        x_values: np.ndarray,
        y_values: np.ndarray,
        background: float = 0.0,
        verbose: bool = False
) -> typing.Tuple[
    float, typing.Tuple[int, int],
    typing.Tuple[float, float]
]:
    """Calculates the full-width-half-maximum (FWHM) using a linear-search from
    both sides of the curve

    :param curve:
    :param background:
    :param verbose:
    :return: Tuple containing the FWHM, the indices and the x-values of the
    used positions
    """
    y_values_bg = y_values - background
    x_values = x_values

    half_maximum = max(y_values_bg) / 2.0
    smaller = np.where(y_values_bg > half_maximum)[0]
    lb_i = smaller[0]
    ub_i = smaller[-1]

    x_left = x_values[lb_i]
    x_right = x_values[ub_i]
    fwhm = x_right - x_left

    if verbose:
        print("FWHM:")
        print("lb, ub    : (%s, %s)" % (x_left, x_right))
        print("fwhm: %s" % fwhm)
    return fwhm, (lb_i, ub_i), (x_left, x_right)


def gaussian_kernel(
        kernel_size: int = 21,
        nsig: float = 3
):
    """Returns a 2D Gaussian kernel array.

    :param kernel_size: the size of the 2D array
    :param nsig: the width of the gaussian in the 2D array.
    :return:
    """
    interval = (2.0 * nsig + 1.) / kernel_size
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernel_size+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


@nb.jit(nopython=True)
def _frc_histogram(lx, rx, ly, ry, f1f2, f12, f22, n_bins, bin_width):
    """Auxiliary function only intented to be used by compute_frc

    Parameters
    ----------
    lx : int
        left boundary of the x axis. For an image with 512 pixel this
        would be -256
    rx : int
        right boundary of the x axis. For 512 pixel 255
    ly : int
        left boundary of the x axis. For an image with 512 pixel this
        would be -256
    ry : int
        right boundary of the x axis. For 512 pixel 255
    f1f2 : numpy.array
        The product of the Fourier transformed images F(1)F(2)',
        where F(2)' is the complex conjugate of F2
    f12 : numpy.array
        The squared absolute value of the F(1) Fourier transform, abs(F(1))**2
    f22 : numpy.array
        The squared absolute value of the F(2) Fourier transform, abs(F(2))**2
    n_bins : int
        The number of bins in the FRC
    bin_width : float
        The width of the Rings in the FRC

    Returns
    -------
    numpy-array:
        The FRC value

    """
    wf1f2 = np.zeros(n_bins, np.float64)
    wf1 = np.zeros(n_bins, np.float64)
    wf2 = np.zeros(n_bins, np.float64)
    for xi in range(lx, rx):
        for yi in range(ly, ry):
            distance_bin = int(np.sqrt(xi ** 2 + yi ** 2) / bin_width)
            if distance_bin < n_bins:
                wf1f2[distance_bin] += f1f2[xi, yi]
                wf1[distance_bin] += f12[xi, yi]
                wf2[distance_bin] += f22[xi, yi]
    return wf1f2 / np.sqrt(wf1 * wf2)


def compute_frc(
        image_1: np.ndarray,
        image_2: np.ndarray,
        bin_width: int = 2.0
):
    """Computes the Fourier Ring Correlation (FRC) between two images.

    The FRC measures the correlation between two images by the overlap of the
    Fourier transforms. The FRC is the normalised cross-correlation coefficient
    between two images over corresponding shells in Fourier space transform.

    Parameters
    ----------
    image_1 : numpy.array
        The first image
    image_2 : numpy.array
        The second image
    bin_width : float
        The bin width used in the computation of the FRC histogram

    Returns
    -------
    Numpy array:
        density of the FRC histogram
    Numpy array:
        bins of the FRC histogram

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
    density = _frc_histogram(
        lx, rx,
        ly, ry,
        f1f2, f12, f22,
        n_bins, bin_width
    )
    return density, bins


def find_bursts(arr, max_gap=0):
    """
    Identifies sequences (bursts) of consecutive ones in a binary array and optionally merges small gaps between them.

    The function detects where the array changes from 0 to 1 (indicating the start of a burst) and 1 to 0
    (indicating the end of a burst). If `max_gap` is greater than 0, gaps smaller than or equal to `max_gap`
    between consecutive bursts are merged.

    Parameters:
    arr (numpy array): A binary array (containing 0s and 1s) where the function will identify bursts of ones.
    max_gap (int, optional): The maximum gap size between bursts that can be merged. Default is 0 (no merging).

    Returns:
    numpy array: A 2D array where each row contains the start and end indices (inclusive) of each burst.

    Example:
    --------
    >>> arr = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0])
    >>> find_bursts(arr)
    array([[1, 2],
           [5, 7]])

    >>> find_bursts(arr, max_gap=1)
    array([[1, 7]])
    """
    # Find where the array changes from 0 to 1 (start of burst) and 1 to 0 (end of burst)
    is_burst = np.diff(arr, prepend=0, append=0)
    starts = np.where(is_burst == 1)[0]
    stops = np.where(is_burst == -1)[0]

    # If max_gap is greater than 0, merge small gaps
    if max_gap > 0:
        merged_starts = [starts[0]]  # Initialize with the first start
        merged_stops = []

        for i in range(1, len(starts)):
            # Check if the gap between current stop and next start is small enough to merge
            if starts[i] - stops[i - 1] - 1 <= max_gap:
                continue  # Skip this start, effectively merging
            else:
                merged_stops.append(stops[i - 1])
                merged_starts.append(starts[i])

        # Append the final stop
        merged_stops.append(stops[-1])

        # Convert merged lists to NumPy arrays
        starts = np.array(merged_starts)
        stops = np.array(merged_stops)

    # Stack the starts and stops into a 2D array
    bursts = np.column_stack((starts, stops - 1))  # stop is exclusive, so subtract 1

    return bursts


def fill_small_gaps_in_array(arr, max_gap):
    """
    Fills small gaps (i.e., sequences of zeros) between consecutive bursts of ones in a binary array.

    The function identifies changes in the array where the value switches from 1 to 0 and vice versa,
    then calculates the size of the gaps between consecutive sequences of ones (bursts). If a gap is
    smaller than or equal to `max_gap`, it is filled by setting the values within the gap to 1.

    Parameters:
    arr (numpy array): A binary array (containing 0s and 1s) where the function will look for gaps.
    max_gap (int): The maximum size of the gap to be filled. Gaps larger than this size will not be filled.

    Returns:
    numpy array: The modified array with small gaps filled with 1s.

    Example:
    --------
    >>> arr = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1])
    >>> max_gap = 2
    >>> fill_small_gaps_in_array(arr, max_gap)
    array([1, 1, 1, 1, 1, 0, 0, 0, 1])
    """
    # Identify where the array changes from 1 to 0 and 0 to 1
    is_burst = np.diff(arr, prepend=0, append=0)
    starts = np.where(is_burst == 1)[0]
    stops = np.where(is_burst == -1)[0]

    # Calculate gap sizes between consecutive bursts
    gaps = starts[1:] - stops[:-1] - 1

    # Identify which gaps are small enough to fill
    small_gaps = np.where(gaps <= max_gap)[0]

    # Fill small gaps by setting the values in those gaps to 1
    for idx in small_gaps:
        arr[stops[idx]:starts[idx + 1]] = 1

    return arr

