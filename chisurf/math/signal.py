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
    Smooth the data using a window with the requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal (with the
    window size) at both ends so that transient parts are minimized in the beginning
    and end of the output signal.

    See also: numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve,
              scipy.signal.lfilter

    Parameters
    ----------
    data : 1D numpy-array
        Input data to be smoothed.
    window_len : int
        The dimension of the smoothing window; should be an odd integer.
    window_function_type : str, optional
        The type of window to use from the following options:
        'flat', 'hanning', 'hamming', 'bartlett', 'blackman'. A 'flat' window will produce a moving
        average smoothing. Default is 'bartlett'.

    Returns
    -------
    1D numpy-array
        The smoothed data.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> data = np.sin(x) + 0.1 * np.random.randn(100)
    >>> smoothed = window(data, window_len=11, window_function_type='hanning')
    """
    if data.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if data.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return data
    if window_function_type not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )
    s = np.r_[2 * data[0] - data[window_len:1:-1], data,
              2 * data[-1] - data[-1:-window_len:-1]]
    if window_function_type == 'flat':  # moving average
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
    """
    Calculate a shifted version of the input array using linear interpolation for non-integer shifts.

    For non-integer shifts, the shifted array is computed as a weighted combination of a rolled
    version of the array and a one-step further roll. Optionally, the parts of the shifted array
    that extend beyond the original boundaries are set to a fixed outside value.

    Parameters
    ----------
    v : 1D numpy-array
        The input array to be shifted.
    shift : float
        The floating point number by which the array is shifted.
    set_outside : bool, optional
        If True (default), values outside the original array bounds are set to `outside_value`.
    outside_value : float, optional
        The value assigned to array elements outside the original boundaries (default is 0.0).

    Returns
    -------
    numpy-array
        The shifted array.
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

    If the input array is non-empty, the function computes the autocorrelation via cross-correlation
    (using FFT) by calling `xcorr_fft`. For an empty array, an empty array is returned.

    Parameters
    ----------
    x : numpy-array
        The time series data. For multidimensional arrays, set the time axis using the ``axis``
        parameter.
    axis : int, optional
        The axis corresponding to time in `x`. Default is 0.
    normalize : bool, optional
        If True, the autocorrelation is normalized by the zero-lag value. Default is True.

    Returns
    -------
    numpy-array
        The autocorrelation function of the input time series.
    """
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
    """
    Compute the cross-correlation function of two arrays using fast Fourier transforms.

    The function computes the FFT of both input arrays (after subtracting their mean), then computes
    the inverse FFT of the product of one FFT with the complex conjugate of the other to obtain the
    cross-correlation.

    Parameters
    ----------
    in_1 : numpy-array
        The first input array.
    in_2 : numpy-array
        The second input array.
    axis : int, optional
        The axis along which to compute the FFT. Default is 0.
    normalize : bool, optional
        If True, the result is normalized by the zero-lag value. Default is True.

    Returns
    -------
    numpy-array
        The cross-correlation function.
    """
    if len(in_1) > 0 and len(in_2) > 0:
        c = in_1
        d = in_2
        n = c.shape[axis]
        m = [slice(None)] * len(c.shape)

        # Compute the FFT (after mean subtraction) for both arrays.
        f1 = np.fft.fft(c - np.mean(c, axis=axis), n=2 * n, axis=axis)
        f2 = np.fft.fft(d - np.mean(d, axis=axis), n=2 * n, axis=axis)

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
) -> typing.Tuple[float, typing.Tuple[int, int], typing.Tuple[float, float]]:
    """
    Calculate the full-width at half-maximum (FWHM) of a peak in a 1D curve.

    The function subtracts the given background from the y-values, determines the half-maximum level,
    and finds the first and last indices where the y-values exceed this level. The FWHM is computed
    as the difference between the corresponding x-values.

    Parameters
    ----------
    x_values : numpy-array
        The x-values corresponding to the data points.
    y_values : numpy-array
        The y-values (intensity or similar) of the curve.
    background : float, optional
        The background level to subtract from y-values. Default is 0.0.
    verbose : bool, optional
        If True, prints details about the computed FWHM, including boundary indices and x-values. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
          - fwhm (float): The full-width at half-maximum.
          - indices (tuple of int): The start and end indices (lb_i, ub_i) where the curve is above half maximum.
          - x_positions (tuple of float): The corresponding x-values (x_left, x_right).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.exp(-((x-5)**2)/0.5)  # a narrow peak
    >>> fwhm, (lb, ub), (x_left, x_right) = calculate_fwhm(x, y)
    >>> print(f"FWHM = {fwhm}")
    """
    y_values_bg = y_values - background
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
    """
    Generate a 2D Gaussian kernel array.

    The kernel is constructed by calculating the 1D Gaussian distribution using the cumulative
    distribution function of the normal distribution and then taking the outer product to form a 2D kernel.
    The resulting kernel is normalized so that its sum equals 1.

    Parameters
    ----------
    kernel_size : int, optional
        The size (number of rows and columns) of the 2D kernel. Default is 21.
    nsig : float, optional
        The number of standard deviations to include in the kernel; determines the width of the Gaussian. Default is 3.

    Returns
    -------
    numpy-array
        The normalized 2D Gaussian kernel.
    """
    interval = (2.0 * nsig + 1.) / kernel_size
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernel_size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


@nb.jit(nopython=True)
def _frc_histogram(lx, rx, ly, ry, f1f2, f12, f22, n_bins, bin_width):
    """
    Auxiliary function to compute the Fourier Ring Correlation (FRC) histogram.

    Parameters
    ----------
    lx : int
        Left boundary of the x-axis in Fourier space (e.g., for an image of 512 pixels, this would be -256).
    rx : int
        Right boundary of the x-axis in Fourier space (e.g., for an image of 512 pixels, this would be 255).
    ly : int
        Left boundary of the y-axis in Fourier space.
    ry : int
        Right boundary of the y-axis in Fourier space.
    f1f2 : numpy-array
        The product of the Fourier transformed images F(1) and the complex conjugate of F(2).
    f12 : numpy-array
        The squared absolute values of the F(1) Fourier transform.
    f22 : numpy-array
        The squared absolute values of the F(2) Fourier transform.
    n_bins : int
        The number of bins (rings) in the FRC histogram.
    bin_width : float
        The width of each ring (bin) in Fourier space.

    Returns
    -------
    numpy-array
        The FRC values computed for each bin.
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
    """
    Compute the Fourier Ring Correlation (FRC) between two images.

    The FRC quantifies the similarity between two images by comparing their Fourier transforms
    over concentric rings (shells) in Fourier space. The result is a normalized correlation
    coefficient computed for each ring.

    Parameters
    ----------
    image_1 : numpy-array
        The first input image.
    image_2 : numpy-array
        The second input image.
    bin_width : float, optional
        The bin width used for constructing the FRC histogram (i.e., the width of the rings in Fourier space). Default is 2.0.

    Returns
    -------
    tuple
        A tuple containing:
          - density (numpy-array): The FRC values for each ring.
          - bins (numpy-array): The bin edges corresponding to the rings in Fourier space.
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
    Identify sequences (bursts) of consecutive ones in a binary array and merge small gaps if requested.

    The function detects transitions from 0 to 1 (start of a burst) and from 1 to 0 (end of a burst).
    If `max_gap` is greater than 0, consecutive bursts separated by a gap smaller than or equal to `max_gap`
    are merged into a single burst.

    Parameters
    ----------
    arr : numpy-array
        A binary array containing 0s and 1s.
    max_gap : int, optional
        The maximum gap size between bursts that can be merged. Default is 0 (no merging).

    Returns
    -------
    numpy-array
        A 2D array where each row contains the start and end indices (inclusive) of each burst.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0])
    >>> find_bursts(arr)
    array([[1, 2],
           [5, 7]])
    >>> find_bursts(arr, max_gap=1)
    array([[1, 7]])
    """
    if len(arr) == 0 or np.all(arr == 0):  # Handle empty or all-zero input
        return np.empty((0, 2), dtype=int)

    # Find where the array changes from 0 to 1 (start of burst) and 1 to 0 (end of burst)
    is_burst = np.diff(arr, prepend=0, append=0)
    starts = np.where(is_burst == 1)[0]
    stops = np.where(is_burst == -1)[0]

    if len(starts) == 0 or len(stops) == 0:  # If no bursts are found
        return np.empty((0, 2), dtype=int)

    # If max_gap is greater than 0, merge small gaps
    if max_gap > 0:
        merged_starts = [starts[0]]
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

    # Stack the starts and stops into a 2D array (stop is exclusive, so subtract 1)
    bursts = np.column_stack((starts, stops - 1))
    return bursts


def fill_small_gaps_in_array(arr, max_gap):
    """
    Fill small gaps (sequences of zeros) between bursts of ones in a binary array.

    The function detects gaps between sequences of ones. If a gap's size is less than or equal to
    `max_gap`, the gap is filled (set to 1).

    Parameters
    ----------
    arr : numpy-array
        A binary array containing 0s and 1s.
    max_gap : int
        The maximum size of the gap that will be filled. Gaps larger than this value will remain unchanged.

    Returns
    -------
    numpy-array
        The modified array with small gaps filled.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1])
    >>> fill_small_gaps_in_array(arr.copy(), max_gap=2)
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
