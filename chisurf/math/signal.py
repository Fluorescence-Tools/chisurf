from __future__ import annotations
import typing

import numpy as np
import chisurf.curve

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
        y: np.array,
        shift: float
) -> np.array:
    """Calculates an array that is shifted by a float. For non-integer shifts
    the shifted array is interpolated.

    :return:
    """
    ts = -shift
    ts_f = np.floor(ts)
    if np.isnan(ts_f):
        ts_f = 0
    tsi = int(ts_f)

    tsf = shift - tsi
    ysh = np.roll(y, tsi) * (1 - tsf) + np.roll(y, tsi + 1) * tsf
    if ts > 0:
        ysh[:tsi] = 0.0
    elif ts < 0:
        ysh[tsi:] = 0.0
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

    If the ccf could not be computed a numpy array filled with ones is returned.

    :param in_1: a numpy array that is cross-correlated with signal_b
    :param in_2: a numpy array that is cross-correlated with signal_a
    :param normalize: if normalize is True a normalized cross correlation function is returned
    :return: a cross-correlation of the two input signals
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
        acf = np.fft.ifft(f1 * np.conjugate(f2), axis=axis)[m].real
        m[axis] = 0
        if normalize:
            return acf / acf[m]
        else:
            return acf
    else:
        return np.array([], dtype=in_1.dtype)


def calculate_fwhm(
        curve: chisurf.curve.Curve,
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
    y_values = curve.y - background
    x_values = curve.x

    half_maximum = max(y_values) / 2.0
    smaller = np.where(y_values > half_maximum)[0]
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
