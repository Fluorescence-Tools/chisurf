"""utility functions for reading data"""
import numpy as np


def fcs_weights(
        times: np.array,
        correlation: np.array,
        measurement_duration,
        mean_count_rate,
        weight_type: str = 'suren'
) -> np.array:
    """
    :param times: correlation times [ms]
    :param correlation: correlation amplitude
    :param measurement_duration: measurement duration [s]
    :param mean_count_rate: count-rate [kHz]
    :param weight_type: weight type 'type' either 'suren' or 'uniform' for
    uniform weighting or Suren-weighting
    """
    if weight_type == 'suren':
        dt = np.diff(times)
        dt = np.hstack([dt, dt[-1]])
        ns = measurement_duration * 1000.0 / dt
        na = dt * mean_count_rate
        syn = (times < 10) + (times >= 10) * 10 ** (-np.log(times + 1e-12) / np.log(10) + 1)
        b = np.mean(correlation[1:5]) - 1

        # imaxhalf = len(g) - np.searchsorted(g[::-1], max(g[70:]) / 2, side='left')
        imaxhalf = np.min(np.nonzero(correlation < b / 2 + 1))
        tmaxhalf = times[imaxhalf]
        A = np.exp(-2 * dt / tmaxhalf)
        B = np.exp(-2 * times / tmaxhalf)
        m = times / dt
        S = (b * b / ns * ((1 + A) * (1 + B) + 2 * m * (1 - A) * B) / (1 - A) +
             2 * b / ns / na * (1 + B) + (
                     1 + b * np.sqrt(B)) / (ns * na * na)
             ) * syn
        S = np.abs(S)
        return 1. / np.sqrt(S)
    elif weight_type == 'uniform':
        return np.ones_like(correlation)


def downsample_trace(trace, bestlength=500):
    """
    Reduces the length of a trace so that there is no undersampling on a
    regular computer screen and the data size is not too large.

    Downsampling is performed by averaging neighboring intensity values
    for two time bins and omitting the first time bin.
    """
    # The trace is too big. Wee need to bin it.
    if len(trace) >= bestlength:
        # We want about 500 bins
        # We need to sum over intervals of length *teiler*
        teiler = int(np.floor(len(trace)/bestlength))
        newlength = int(np.floor(len(trace)/teiler))
        newsignal = np.zeros(newlength)
        # Simultaneously sum over all intervals
        for j in np.arange(teiler):
            newsignal = \
                newsignal+trace[j:newlength*teiler:teiler][:, 1]
        newsignal = 1. * newsignal / teiler
        newtimes = trace[teiler-1:newlength*teiler:teiler][:, 0]
        if len(trace) % teiler != 0:
            # We have a rest signal
            # We average it and add it to the trace
            rest = trace[newlength*teiler:][:, 1]
            lrest = len(rest)
            rest = np.array([sum(rest)/lrest])
            newsignal = np.concatenate((newsignal, rest),
                                       axis=0)
            timerest = np.array([trace[-1][0]])
            newtimes = np.concatenate((newtimes, timerest),
                                      axis=0)
        newtrace = np.zeros((len(newtimes), 2))
        newtrace[:, 0] = newtimes
        newtrace[:, 1] = newsignal
    else:
        # Declare newtrace -
        # otherwise we have a problem down three lines ;)
        newtrace = trace
    return newtrace
