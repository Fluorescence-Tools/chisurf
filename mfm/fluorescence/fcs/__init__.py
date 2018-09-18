"""
Auu
"""
from . fcs import *
import mfm


def weights(t, g, dur, cr, **kwargs):
    """
    :param t: correlation times [ms]
    :param g: correlation amplitude
    :param dur: measurement duration [s]
    :param cr: count-rate [kHz]
    :param kwargs: optional weight type 'type' either 'suren' or 'uniform' for uniform weighting or Suren-weighting
    """
    weight_type = kwargs.get('type', mfm.settings['fcs']['weight_type'])
    if weight_type == 'suren':
        dt = np.diff(t)
        dt = np.hstack([dt, dt[-1]])
        ns = dur * 1000.0 / dt
        na = dt * cr
        syn = (t < 10) + (t >= 10) * 10 ** (-np.log(t + 1e-12) / np.log(10) + 1)
        b = np.mean(g[1:5]) - 1

        #imaxhalf = len(g) - np.searchsorted(g[::-1], max(g[70:]) / 2, side='left')
        imaxhalf = np.min(np.nonzero(g < b / 2 + 1))
        tmaxhalf = t[imaxhalf]
        A = np.exp(-2 * dt / tmaxhalf)
        B = np.exp(-2 * t / tmaxhalf)
        m = t / dt
        S = (b * b / ns * ((1 + A) * (1 + B) + 2 * m * (1 - A) * B) / (1 - A) + 2 * b / ns / na * (1 + B) + (1 + b * np.sqrt(B)) / (ns * na * na)) * syn
        S = np.abs(S)
        return 1. / np.sqrt(S)
    elif weight_type == 'uniform':
        return np.ones_like(g)