"""

"""
from __future__ import annotations

from chisurf import typing
import numpy as np
import deprecation

import chisurf.fluorescence.tcspc.convolve
import chisurf.fluorescence.tcspc.corrections
import scikit_fluorescence as skf
import scikit_fluorescence.decay.tcspc
from scikit_fluorescence.decay.tcspc import counting_noise, \
    combine_parallel_perpendicular


from .tcspc import rescale_w_bg

counting_noise_combined_parallel_perpendicular = skf.decay.tcspc.combined_counting_noise_parallel_perpendicular
initial_fit_range = skf.decay.analysis.get_analysis_range

# def initial_fit_range(
#         fluorescence_decay,
#         threshold: float = 10.0,
#         area: float = 0.999
# ) -> typing.Tuple[int, int]:
#     """Determines a fitting range based on the total number of photons to be
#     fitted (fitting area).
#
#     Parameters
#     ----------
#     fluorescence_decay : numpy-array
#         a numpy array containing the photon counts
#     threshold : float
#         a threshold value. Lowest index of the fitting range is the first
#         encountered bin with a photon count higher than the threshold.
#     area : float
#         The area which should be considered for fitting. Here 1.0 corresponds
#         to all measured photons. 0.9 to 90% of the total measured photons.
#     threshold : float
#          (Default value = 10.0)
#     area: float
#          (Default value = 0.999)
#
#     Returns
#     -------
#         A tuple of integers (start, stop)
#     """
#     f = fluorescence_decay
#     x_min = np.where(f > threshold)[0][0]
#     cumsum = np.cumsum(f, dtype=np.float64)
#     s = np.sum(f, dtype=np.float64)
#     x_max = np.where(cumsum >= s * area)[0][0]
#     if fluorescence_decay[x_max] < threshold:
#         x_max = len(f) - np.searchsorted(f[::-1], [threshold])[-1]
#     return x_min, min(x_max, len(f) - 1)
