from typing import List, Tuple

import numpy as np
import pylab as p


def calc_vm_decay(
        fluorescence_lifetimes: List[Tuple[float, float]],
        times: np.array
) -> np.array:
    """Calculates a decay for a set of fluorescence lifetimes and an array of times

    :param fluorescence_lifetimes: a list composed of tuples (amplitude, lifetime)
    :param times: an numpy array of the times where the decay is evaluated
    :return:
    """
    decay = np.zeros_like(times)
    for a, tau in fluorescence_lifetimes:
        decay += a * np.exp(-times / tau)
    return decay


def vm_rt_to_vv_vh(
        times: np.array,
        vm: np.array,
        rs: List[
            Tuple[float, float]
        ],
        g_factor: float = 1.0,
        l1: float = 0.0,
        l2: float = 0.0
) -> Tuple[np.array, np.array]:
    """Calculate VV, VH from an VM decay given an anisotropy spectrum

    :param times: time-axis
    :param vm: magic angle decay
    :param rs: anisotropy spectrum
    :param g_factor: g-factor
    :param l1:
    :param l2:
    :return: vv, vm
    """
    rt = np.zeros_like(vm)
    for b, rho in rs:
        rt += b * np.exp(-times / rho)
    vv = vm * (1. + 2.0 * rt)
    vh = vm * (1. - g_factor * rt)
    vv_j = vv * (1. - l1) + vh * l1
    vh_j = vv * l2 + vh * (1. - l2)
    return vv_j, vh_j


dt = 0.097  # resolution of the histogram

# load the instrument response function (IRF)
irf_vv = np.loadtxt('IRF.txt', skiprows=1).T[1]
irf_vh = np.loadtxt('IRF.txt', skiprows=1).T[1]
times = np.arange(0, irf_vv.shape[0]) * dt

amplitudes = [0.8, 0.2]
lifetimes = [0.2, 2.0]
fluorescence_lifetimes = list(zip(amplitudes, lifetimes))
vm = calc_vm_decay(fluorescence_lifetimes, times)


r0 = 0.38
b_values = np.array([0.1, 0.9])
b_values *= r0 / np.sum(b_values)
rhos = [0.1, 20]
rs = list(zip(b_values, rhos))
vv, vh = vm_rt_to_vv_vh(times, vm, rs)

# normalize the IRFs and scales the
scale = 100000
irf_vv *= scale / sum(irf_vv)
irf_vh *= scale / sum(irf_vh)

# background
bg_vv = 10
bg_vh = 10

# periodic convolution not considered
vv_conv = np.convolve(irf_vv, vv, mode='full')[:len(vv)]
vh_conv = np.convolve(irf_vh, vh, mode='full')[:len(vh)]

vv_conv_bg = vv_conv + bg_vv
vh_conv_bg = vh_conv + bg_vh

vv_conv_noise = np.random.poisson(vv_conv_bg)
vh_conv_noise = np.random.poisson(vh_conv_bg)

p.semilogy(times, irf_vv)
p.semilogy(times, vv_conv_noise)

p.semilogy(times, irf_vh)
p.semilogy(times, vh_conv_noise)

p.show()


