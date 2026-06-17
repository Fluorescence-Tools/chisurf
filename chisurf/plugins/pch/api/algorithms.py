from __future__ import annotations

import numpy as np
from numba import njit
from scipy.signal import fftconvolve
from scipy.stats import poisson


@njit(fastmath=True)
def compute_p1(k_vals, brightness, x_vals, dx):
    n = k_vals.shape[0]
    p1 = np.zeros(n, np.float64)
    for i in range(1, n):
        k = int(k_vals[i])
        fact = 1.0
        for j in range(1, k + 1):
            fact *= j
        total = 0.0
        for xi in x_vals:
            exp_term = np.exp(-2.0 * xi * xi)
            total += (brightness * exp_term) ** k / fact * np.exp(-brightness * exp_term)
        p1[i] = total * dx
    p1[0] = 1.0 - p1[1:].sum()
    return p1


def pch_single_species(k_vals, brightness):
    x_vals = np.linspace(0, 5, 1000)
    dx = x_vals[1] - x_vals[0]
    return compute_p1(k_vals, brightness, x_vals, dx)


@njit(fastmath=True)
def convolve_pch_numba(p1, N, length):
    pk = np.zeros(length, np.float64)
    if N == 0:
        pk[0] = 1.0
        return pk
    temp = p1.copy()
    for _ in range(1, N):
        out = np.zeros(length, np.float64)
        for i in range(length):
            for j in range(length - i):
                out[i + j] += temp[i] * p1[j]
        temp = out
    return temp


def pch_open_system(k_vals, brightness, avgN, maxN=30):
    p1 = pch_single_species(k_vals, brightness)
    length = k_vals.shape[0]
    pk_tot = np.zeros(length)
    for N in range(maxN + 1):
        pk_tot += poisson.pmf(N, avgN) * convolve_pch_numba(p1, N, length)
    return pk_tot


def pch_mixture(k_vals, epsilons, avgNs):
    pk = np.zeros_like(k_vals, dtype=float)
    pk[0] = 1.0
    for eps, n in zip(epsilons, avgNs):
        pj = pch_open_system(k_vals, eps, n)
        pk = fftconvolve(pk, pj)[:len(k_vals)]
    return pk
