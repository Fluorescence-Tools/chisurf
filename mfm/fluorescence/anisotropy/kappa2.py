import numba as nb
import numpy as np
from numpy import linalg as linalg

import mfm


@nb.jit
def kappasqAllDelta(delta, sD2, sA2, step=0.25, n_bins=31):
    """
    :param delta:
    :param sD2:
    :param sA2:
    :param step: step in degree
    :return:
    """
    #beta angles
    beta1 = np.arange(0.001, np.pi/2, step*np.pi/180.0)
    phi = np.arange(0.001, 2*np.pi, step*np.pi/180.0)
    n = beta1.shape[0]
    m = phi.shape[0]
    R = np.array([1, 0, 0])

    # kappa-square values for allowed betas
    k2 = np.zeros((n, m))
    k2hist = np.zeros(n_bins - 1)
    k2scale = np.linspace(0, 4, n_bins) # histogram bin edges

    for i in range(n):
        d1 = np.array([np.cos(beta1[i]),  0, np.sin(beta1[i])])
        n1 = np.array([-np.sin(beta1[i]), 0, np.cos(beta1[i])])
        n2 = np.array([0, 1, 0])
        for j in range(m):
            d2 = (n1*np.cos(phi[j])+n2*np.sin(phi[j]))*np.sin(delta)+d1*np.cos(delta)
            beta2 = np.arccos(abs(np.dot(d2, R)))
            k2[i, j] = mfm.fluorescence.kappasq(delta, sD2, sA2, beta1[i], beta2)
        y, x = np.histogram(k2[i, :], bins=k2scale)
        k2hist += y*np.sin(beta1[i])
    return k2scale, k2hist, k2


@nb.jit
def kappasq_all(sD2, sA2, n=100, m=100):
    k2 = np.zeros((n, m))
    k2scale = np.arange(0, 4, 0.05)
    k2hist = np.zeros(len(k2scale) - 1)
    for i in range(n):
        d1 = np.random.random((m, 3))
        d2 = np.random.random((m, 3))
        for j in range(m):
            delta = np.arccos(np.dot(d1[j, :], d2[j, :]) / linalg.norm(d1[j, :])/linalg.norm(d2[j, :]))
            beta1 = np.arccos(d1[j, 0]/linalg.norm(d1[j, :]))
            beta2 = np.arccos(d2[j, 0]/linalg.norm(d2[j, :]))
            k2[i, j] = mfm.fluorescence.kappasq(delta, sD2, sA2, beta1, beta2)
        y, x = np.histogram(k2[i, :], bins=k2scale)
        k2hist += y
    return k2scale, k2hist, k2