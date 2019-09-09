import numpy as np


def vm_rt_to_vv_vh(
        times: np.array,
        vm: np.array,
        anisotropy_spectrum: np.array,
        g_factor: float = 1.0,
        l1: float = 0.0,
        l2: float = 0.0
):
    """Get the VV, VH decay from an VM decay given an anisotropy spectrum

    :param times: time-axis
    :param vm: magic angle decay
    :param anisotropy_spectrum: anisotropy spectrum
    :param g_factor: g-factor
    :param l1:
    :param l2:
    :return: vv, vm
    """
    rt = np.zeros_like(vm)
    for i in range(0, len(anisotropy_spectrum), 2):
        b = anisotropy_spectrum[i]
        rho = anisotropy_spectrum[i + 1]
        rt += b * np.exp(-times / rho)
    vv = vm * (1 + 2.0 * rt)
    vh = vm * (1. - g_factor * rt)
    vv_j = vv * (1. - l1) + vh * l1
    vh_j = vv * l2 + vh * (1. - l2)
    return vv_j, vh_j

