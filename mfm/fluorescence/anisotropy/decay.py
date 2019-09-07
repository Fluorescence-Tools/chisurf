import numpy as np


def vm_rt_to_vv_vh(t, vm, rs, g_factor=1.0, l1=0.0, l2=0.0):
    """Get the VV, VH decay from an VM decay given an anisotropy spectrum

    :param t: time-axis
    :param vm: magic angle decay
    :param rs: anisotropy spectrum
    :param g_factor: g-factor
    :param l1:
    :param l2:
    :return: vv, vm
    """
    rt = np.zeros_like(vm)
    for i in range(0, len(rs), 2):
        b = rs[i]
        rho = rs[i+1]
        rt += b * np.exp(-t/rho)
    vv = vm * (1 + 2.0 * rt)
    vh = vm * (1. - g_factor * rt)
    vv_j = vv * (1. - l1) + vh * l1
    vh_j = vv * l2 + vh * (1. - l2)
    return vv_j, vh_j
