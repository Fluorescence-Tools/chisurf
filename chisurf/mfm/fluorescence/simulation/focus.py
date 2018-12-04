import numpy as np
sqrt_8 = np.sqrt(8)


def focus_3dgauss(xyz, w0, z0):
    """3D Gaussian, uniform CEF = 1

    :param xyz: cartesian coordinates
    :param w0:
    :param z0:
    :return:
    """
    x, y, z = xyz
    return sqrt_8 * np.exp(-2. * ((x**2 + y**2) / w0**2 + z**2 / z0**2))


def focus_3dgauss2(xyz, w0, z0):
    """3D Gaussian excitation and CEF

    :param xyz:
    :param w0:
    :param z0:
    :return:
    """
    x, y, z = xyz
    Iex = sqrt_8 * np.exp(-((x**2 + y**2) / w0**2 + z**2 / z0**2))
    return Iex*Iex/sqrt_8


def focus_rectangular(xyz, w0, z0):
    """rectangular excitation, uniform CEF

    :param xyz:
    :param w0:
    :param z0:
    :return:
    """
    x, y, z = xyz
    Iex = float((abs(x) < w0) and (abs(y) < w0) and (abs(z) < z0))
    return Iex



focus_functions = [focus_3dgauss, focus_3dgauss2, focus_rectangular]