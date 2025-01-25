from __future__ import annotations

import numpy as np
import os

import chisurf.fio as io


def save_bvox(
    data: np.ndarray,
    filename: str
) -> None:
    """Saves as 3D voxel array containing for instance densities as a
    bvox (binary voxel) file that can be opened in Blender.

    :param data: A 3D density map
    :param filename: The output filename
    :return:
    """
    nx, ny, nz = data.shape
    header = np.array([nx, ny, nz, 1])
    filename = "".join(os.path.abspath(filename).split(".")[:-1]) + ".bvox"
    with io.zipped.open_maybe_zipped(filename, 'wb') as binfile:
        header.astype('<i4').tofile(binfile)
        data.astype('<f4').tofile(binfile)


def write_open_dx(
        filename: str,
        density: np.array,
        r0: np.array,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        dy: float,
        dz: float
) -> None:
    """Writes a density into a dx-file

    :param filename: output filename
    :param density: 3d-grid with values (densities)
    :param density:
    :param r0: origin (x, y, z)
    :param nx:
    :param ny:
    :param nz:
    :param dx:
    :param dy:
    :param dz:

    :return:
    """
    with io.zipped.open_maybe_zipped(
            filename=filename + '.dx',
            mode='w'
    ) as fp:
        s = open_dx(density, r0, (nx, ny, nz), (dx, dy, dz))
        fp.write(s)


def open_dx(
        density: np.array,
        ro: np.array,
        rn: np.array,
        dr: np.array
) -> str:
    """ Returns a open_dx string compatible with PyMOL

    :param density: 3d-grid with values (densities)
    :param ro: origin (x, y, z)
    :param rn: number of grid-points in x, y, z
    :param dr: grid-size (dx, dy, dz)
    :return: string
    """
    xo, yo, zo = ro
    xn, yn, zn = rn
    dx, dy, dz = dr
    s = ""
    s += "object 1 class gridpositions counts %i %i %i\n" % (xn, yn, zn)
    s += "origin " + str(xo) + " " + str(yo) + " " + str(zo) + "\n"
    s += "delta %s 0 0\n" % dx
    s += "delta 0 %s 0\n" % dy
    s += "delta 0 0 %s\n" % dz
    s += "object 2 class gridconnections counts %i %i %i\n" % (xn, yn, zn)
    s += "object 3 class array type double rank 0 items " + str(xn*yn*zn) + " data follows\n"
    n = 0
    for i in range(0, xn):
        for j in range(0, yn):
            for k in range(0, zn):
                s += str(density[i, j, k])
                n += 1
                if n % 3 == 0:
                    s += "\n"
                else:
                    s += " "
    s += "\nobject \"density (all) [A^-3]\" class field\n"
    return s
