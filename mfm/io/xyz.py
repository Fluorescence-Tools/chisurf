from __future__ import annotations

import numpy as np

import mfm


def write_xyz(
        filename: str,
        points: np.array,
        verbose: bool = mfm.verbose
):
    """
    Writes the points as xyz-format file. The xyz-format file can be opened and displayed for instance
    in PyMol

    :param filename: string
    :param points: array
    :param verbose: bool

    """
    if verbose:
        print("write_xyz\n")
        print("Filename: %s\n" % filename)
    with open(filename, 'w') as fp:
        npoints = len(points)
        fp.write('%i\n' % npoints)
        fp.write('Name\n')
        for p in points:
            fp.write('D %.3f %.3f %.3f\n' % (p[0], p[1], p[2]))


def read_xyz(
        filename: str
) -> np.array:
    t = np.loadtxt(
        filename,
        skiprows=2,
        usecols=(1, 2, 3),
        delimiter=" "
    )
    return t
