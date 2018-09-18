import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, sqrt, fabs


@cython.wraparound(False)
@cython.boundscheck(False)
def min_distance_sq(float[:,:,:] xyz):
    """Determines the minimum distance in each frame of a trajectory

    :param xyz: numpy array
        The coordinates (frame nbr, atom nbr, coord)

    :return: numpy-array

    """
    cdef int i_frame, i_atom, j_atom
    cdef float x1, y1, z1
    cdef float x2, y2, z2
    cdef float distance_sq

    n_frames = xyz.shape[0]
    n_atoms = xyz.shape[1]

    cdef np.ndarray[ndim=1, dtype=np.float32_t] re = np.zeros(n_frames, dtype=np.float32)
    for i_frame in prange(n_frames, nogil=True):

        for i_atom in range(n_atoms):
            x1 = xyz[i_frame, i_atom, 0]
            y1 = xyz[i_frame, i_atom, 1]
            z1 = xyz[i_frame, i_atom, 2]

            for j_atom in range(i_atom + 1, n_atoms):
                x2 = xyz[i_frame, j_atom, 0]
                y2 = xyz[i_frame, j_atom, 1]
                z2 = xyz[i_frame, j_atom, 2]

                distance_sq = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)
                re[i_frame] = min(re[i_frame], distance_sq)
    return re


