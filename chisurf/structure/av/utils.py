import numpy as np
import numba as nb


@nb.jit(nopython=True)
def atoms_in_reach(xyz, vdw, dmaxsq, atom_i):
    """Return the xyz coordinates of atoms within reach (defined by dmaxsq) of a list of atoms

    Parameters
    ----------
    xyz : ndarray
        Coordinates of atoms
    vdw : ndarray
        Van der Waals radii of atoms
    dmaxsq : float
        Maximum squared distance
    atom_i : int
        Attachment atom index

    Returns
    -------
    ra : ndarray
        Coordinates of atoms within reach
    vdwr : ndarray
        Van der Waals radii of atoms within reach
    """
    # copy all atoms in proximity to the dye into a smaller array and move coordinate frame to attachment point
    n_atoms = xyz.shape[0]
    atomindex = np.empty(n_atoms, dtype=np.uint32)
    r0 = xyz[atom_i]
    natomsgrid = 0
    for i in range(0, n_atoms):
        dsq = ((xyz[i] - r0)**2.0).sum()
        if (dsq < dmaxsq) and (i != atom_i):
            atomindex[natomsgrid] = i
            natomsgrid += 1

    ra = np.empty((natomsgrid, 3), dtype=np.float64)
    vdwr = np.empty(natomsgrid, dtype=np.float64)
    for i in range(natomsgrid):
        n = atomindex[i]
        ra[i] = xyz[n]
        vdwr[i] = vdw[n]
    return ra, vdwr