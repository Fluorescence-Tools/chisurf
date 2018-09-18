from collections import OrderedDict
import copy
import numpy as np
from PyQt4 import QtCore, QtGui, uic
from mfm.structure.potential import cPotentials
import mfm.widgets
try:
    import numbapro as nb
except ImportError:
    import numba as nb
import math
import mfm


@nb.njit
def centroid2(atom_lookup, res_types, ca_dist, r, potential,
              cutoff=6.5, centroid_pos=4, min_dist=3.5, max_dist=19.0, bin_width=0.05, repulsion=100.0):
    """
    Calculate the potential energy given the UNRES GBV-Sidechain potential and isotropic conditions.

    Parameters
    ----------
    atom_lookup : integer array
            A two dimensional lookup array to find for each residue and atom-type the corresponding
            position within the coordinate array `r`.
    centroid_pos : integer
            This variable specifies which position in the atom_lookup contains the number of the C-beta
            atom.
    res_types : integer array
            An array containing the residue types of the residues as a one dimensional array. The residue types
            are in the order as in :py:const:mfm.structure.residue_atoms_internal:
    ca_dist : double array
            The array `ca_dist` should contain the pair-wise distances of all C-alpha atoms of the protein
    r : double array
            The two-dimensional array `r` contains all Cartesian coordinates of all atoms.
    potential : double array
            This 3-dimensional array contains pre-calculated potentials of all possible combinations of amino-
            acids. potential[1,2,10] has to contain the potential of the amino-acid (1) and (2) at a distance
            of 10 whereas the actual distance is dependent on the range in which the potential was actually
            calculated. The potential can be calculated with the small program located in:
            mfm.structure.potential.database.make_unres_lookup.py
    min_dist : double
            The minimum distance for which the parameter `potential` was calculated. This parameter has to be
            supplied and depends on the provided pre-calculated potential file.
    max_dist :double
            The minimum distance for which the parameter `potential` was calculated (see min_dist)
    bin_width : double
            The spacing between the bin of the distance points in the pre-calculated potential
    cutoff : double
            Distances between the centroids bigger than the cut-off distance will be neglected

    Citation
    --------

    Citation
    --------
    J. Comp. Chem. Vol. 18, 7, 849-873, 1997
    """

    nRes = atom_lookup.shape[0]

    E, nCont = 0.0, 0
    for residue_i in range(nRes):
        c_beta_atom_nbr_i = atom_lookup[residue_i, centroid_pos]
        if c_beta_atom_nbr_i < 0: # no side-chain -> next
            continue

        residue_type_i = res_types[residue_i]

        xi = r[c_beta_atom_nbr_i, 0]
        yi = r[c_beta_atom_nbr_i, 1]
        zi = r[c_beta_atom_nbr_i, 2]

        for residue_j in range(residue_i + 1, nRes - 1):
            c_beta_atom_nbr_j = atom_lookup[residue_j, centroid_pos]
            if c_beta_atom_nbr_j < 0 or ca_dist[residue_i, residue_j] > cutoff:
                continue
            nCont += 1

            residue_type_j = res_types[residue_j]

            xj = r[c_beta_atom_nbr_j, 0]
            yj = r[c_beta_atom_nbr_j, 1]
            zj = r[c_beta_atom_nbr_j, 2]

            dij = math.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2)

            if dij < min_dist:
                E += repulsion
            else:
                dij = min(dij, max_dist)
                bin_nbr = int(dij / bin_width)
                E += potential[residue_type_i, residue_type_j, bin_nbr]

    return nCont, E


@nb.jit(nopython=True)
def internal_potential(internal_coordinates, equilibrium_internal=None, k_bond=0.5, k_angle=0.1, k_dihedral=0.05):
    """ Simple potential based internal coordinates for C-alpha bead model
    """
    if equilibrium_internal is None:
        return 0.0
    e = 0.0
    if k_bond > 0:
        bond = k_bond * (internal_coordinates['b'] - equilibrium_internal['b'])**2
        e += bond.sum()
    if k_angle > 0:
        angle = k_angle * (internal_coordinates['a'] - equilibrium_internal['a'])**2
        e += angle.sum()
    if k_dihedral > 0:
        dihedral = k_dihedral * (internal_coordinates['d'] - equilibrium_internal['d'])**2
        e += dihedral.sum()
    return e


def internal_potential_calpha(structure, **kwargs):
    """Calculates a *internal* potential based on the similarity of the bonds length, angles, and dihedrals
    to a reference structure.

    :param structure:
    :param kwargs:
    :return:
    """
    eq = kwargs.get('equilibrium_internal', None)
    kb = kwargs.get('k_bonds', 0.5)
    ka = kwargs.get('k_angle', 0.1)
    kd = kwargs.get('k_dihedral', 0.05)
    return internal_potential(structure.internal_coordinates, eq, kb, ka, kd)


@nb.jit(nopython=True)
def lj_calpha(ca_coordinates, rm=3.8208650279):
    """Truncated Lennard Jones

    :param ca_coordinates:
    :param epsilon:
    :param rm:
    :return:
    """
    energy = 0.0  # total energy
    n_atoms = ca_coordinates.shape[0]
    rm2 = rm**2
    for i in range(n_atoms):
        x_i, y_i, z_i = ca_coordinates[i]
        for j in range(i + 1, n_atoms):
            x_j, y_j, z_j = ca_coordinates[j]
            r2 = (x_i - x_j)**2 + (y_i - y_j)**2 + (z_i - z_j)**2
            sr = (rm2 / r2) ** 3
            energy += sr * (sr - 2.)
    return energy


def lennard_jones_calpha(structure, rm=3.8208650279):
    """

    :param structure: a structure object
    :param rm: is the C-alpha equilibrium distance
    :return:
    """
    sel = mfm.structure.get_atom_index_by_name(structure.atoms, ['CA'])
    ca_atoms = structure.atoms.take(sel)
    return lj_calpha(ca_atoms['coord'][0], rm)


@nb.njit
def gb(xyz, epsilon=4.0, epsilon0=80.1, cutoff=12.0):
    """
    Generalized Born http://en.wikipedia.org/wiki/Implicit_solvation
    with cutoff-potential potential shifted by value at cutoff to smooth the energy
    :param xyz: atom-list
    :param epsilon: dielectric constant of proteins: 2-40
    :param epsilon0: dielectric constant of water 80.1
    :param cutoff: cutoff-distance in Angstrom
    :return:
    """

    nAtoms = xyz.shape[0]
    cutoff2 = cutoff ** 2
    rs = xyz['coord']
    a_radii = xyz['radius']
    a_charge = xyz['charge']
    pre = 1. / (8 * 3.14159265359) * (1. / epsilon0 - 1. / epsilon)

    energy = 0.0
    for i in range(nAtoms):
        qi = a_charge[i]
        if qi == 0.0:
            continue
        r1x = rs[i, 0]
        r1y = rs[i, 1]
        r1z = rs[i, 2]
        ai = a_radii[i]
        for j in range(i, nAtoms):
            qj = a_charge[j]
            if qj == 0.0:
                continue
            r2x = rs[j, 0]
            r2y = rs[j, 1]
            r2z = rs[j, 2]
            aj = a_radii[j]
            rij2 = (r1x - r2x)*(r1x - r2x) + (r1y - r2y)*(r1y - r2y) + (r1z - r2z)*(r1z - r2z)
            if rij2 > cutoff2:
                continue
            else:
                aij2 = ai * aj
                qij = qi * qj
                D = rij2 / (4.0 * aij2)
                fgbr = math.sqrt(rij2 + aij2 * math.exp(-D))
                fgbc = math.sqrt(cutoff + aij2 * math.exp(-D))
                energy = energy + qij / fgbr - qij/fgbc
    return energy * pre


@nb.jit
def go(ca_dist, energy_matrix, rm_matrix):
    """
    If cutoff is True the LJ-Potential is cutoff and shifted at 2.5 sigma

    :param resLookUp:
    :param ca_dist:
    :param energy_matrix:
    :param rm_matrix:
    :param cutoff:
    :return:
    """

    tmp = 0.0
    energy = 0.0  # total energy
    n_residues = ca_dist.shape[0]
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            epsilon = energy_matrix[i, j]
            rm = rm_matrix[i, j]
            r = ca_dist[i, j]
            if rm < r < rm * 2.5:
                sr = 2*(rm / r) ** 6
                tmp -= sr
                tmp += sr ** 2 / 4.0
                tmp *= epsilon
                tmp += 0.00818 * epsilon
            else:
                tmp = - epsilon
            energy += tmp
    return energy


class Ramachandran(object):

    def __init__(self, structure, filename='./mfm/structure/potential/database/rama_ala_pro_gly.npy'):
        """
        :param filename:
        :return:
        """
        self.structure = structure
        self.name = 'rama'
        self.filename = filename
        self.ramaPot = np.load(self.filename)

    def getEnergy(self):
        c = self.structure
        Erama = cPotentials.ramaEnergy(c.residue_lookup_i, c.iAtoms, self.ramaPot)
        self.E = Erama
        return Erama


class Electrostatics(object):

    def __init__(self, structure, type='gb'):
        """
        :param type:
        :return:
        """
        self.structure = structure
        self.name = 'ele'
        if type == 'gb':
            self.p = cPotentials.gb

    def getEnergy(self):
        structure = self.structure
        #Eel = cPotentials.gb(structure.xyz)
        Eel = gb(structure.xyz)
        self.E = Eel
        return Eel


class LJ_Bead(object):

    def __init__(self, structure):
        self.structure = structure
        self.name = 'LJ_bead'

    def getEnergy(self):
        structure = self.structure
        self.E = lennard_jones_calpha(self.structure.atoms['coord'])
        return self.E


class HPotential(object):

    name = 'H-Bond'

    def __init__(self, structure, cutoff_ca=8.0, cutoff_hbond=3.0, **kwargs):
        self.structure = structure
        self.cutoffH = cutoff_hbond
        self.cutoffCA = cutoff_ca
        self.potential = kwargs.get('potential', './mfm/structure/potential/database/hb.csv')
        self.oh = kwargs.get('oh', 1.0)
        self.on = kwargs.get('on', 1.0)
        self.cn = kwargs.get('cn', 1.0)
        self.ch = kwargs.get('ch', 1.0)
        self.updateParameter()

    def getEnergy(self):
        s1 = self.structure
        cca2 = self.cutoffCA ** 2
        ch2 = self.cutoffH ** 2
        nHbond, Ehbond = cPotentials.hBondLookUpAll(s1.l_res, s1.dist_ca, s1.xyz, self._hPot, cca2, ch2)
        self.E = Ehbond
        self.nHbond = nHbond
        return self.E

    def getNbrBonds(self):
        """
        :return:
        """
        if self.nHbond is None:
            return 0
        return self.nHbond

    def updateParameter(self):
        hPot = copy.deepcopy(self._hPot)
        if not self.oh:
            hPot[2, :] *= 0.0
        if not self.on:
            hPot[1, :] *= 0.0
        if not self.cn:
            hPot[3, :] *= 0.0
        if not self.ch:
            hPot[0, :] *= 0.0
        self.hPot = hPot

    @property
    def potential(self):
        return self.hPot

    @potential.setter
    def potential(self, v):
        self._hPot = np.loadtxt(v, skiprows=1, dtype=np.float64).T[1:, :]
        self.hPot = self._hPot


class HPotentialWidget(HPotential, QtGui.QWidget):

    def __init__(self, structure, parent, cutoff_ca=8.0, cutoff_hbond=3.0):
        QtGui.QWidget.__init__(self, parent=parent)
        uic.loadUi('mfm/ui/Potential_Hbond_2.ui', self)
        HPotential.__init__(self, structure)
        self.connect(self.checkBox, QtCore.SIGNAL("stateChanged (int)"), self.updateParameter)
        self.connect(self.checkBox_2, QtCore.SIGNAL("stateChanged (int)"), self.updateParameter)
        self.connect(self.checkBox_3, QtCore.SIGNAL("stateChanged (int)"), self.updateParameter)
        self.connect(self.checkBox_4, QtCore.SIGNAL("stateChanged (int)"), self.updateParameter)
        self.connect(self.actionLoad_potential, QtCore.SIGNAL('triggered()'), self.onOpenFile)
        self.cutoffCA = cutoff_ca
        self.cutoffH = cutoff_hbond

    def onOpenFile(self):
        filename = mfm.widgets.open_file('Open File', 'CSV data files (*.csv)')
        self.potential = filename

    @property
    def potential(self):
        return self.hPot

    @potential.setter
    def potential(self, v):
        self._hPot = np.loadtxt(v, skiprows=1, dtype=np.float64).T[1:, :]
        self.hPot = self._hPot
        self.lineEdit_3.setText(str(v))

    @property
    def oh(self):
        return int(self.checkBox.isChecked())

    @oh.setter
    def oh(self, v):
        self.checkBox.setChecked(v)

    @property
    def cn(self):
        return int(self.checkBox_2.isChecked())

    @cn.setter
    def cn(self, v):
        self.checkBox_2.setChecked(v)

    @property
    def ch(self):
        return int(self.checkBox_3.isChecked())

    @ch.setter
    def ch(self, v):
        self.checkBox_3.setChecked(v)

    @property
    def on(self):
        return int(self.checkBox_4.isChecked())

    @on.setter
    def on(self, v):
        self.checkBox_4.setChecked(v)

    @property
    def cutoffH(self):
        return float(self.doubleSpinBox.value())

    @cutoffH.setter
    def cutoffH(self, v):
        self.doubleSpinBox.setValue(float(v))

    @property
    def cutoffCA(self):
        return float(self.doubleSpinBox_2.value())

    @cutoffCA.setter
    def cutoffCA(self, v):
        self.doubleSpinBox_2.setValue(float(v))


class GoPotential(object):

    def __init__(self, structure):
        self.structure = structure
        self.name = 'go'

    def setGo(self):
        c = self.structure
        nnEFactor = self.nnEFactor if self.non_native_contact_on else 0.0
        cutoff = self.cutoff if self.native_cutoff_on else 1e6
        self.eMatrix, self.sMatrix = cPotentials.go_init(c.residue_lookup_r, c.dist_ca,
                                                         self.epsilon, nnEFactor, cutoff)

    def getEnergy(self):
        c = self.structure
        Etot, nNa, Ena, nNN, Enn = cPotentials.go(c.residue_lookup_r, c.dist_ca, self.eMatrix, self.sMatrix)
        #Etot = go(c., c.dist_ca, self.eMatrix, self.sMatrix)
        self.E = Etot
        return Etot

    def getNbrNonNative(self):
        return self.nNN

    def getNbrNative(self):
        return self.nNa

    def set_sMatrix(self, sMatrix):
        self.sMatrix = sMatrix

    def set_eMatrix(self, eMatrix):
        self.eMatrix = eMatrix

    def set_nMatrix(self, nMatrix):
        self.nMatrix = nMatrix


class GoPotentialWidget(GoPotential, QtGui.QWidget):

    def __init__(self, structure, parent):
        GoPotential.__init__(self, structure)
        QtGui.QWidget.__init__(self, parent=None)
        uic.loadUi('mfm/ui/Potential-CaLJ.ui', self)
        self.connect(self.lineEdit, QtCore.SIGNAL("textChanged(QString)"), self.setGo)
        self.connect(self.lineEdit_2, QtCore.SIGNAL("textChanged(QString)"), self.setGo)
        self.connect(self.lineEdit_3, QtCore.SIGNAL("textChanged(QString)"), self.setGo)

    @property
    def native_cutoff_on(self):
        return bool(self.checkBox.isChecked())

    @property
    def non_native_contact_on(self):
        return bool(self.checkBox_2.isChecked())

    @property
    def epsilon(self):
        return float(self.lineEdit.text())

    @property
    def nnEFactor(self):
        return float(self.lineEdit_2.text())

    @property
    def cutoff(self):
        return float(self.lineEdit_3.text())


class MJPotential(object):

    name = 'Miyazawa-Jernigan'

    def __init__(self, structure, filename='./mfm/structure/potential/database/mj.csv', ca_cutcoff=6.5):
        self.filename = filename
        self.structure = structure
        self.potential = filename
        self.ca_cutoff = ca_cutcoff

    @property
    def potential(self):
        return self.mjPot

    @potential.setter
    def potential(self, v):
        self.mjPot = np.loadtxt(v)

    def getEnergy(self):
        c = self.structure
        nCont, Emj = cPotentials.mj(c.l_res, c.residue_types, c.dist_ca, c.xyz, self.mjPot, cutoff=self.ca_cutoff)
        self.E = Emj
        self.nCont = nCont
        return Emj

    def getNbrContacts(self):
        return self.nCont


class MJPotentialWidget(MJPotential, QtGui.QWidget):

    def __init__(self, structure, parent, filename='./mfm/structure/potential/database/mj.csv', ca_cutoff=6.5):
        QtGui.QWidget.__init__(self, parent=None)
        uic.loadUi('mfm/ui/MJ-resource.ui', self)
        MJPotential.__init__(self, structure)
        self.connect(self.pushButton, QtCore.SIGNAL("clicked()"), self.onOpenFile)
        self.potential = filename
        self.ca_cutoff = ca_cutoff

    def onOpenFile(self):
        filename = mfm.widgets.open_file('Open MJ-Potential', 'CSV data files (*.csv)')
        self.potential = filename

    @property
    def potential(self):
        return self.mjPot

    @potential.setter
    def potential(self, v):
        self.mjPot = np.loadtxt(v)
        self.lineEdit.setText(v)

    @property
    def ca_cutoff(self):
        return float(self.lineEdit_2.text())

    @ca_cutoff.setter
    def ca_cutoff(self, v):
        self.lineEdit_2.setText(str(v))


class ASA(object):
    def __init__(self, structure, probe=1.0, n_sphere_point=590, radius=2.5):
        self.structure = structure
        self.probe = probe
        self.n_sphere_point = n_sphere_point
        self.sphere_points = cPotentials.spherePoints(n_sphere_point)
        self.radius = radius

    def getEnergy(self):
        c = self.structure
        asa = cPotentials.asa(c.rAtoms['coord'], c.residue_lookup_r, c.dist_ca,
                              self.sphere_points, self.probe, self.radius)
        return asa


class CEPotential(object):
    """
    Examples
    --------

    >>> import mfm.structure
    >>> import mfm.structure.potential

    >>> s = mfm.Structure('./sample_data/model/hgbp1/hGBP1_closed.pdb', verbose=True, make_coarse=True)
    >>> pce = mfm.structure.potential.potentials.CEPotential(s, ca_cutoff=64.0)
    >>> pce.getEnergy()
    -0.15896629131635745
    """

    name = 'Iso-UNRES'

    def __init__(self, structure, **kwargs):
        """
        scaling_factor : factor to scale energies from kCal/mol to kT=1.0 at 298K
        """
        scaling_factor = 0.593
        self.structure = structure
        self.ca_cutoff = kwargs.get('ca_cutoff', 15.0)
        self._potential = None
        potential = kwargs.get('potential', './mfm/structure/potential/database/unres.npy')
        self.potential = potential
        self.scaling_factor = scaling_factor
        # the number of the atom in the lookup table
        # (4=C-beta, 1=C-alpha, see: mfm.structure.protein internal_atom_numbers)
        self.centroid_number = kwargs.get('centroid_number', 4)
        self.repulsion = kwargs.get('repulsion', 100.0)
        self.min_dist = kwargs.get('min_dist', 3.5)
        self._max_dist = kwargs.get('max_dist', 19.0)
        self._bin_width = kwargs.get('bin_width', 0.05)

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, v):
        self._potential = np.load(v)

    def getEnergy(self, cutoff=None, **kwargs):
        cutoff = cutoff if cutoff is not None else self.ca_cutoff
        c = self.structure
        coord = np.ascontiguousarray(c.xyz)
        dist_ca = np.ascontiguousarray(c.dist_ca)
        residue_types = np.ascontiguousarray(c.residue_types)
        l_res = np.ascontiguousarray(c.l_res)
        centroid_pos = kwargs.get('centroid_number', self.centroid_number)
        repulsion = kwargs.get('repulsion', self.repulsion)
        min_dist = kwargs.get('min_dist', self.min_dist)
        nCont, E = centroid2(l_res, residue_types, dist_ca, coord,
                             self.potential, cutoff=cutoff,
                             centroid_pos=centroid_pos, min_dist=min_dist,
                             max_dist=self._max_dist, bin_width=self._bin_width,
                             repulsion=repulsion)

        self.nCont = nCont
        return float(E * self.scaling_factor)

    def getNbrContacts(self):
        return self.nCont


class CEPotentialWidget(CEPotential, QtGui.QWidget):

    def __init__(self, structure, parent, potential='./mfm/structure/potential/database/unres.npy',
                 ca_cutoff=25.0):
        QtGui.QWidget.__init__(self, parent=None)
        uic.loadUi('mfm/ui/unres-cb-resource.ui', self)
        CEPotential.__init__(self, structure, potential, ca_cutoff=ca_cutoff)
        self.connect(self.actionOpen_potential_file, QtCore.SIGNAL('triggered()'), self.onOpenPotentialFile)
        self.ca_cutoff = ca_cutoff

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, v):
        self.lineEdit.setText(str(v))
        self._potential = np.load(v)

    @property
    def ca_cutoff(self):
        return float(self.doubleSpinBox.value())

    @ca_cutoff.setter
    def ca_cutoff(self, v):
        self.doubleSpinBox.setValue(float(v))

    def onOpenPotentialFile(self):
        #filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open CE-Potential', '', 'Numpy file (*.npy)'))
        filename = mfm.widgets.open_file('Open CE-Potential', 'Numpy file (*.npy)')
        self.potential = filename


class ASA(object):

    name = 'Asa-Ca'

    def __init__(self, structure, probe=1.0, n_sphere_point=590, radius=2.5):
        self.structure = structure
        self.probe = probe
        self.n_sphere_point = n_sphere_point
        self.sphere_points = cPotentials.spherePoints(n_sphere_point)
        self.radius = radius

    def getEnergy(self):
        c = self.structure
        #def asa(double[:, :] xyz, int[:, :] resLookUp, double[:, :] caDist, double[:, :] sphere_points,
        #double probe=1.0, double radius = 2.5, char sum=1)
        asa = cPotentials.asa(c.xyz, c.l_res, c.dist_ca, self.sphere_points, self.probe, self.radius)
        return asa


class AsaWidget(ASA, QtGui.QWidget):

    def __init__(self, structure, parent):
        ASA.__init__(self, structure)
        QtGui.QWidget.__init__(self, parent=None)
        uic.loadUi('mfm/ui/Potential_Asa.ui', self)

        self.connect(self.lineEdit, QtCore.SIGNAL("textChanged(QString)"), self.setParameterSphere)
        self.connect(self.lineEdit_2, QtCore.SIGNAL("textChanged(QString)"), self.setParameterProbe)

        self.lineEdit.setText('590')
        self.lineEdit_2.setText('3.5')

    def setParameterSphere(self):
        self.n_sphere_point = int(self.lineEdit.text())

    def setParameterProbe(self):
        self.probe = float(self.lineEdit_2.text())


class RadiusGyration(QtGui.QWidget):

    name = 'Radius-Gyration'

    def __init__(self, structure, parent=None):
        QtGui.QWidget.__init__(self, parent=parent)
        self.structure = structure
        self.parent = parent

    def getEnergy(self, c=None):
        if c is None:
            c = self.structure
        return c.radius_gyration


class ClashPotential(object):

    name = 'Clash-Potential'

    def __init__(self, **kwargs):
        """
        :param kwargs:
        :return:

        Examples
        --------

        >>> import mfm.structure
        >>> import mfm.structure.potential

        >>> s = mfm.Structure('./sample_data/model/hgbp1/hGBP1_closed.pdb', verbose=True, make_coarse=True)
        >>> pce = mfm.structure.potential.potentials.ClashPotential(structure=s, clash_tolerance=6.0)
        >>> pce.getEnergy()

        """
        self.structure = kwargs.get('structure', None)
        self.clash_tolerance = kwargs.get('clash_tolerance', 2.0)
        self.covalent_radius = kwargs.get('covalent_radius', 1.5)

    def getEnergy(self):
        c = self.structure
        return cPotentials.clash_potential(c.xyz, c.vdw, self.clash_tolerance, self.covalent_radius)


class ClashPotentialWidget(ClashPotential, QtGui.QWidget):

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/potential-clash.ui', self)
        ClashPotential.__init__(self, **kwargs)

    @property
    def clash_tolerance(self):
        return float(self.doubleSpinBox.value())

    @clash_tolerance.setter
    def clash_tolerance(self, v):
        self.doubleSpinBox.setValue(v)

    @property
    def covalent_radius(self):
        return float(self.doubleSpinBox_2.value())

    @covalent_radius.setter
    def covalent_radius(self, v):
        self.doubleSpinBox_2.setValue(v)


potentialDict = OrderedDict()
potentialDict['H-Bond'] = HPotentialWidget
potentialDict['Iso-UNRES'] = CEPotentialWidget
potentialDict['Miyazawa-Jernigan'] = MJPotentialWidget
potentialDict['Go-Potential'] = GoPotentialWidget
potentialDict['ASA-Calpha'] = AsaWidget
potentialDict['Radius of Gyration'] = RadiusGyration
potentialDict['Clash potential'] = ClashPotentialWidget