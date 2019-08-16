import argparse
from math import sqrt

import mdtraj as md
import numba as nb
import numpy as np


def convert_chain_id_to_numbers(chain_id):
    import string
    di = dict(zip(string.letters, [ord(c) % 32 for c in string.letters]))
    try:
        return int(chain_id)
    except ValueError:
        return di[chain_id]


def mdtraj_selection_string(chain_id, res_id, atom_name):
    return "(chainid == %s) and (resid == %s) and (name == %s)" % \
           (convert_chain_id_to_numbers(chain_id),
            int(res_id) - 1,
            atom_name
            )


@nb.jit
def distance2fretrate(r, R0, tau0, kappa2=2./3.):
    """ Converts the DA-distance to a FRET-rate

    :param r: donor-acceptor distance
    :param R0: Forster-radius
    :param tau0: lifetime
    :param kappa2: orientation factor
    :return:
    """
    return 3./2. * kappa2 * 1./tau0*(R0/r)**6.0


@nb.jit
def kappa_distance(d1, d2, a1, a2):
    """Calculates the orientation factor kappa and the distance between two dipoles defined by the
    two vectors (d1, d2) and (a1, a2). Returns the distance between the dipoles and the orientation
    factor
    """

    # cartesian coordinates of the donor
    d11 = d1[0]
    d12 = d1[1]
    d13 = d1[2]

    d21 = d2[0]
    d22 = d2[1]
    d23 = d2[2]

    # distance between the two end points of the donor
    dD21 = sqrt((d11 - d21)*(d11 - d21) +
                 (d12 - d22)*(d12 - d22) +
                 (d13 - d23)*(d13 - d23)
                )

    # normal vector of the donor-dipole
    muD1 = (d21 - d11) / dD21
    muD2 = (d22 - d12) / dD21
    muD3 = (d23 - d13) / dD21

    # vector to the middle of the donor-dipole
    dM1 = d11 + dD21 * muD1 / 2.0
    dM2 = d12 + dD21 * muD2 / 2.0
    dM3 = d13 + dD21 * muD3 / 2.0

    ### Acceptor ###
    # cartesian coordinates of the acceptor
    a11 = a1[0]
    a12 = a1[1]
    a13 = a1[2]

    a21 = a2[0]
    a22 = a2[1]
    a23 = a2[2]

    # distance between the two end points of the acceptor
    dA21 = sqrt( (a11 - a21)*(a11 - a21) +
                 (a12 - a22)*(a12 - a22) +
                 (a13 - a23)*(a13 - a23)
    )

    # normal vector of the acceptor-dipole
    muA1 = (a21 - a11) / dA21
    muA2 = (a22 - a12) / dA21
    muA3 = (a23 - a13) / dA21

    # vector to the middle of the acceptor-dipole
    aM1 = a11 + dA21 * muA1 / 2.0
    aM2 = a12 + dA21 * muA2 / 2.0
    aM3 = a13 + dA21 * muA3 / 2.0

    # vector connecting the middle of the dipoles
    RDA1 = dM1 - aM1
    RDA2 = dM2 - aM2
    RDA3 = dM3 - aM3

    # Length of the dipole-dipole vector (distance)
    dRDA = sqrt(RDA1*RDA1 + RDA2*RDA2 + RDA3*RDA3)

    # Normalized dipole-diple vector
    nRDA1 = RDA1 / dRDA
    nRDA2 = RDA2 / dRDA
    nRDA3 = RDA3 / dRDA

    # Orientation factor kappa2
    kappa = muA1*muD1 + muA2*muD2 + muA3*muD3 - 3.0 * (muD1*nRDA1+muD2*nRDA2+muD3*nRDA3) * (muA1*nRDA1+muA2*nRDA2+muA3*nRDA3)
    return dRDA, kappa


def calculate_kappa_distance(xyz, aid1, aid2, aia1, aia2):
    """Calculates the orientation factor kappa2 and the distance of a trajectory given the atom-indices of the
    donor and the acceptor.

    :param xyz: numpy-array (frame, atom, xyz)
    :param aid1: int, atom-index of d-dipole 1
    :param aid2: int, atom-index of d-dipole 2
    :param aia1: int, atom-index of a-dipole 1
    :param aia2: int, atom-index of a-dipole 2

    :return: distances, kappa2
    """
    n_frames = xyz.shape[0]
    ks = np.empty(n_frames, dtype=np.float32)
    ds = np.empty(n_frames, dtype=np.float32)

    for i_frame in range(n_frames):
        try:
            d, k = kappa_distance(xyz[i_frame, aid1], xyz[i_frame, aid2], xyz[i_frame, aia1], xyz[i_frame, aia2])
            ks[i_frame] = k
            ds[i_frame] = d
        except:
            print("Frame skipped, calculation error")

    return ds, ks


def traj2anisotropy(traj, t_step, d1, d2, t_max):
    """
    Biophysical Journal Volume 89 December 2005 3757-3770
    Gunnar F. Schroder, Ulrike Alexiev,y and Helmut Grubmuller

    Time-dependent fluorescence depolarization and Brownian rotational diffusion coefficients of macromolecules
    Terence Tao, Biopolymers, 1969

    :return:
    """
    n_t_max = int(t_max / t_step)

    p2 = lambda x: (3.0 * x**2.0 - 1.0) / 2.0
    d = traj.xyz[:, d1, :] - traj.xyz[:, d2, :]
    n = np.sqrt((d**2).sum(axis=1))
    dn = (d.T / n).T
    r = np.zeros(n_t_max, dtype=np.float64)

    for i in range(0, len(traj) - n_t_max):
        for j in range(0, n_t_max):
            dp = np.dot(dn[i], dn[i+j])
            r[j] += 2.0 / 5.0 * p2(dp)
    r /= (len(traj) - n_t_max)
    t = np.arange(0, n_t_max, t_step)
    return t, r


@nb.jit(nopython=True)
def integrate_rate_traj(k, t_step, t_max):
    """Calculates an average array of rate constants, k, up to a maximum time t_max.

    :param k: array (list) of rate constants
    :param t_step: time between the rate constants (time-step of trajectory)
    :param t_max: maximum time for summation of rate constants
    :return:
    """
    n_t_max = int(t_max / t_step)
    sk = np.zeros(n_t_max, dtype=np.float64)
    frame_max = k.shape[0] - k.shape[0] % n_t_max
    for frame_i in range(0, frame_max - n_t_max):
        for dt_i in range(0, n_t_max):
            sk[dt_i] += k[frame_i: frame_i + dt_i].sum()
    sk /= k.shape[0]
    return sk * t_step


def traj2decay(k, t_step, t_max, tau0=None):
    """Converts a FRET-rate constant trajectory to a FRET-induced donor decay
    or a fluorescence intensity decay.
    If the parameter tau0 is None the FRET-induced donor decay is calculated. If tau0
    is a number the fluorescence decay is calculated.

    The function returns a time-axis and the decay

    :param k: the trajectory of FRET-rate constants (list-like)
    :param t_step: time between the frames
    :param t_max: maximum time of the decay
    :param tau0: fluorescence lifetime or None
    :return: time-axis and decay
    """
    ki = integrate_rate_traj(k, t_step, t_max)
    t = np.arange(0, t_max, t_step, dtype=np.float64)
    fret_decay = np.exp(-ki)
    if tau0 is None:
        return t, fret_decay
    else:
        fd0 = np.exp(-t / tau0)
        return t, fd0 * fret_decay


class CalculateTransfer(object):

    def __init__(self, trajectory_file=None, **kwargs):
        """

        :param trajectory: TrajectoryFile
            A list of PDB-filenames
        :param dipoles: bool
            If dipoles is True the transfer-efficiency is calculated using the distance and
            the orientation factor kappa2. If dipoles is False only the first atoms defining the
            Donor and the Acceptor are used to calculate the transfer-efficiency. If dipoles is True
            the first and the second atom of donor and acceptor are used.
        :param kappa2: float
            This parameter defines kappa2 if dipoles is False.
        :param verbose: bool
            If verbose is True -> output to std-out.
        """
        self._trajectory_file = trajectory_file
        self.__donorAtomID = None
        self.__acceptorAtomID = None
        self._dipoles = kwargs.get('dipoles', True)
        self.__kappa2s = None
        self.__distances = None
        self.__tauD0 = kwargs.get('tauD0', True)
        self.__forster_radius = kwargs.get('forster_radius', 52.0)

        self._stride = kwargs.get('stride', 1)
        self.verbose = kwargs.get('verbose', True)
        self._kappa2 = kwargs.get('kappa2', 0.66666666)
        self._tau0 = kwargs.get('tau0', 2.6)
        self._settings = kwargs

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, v):
        self._stride = v

    @property
    def tau0(self):
        return self._tau0

    @tau0.setter
    def tau0(self, v):
        self._tau0 = v

    @property
    def dipoles(self):
        return self._dipoles

    @dipoles.setter
    def dipoles(self, v):
        self._dipoles = v

    @property
    def trajectory_file(self):
        return self._trajectory_file

    @trajectory_file.setter
    def trajectory_file(self, v):
        self._trajectory_file = v

    @property
    def donor(self):
        return self.__donorAtomID

    @donor.setter
    def donor(self, v):
        self.__donorAtomID = v

    @property
    def acceptor(self):
        return self.__acceptorAtomID

    @acceptor.setter
    def acceptor(self, v):
        self.__acceptorAtomID = v

    @property
    def kappa2(self):
        if self.dipoles:
            return self.__kappa2s
        else:
            return self.__kappa2

    @kappa2.setter
    def kappa2(self, v):
        if isinstance(v, np.ndarray):
            self.__kappa2s = v
        else:
            self.__kappa2 = float(v)

    @property
    def distances(self):
        return self.__distances

    @property
    def tau0(self):
        return self.__tauD0

    @tau0.setter
    def tau0(self, v):
        self.__tauD0 = float(v)

    @property
    def forster_radius(self):
        return self.__forster_radius

    @forster_radius.setter
    def forster_radius(self, v):
        self.__forster_radius = float(v)

    def calc(self, output_file, **kwargs):
        verbose = kwargs.get('verbose', self.verbose)
        trajectory_file = kwargs.get('trajectory_file', self.trajectory_file)
        donor = self.donor
        acceptor = self.acceptor
        stride = kwargs.get('stride', self.stride)
        time_step = self._settings['t_step'] * stride
        dipoles = kwargs.get('dipoles', self.dipoles)
        chunk = kwargs.get('chunk', 1000)
        #traj = kwargs.get('traj', None)
        #if traj is None:
        #    md.load(trajectory_file, stride=stride)

        if verbose:
            print("Trajectory: %s" % trajectory_file)
            print("Donor-Dipole atoms: %s, %s" % donor)
            print("Acceptor-Dipole atoms: %s, %s" % acceptor)
            print("Stride: %s" % stride)
            print("time_step: %s" % time_step)
            print("Calculate kappa: %s" % dipoles)
            print("Donor fluorescence lifetime: %s" % self.tau0)
            print("Output file: %s" % output_file)
            print("-------------------------")

        # Write header
        open(output_file, 'w').write(b'Frame\ttime[ns]\tRDA[Ang]\tkappa\tkappa2\tFRETrate[1/ns]\n')
        n = 0
        try:
            for chunk in md.iterload(trajectory_file, stride=self.stride, chunk=chunk):
                if dipoles:
                    ds, ks = calculate_kappa_distance(chunk.xyz,
                                                      self.donor[0],
                                                      self.donor[1],
                                                      self.acceptor[0],
                                                      self.acceptor[1]
                                                      )
                else:
                    ks = np.zeros(chunk.n_frames, dtype=np.float32)
                    d1 = chunk.xyz[:, self.donor[0], :]
                    a1 = chunk.xyz[:, self.acceptor[0], :]
                    ds = np.sqrt(np.sum((a1 - d1)**2, axis=1))
                i = np.arange(0, ds.shape[0]) + n
                t = i * time_step
                n += ds.shape[0]

                with open(output_file, 'a') as f_handle:
                    r = np.array([
                        i * stride, # frame number
                        t, # time
                        ds * 10.0, # RDA-distance in Angstrom
                        ks,  # kappa
                        ks ** 2,  # kappa2
                        distance2fretrate(ds * 10.0, self.forster_radius, self.tau0, ks ** 2)]  # FRET-rate constant
                    ).T
                    np.savetxt(f_handle,
                               r,
                               delimiter='\t',
                               fmt=['%d', '%.3f', '%.2f', '%.4e', '%.4e', '%.4e']
                    )
        except:
            print("Probably some problems in HDF-file")
        if verbose:
            print("\nFinished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
TRAJ2FRET\n
=========\n
Converts a MD-trajectory provided as .h5 file (MDtraj format
into a trajectory of donor-acceptor (DA) distances (RDA)
and orientation factors (kappa). For convenience also the FRET-
efficiency (E) is calculated if the Forster-radius (R0)
of the DA-pair is provided. If the fluorescence lifetime of
the donor in absence of FRET (tau0) is given the FRET-rate
constant (kRET) is calculated.

By default a R0 of 52 Angstrom and a tau0 of 2.3 nanoseconds
 (ns) are used in the calculations. The dipoles of the
fluorophores are defined by choosing two atoms of the trajectory
 for each fluorophore. The atoms are chosen by providing
the chain-id, the residue number and the atom-name. This
selection is passed as a list to the arguments "-d" and "a"
for the donor and acceptor, respectively.

In case the parameter "C" deCay is set the anisotropy decay of
the donor and the acceptor are calculated and saved to the file
passed as an argument. In the generated decay file the first column
corresponds to the time-axis. The first and second column are the
anisotropy decay of the donor and acceptor, respectively.

Example
-------

python traj2fret.py traj.h5 -a A 22 CA A 32 CA -d A 101 CA A 152 CA -o output.csv

""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('trajectory_file', metavar='file', type=str,
                        help='Filename of the .h5 trajectory file')

    parser.add_argument("-o", "--output", type=str, required=True,
                        help='The output csv-file')

    parser.add_argument("-td0", "--tauD0", type=float, default=2.3, required=False,
                        help='Fluorescence lifetime of the donor in ns')

    parser.add_argument("-d", "--donor", type=lambda x: x.split(' ')[0], nargs='+', required=True,
                        help='Definition of the donor dipole chain-ids, residue-ids and the atom names, '
                             'e.g. "A 201 CA B 302 N" chooses the "CA" atom on resiude 201 of chain A'
                             'and the "N" atoms on residue 302 of chain "B". The chain is either numbered'
                             'or the PDB-typical chain identifier, e.g. "A". The resiude numbers start with 1.')
    parser.add_argument("-a", "--acceptor", nargs='+', type=lambda x: x.split(' ')[0], required=True,
                        help='Definition of the acceptor dipole analogous to the donor')

    parser.add_argument("-p", "--dipoles", help='If this is set to False the orientation factor is not calculated'
                                                'and only the distance between the first two chosen D and A atom'
                                                'are calculated.', default=1, required=False, type=int)
    parser.add_argument("-dt", "--t_step", type=float, default=None, required=False,
                        help='Time-step of trajectory in ns')
    parser.add_argument("-s", "--stride", type=int, default=1, required=False,
                        help='Load only every stride-th frame from the input file(s), to subsample')
    parser.add_argument("-v", "--verbose", type=bool, default=False, required=False,
                        help='If True outputs more information to the stdout.')
    parser.add_argument("-r", "--forster_radius", type=float, default=52.0, required=False,
                        help='Forster-radius of the dye pair.')
    parser.add_argument("-c", "--decay_file", type=str, default=None, required=False,
                        help='File to save time-resolved decays')
    parser.add_argument("-cm", "--decay_time_max", type=float, default=100.0, required=False,
                        help='Maximum time of the decays')
    parser.add_argument("-nk", "--chunk", type=int, default=5000, required=False,
                        help='Chunk size used to process trajectory')
    parser.add_argument("-qa", "--quenching_atoms", type=lambda x: x.split(' ')[0], nargs='+', default=None,
                        required=False, help='List of quenching atoms')
    parser.add_argument("-qd", "--quenching_distance", type=float, default=2.5,
                        required=False, help='Characteristic distance for PET')
    parser.add_argument("-qk", "--quenching_constant", type=float, default=3.0,
                        required=False, help='Quenching rate constant of PET (at zero distance)')



    args = parser.parse_args()
    if args.verbose:
        print("\nMDFRET (FRET-MD analysis)")
        print("=========================")
        print("")
    kwargs = vars(args)
    kwargs['dipoles'] = kwargs['dipoles'] > 0
    # Save the first frame to a PDB and pick the atom numbers
    if args.verbose:
        print("Opening first frame of trajectory.")
    frame = md.load_frame(args.trajectory_file, 0)
    if kwargs['t_step'] is None:
        try:
            kwargs['t_step'] = frame.timestep
        except ValueError:
            kwargs['t_step'] = 1.0

    calc_fret = CalculateTransfer(**kwargs)
    if args.verbose:
        print("Applying user parameters.")

    d1_atom = frame.top.select(mdtraj_selection_string(args.donor[0],
                                                       args.donor[1],
                                                       args.donor[2])
                               )[0]
    a1_atom = frame.top.select(mdtraj_selection_string(args.acceptor[0],
                                                       args.acceptor[1],
                                                       args.acceptor[2])
                               )[0]

    if kwargs['dipoles']:
        a2_atom = frame.top.select(mdtraj_selection_string(args.acceptor[3],
                                                           args.acceptor[4],
                                                           args.acceptor[5])
                                   )[0]
        d2_atom = frame.top.select(mdtraj_selection_string(args.donor[3],
                                                           args.donor[4],
                                                           args.donor[5])
                                   )[0]
    else:
        d2_atom = None
        a2_atom = None

    calc_fret.donor = d1_atom, d2_atom
    calc_fret.acceptor = a1_atom, a2_atom
    calc_fret.forster_radius = args.forster_radius

    if args.verbose:
        print("Calculating FRET-parameters.")
    calc_fret.calc(args.output, verbose=args.verbose, chunk=args.chunk)

    # If the decay-file is provided calculate the anisotropy decay
    if args.verbose:
        print("Calculating fluorescence decays.")
    if args.decay_file is not None:
        traj = md.load(args.trajectory_file)
        t_step = kwargs['t_step']
        t, rD = traj2anisotropy(traj, t_step, args.donor[0], args.donor[1], t_max=args.decay_time_max)
        t, rA = traj2anisotropy(traj, t_step, args.acceptor[0], args.acceptor[1], t_max=args.decay_time_max)
        d = np.vstack([t, rD, rA]).T
        np.savetxt(args.decay_file, d,
                   delimiter='\t',
                   fmt=['%.3f', '%.3f', '%.3f']
                   )
