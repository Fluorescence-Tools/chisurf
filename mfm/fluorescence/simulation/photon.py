import numpy as np
import numpy.random as random

import mfm
from mfm.fluorescence.simulation import _simulation


def simulate_photon_trace_rate(n_ph, quench, t_step=0.01, tau0=4.10, n_traj=4, **kwargs):
    """
    This function generates a stream of photons considering quenching of the excited state
    using a mix of a purely Poissonian process and the know solution of a Poissonian process
    (a exponential decay).

    In a first step a random number in the interval between [0, 1) is generated. Using this
    random number the position on the exponential decay is looked-up. This position corresponds
    to the waiting time of the emission of a photon.

    Given the time between the excitation and the photon emission the quenching trajectory
    is used to determine whether this photon could have been emitted. Starting from the time
    of excitation to the time of emission of the photon for each quenching event is modelled
    as Poissonian process. If the excited state is quenched no photon is emitted.

    :param n_ph: int
        The total number of photons to be simulated
    :param quench: numpy-array (float32)
        The quenching array containing the instantaneous rate of quenching for each
        frame of the dye-trajectory
    :param t_step: float
        The time-step of the simulation
    :param tau0: float
        The lifetime of the dye in absence of quenching
    :param n_traj: int
        The number of photon trajectories to be simulated
    :param ravel_traj: bool
        If True the trajectories are merged/stacked into one big trajectory
    :param kwargs:
    :return: a tuple of two numpy-arrays
        The first element contains the waiting times of the unquenched dye
        The second element contains an array if actually a photon was emitted of the type np.unit8
    """
    verbose = kwargs.get('verbose', mfm.verbose)
    ravel_traj = kwargs.get('ravel_traj', True)

    n_ph /= n_traj
    n_ph = int(n_ph)
    n_frames = quench.shape[0]

    if verbose:
        print("")
        print("=====================")
        print("Photon simulation")
        print("simulation method: simulate_photon_trace_rate")
        print("nPhotons: %s" % n_ph)
        print("nTrajs: %s" % n_traj)
        print("nFrames: %s" % n_frames)
        print("=====================")
        print("")

    dts = np.empty((n_traj, n_ph), dtype=np.float32)
    phs = np.empty((n_traj, n_ph), dtype=np.uint8)
    rand_shift = random.random_integers(0, int(n_frames / 2), (n_traj, n_ph))

    for i in range(n_traj):
        dts_i = dts[i, :]
        phs_i = phs[i, :]
        rand_shift_i = rand_shift[i]
        _simulation.simulate_photons(dts_i, phs_i, n_ph, quench, t_step, tau0, n_traj, rand_shift_i)

    if ravel_traj:
        dts = dts.ravel()
        phs = phs.ravel()

    return dts, phs


def simulate_decay_quench(n_curves, decay, dt_tac, k_quench, t_step, tau0, **kwargs):
    """
    :param n_curves: int
        Number of samples taken from the trajectory
    :param decay: numpy-array
        An array later containing the fluorescence intensity
    :param dt_tac: float
        The time-resolution of the decay
    :param k_quench: array (float32)
        Array of quenching rates (FRET-rates+kQ) for each frame
    :param t_step: float
        Time resolution of the diffusion simulation
    :param tau0: float
        Lifetime of the dye in absence of quenching

    """
    verbose = kwargs.get('verbose', mfm.verbose)
    shift = np.random.randint(0, int(k_quench.shape[0] / 2), int(n_curves))
    _simulation.simulate_decay(n_curves=n_curves,
                           decay=decay,
                           dt_tac=dt_tac,
                           k_quench=k_quench,
                           t_step=t_step,
                           tau0=tau0,
                           shift=shift
    )



