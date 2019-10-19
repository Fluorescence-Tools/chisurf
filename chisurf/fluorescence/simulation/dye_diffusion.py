import itertools

import numexpr as ne
import numpy as np
import tables
from chisurf.fluorescence.fps.dynamic import DiffusionSimulation

import chisurf.mfm.settings
import chisurf.mfm as mfm
from chisurf.curve import Curve
from chisurf.fitting.parameter import FittingParameterGroup, FittingParameter


def simulate_decays(
        dyes,
        decay_parameter,
        simulation_parameter,
        quenching_parameter,
        save_decays: bool = True,
        directory: str = "./",
        file_id: str = "",
        get_qy: bool = False
):
    """
    Function for batch procession of decays (use-case see notebooks: quenching_and_fret.ipynb)

    :param dyes: a dictionary of dyes
    :param decay_parameter: a instance of the `DecaySimulationParameter` class
    :param simulation_parameter: a instance of the `DiffusionSimulationParameter` class
    :param quenching_parameter: a instance of the `ProteinQuenching` class
    :param save_decays: bool
    :param directory: string pointing to directory in which the simulation results are saved

    :return: dictionary containing the simulated decays

    """
    dye_decays = dict()
    dye_qy = dict()

    for dye_key in dyes:
        print("Simulating decay: %s" % dye_key)
        dye = dyes[dye_key]
        diffusion_simulation = DiffusionSimulation(dye,
                                                   quenching_parameter,
                                                   simulation_parameter)
        diffusion_simulation.run()

        fd0_sim_curve = DyeDecay(decay_parameter, diffusion_simulation)
        fd0_sim_curve.update()
        decay = fd0_sim_curve.get_histogram()
        print("QY: %s" % fd0_sim_curve.quantum_yield)
        filename = directory+file_id+"_Donor-%s.txt" % dye_key
        decay = np.vstack(decay)
        if save_decays:
            np.savetxt(filename, decay.T)
        dye_decays[dye_key] = decay
        dye_qy[dye_key] = fd0_sim_curve.quantum_yield
    if not get_qy:
        return dye_decays
    else:
        return dye_decays, dye_qy


def simulate_fret_decays(
        donors,
        acceptors,
        decay_parameter,
        simulation_parameter,
        donor_quenching,
        acceptor_quenching,
        fret_parameter,
        save: bool = True,
        directory: str = "./",
        prefix: str = "",
        get_number_of_av_points: bool = False
):
    """
    Function for batch procession of decays (use-case see notebooks: quenching_and_fret.ipynb)

    :param donors:
    :param acceptors:
    :param decay_parameter:
    :param simulation_parameter:
    :param donor_quenching:
    :param acceptor_quenching:
    :param fret_parameter:
    :param save:
    :param directory:
    :param prefix:
    :return:
    """
    donor_keys = donors.keys()
    acceptor_keys = acceptors.keys()
    fret_decays = dict(
        (donor_key, dict()) for donor_key in donor_keys
    )

    distances = dict(
        (donor_key, dict()) for donor_key in donor_keys
    )
    n_donor = dict()
    n_acceptor = dict()

    dye_combinations = itertools.product(donor_keys, acceptor_keys)
    for donor_key, acceptor_key in dye_combinations:
        print("Simulating: %sD-%sA" % (donor_key, acceptor_key))
        donor = donors[donor_key]
        acceptor = acceptors[acceptor_key]

        donor_diffusion_simulation = DiffusionSimulation(
            donor,
            donor_quenching,
            simulation_parameter)
        donor_diffusion_simulation.run()
        donor_diffusion_simulation.save(
            directory + prefix + '_%sD_diffusion.xyz' % donor_key,
            mode='xyz',
            skip=5
        )
        donor_diffusion_simulation.av.save(
            directory + prefix + '_%sD' % donor_key)
        n_donor[donor_key] = len(donor_diffusion_simulation.av.points)

        acceptor_diffusion_simulation = DiffusionSimulation(
            acceptor,
            acceptor_quenching,
            simulation_parameter
        )
        acceptor_diffusion_simulation.run()
        acceptor_diffusion_simulation.save(
            directory + prefix + '_%sA_diffusion.xyz' % acceptor_key,
            mode='xyz',
            skip=5
        )
        acceptor_diffusion_simulation.av.save(
            directory + prefix + '_%sA' % acceptor_key
        )
        n_acceptor[acceptor_key] = len(acceptor_diffusion_simulation.av.points)

        fret_sim = FRETDecay(
            donor_diffusion_simulation,
            acceptor_diffusion_simulation,
            fret_parameter, decay_parameter
        )
        fret_sim.update()
        decay = fret_sim.get_histogram()
        decay = np.vstack(decay)
        if save:
            np.savetxt(
                directory + prefix + "_FRET-%sD-%sA-dRDA.txt" % (
                donor_key, acceptor_key),
                fret_sim.dRDA.T,
                delimiter='\t'
            )
            np.savetxt(
                directory + prefix + "_FRET-%sD-%sA.txt" % (
                donor_key, acceptor_key),
                decay.T,
                delimiter='\t'
            )
        fret_decays[donor_key][acceptor_key] = decay
        distances[donor_key][acceptor_key] = np.histogram(
            fret_sim.dRDA,
            bins=np.linspace(0, 150, 150),
            density=True
        )
    if not get_number_of_av_points:
        return fret_decays, distances
    else:
        return fret_decays, distances, n_donor, n_acceptor


class DecaySimulationParameter(
    FittingParameterGroup
):

    @property
    def dt_tac(self):
        return self._dt_tac.value

    @dt_tac.setter
    def dt_tac(self, v):
        self._dt_tac.value = v

    @property
    def n_tac(self):
        return self._n_tac.value

    @n_tac.setter
    def n_tac(self, v):
        self._n_tac.value = v

    @property
    def n_curves(self):
        return self._n_curves.value

    @n_curves.setter
    def n_curves(self, v):
        self._n_curves.value = v

    @property
    def n_photons(self):
        return self._n_photons.value

    @n_photons.setter
    def n_photons(self, v):
        self._n_photons.value = v

    @property
    def decay_mode(self):
        return self._decay_mode

    @decay_mode.setter
    def decay_mode(self, v):
        self._decay_mode = v

    @property
    def dt_mt(self):
        """The length of each macros-time step (this is used to bring the photon stream to a
        micro, macros-time form)
        """
        return self._dt_mt.value

    @dt_mt.setter
    def dt_mt(self, v):
        self._dt_mt.value = v

    @property
    def tac_range(self):
        """The tac-range used when making a histogram out of the photon trace
        """
        return self._tac_range

    @tac_range.setter
    def tac_range(self, v):
        self._tac_range = v

    def __init__(
            self,
            dt_mt: float = 100.0,
            dt_tac: float = 0.032,
            n_tac: int = 4096,
            n_curves: int = 4096,
            n_photons: float = 10e6,
            decay_mode: str = 'photon',
            **kwargs
    ):
        FittingParameterGroup.__init__(self, **kwargs)
        self._dt_mt = FittingParameter(value=dt_mt, name='dtMT[ns]')
        self._dt_tac = FittingParameter(value=dt_tac, name='dtTAC[ns]')
        self._n_tac = FittingParameter(value=n_tac, name='nTAC')
        self._n_curves = FittingParameter(value=n_curves, name='n curves')
        self._n_photons = FittingParameter(value=n_photons, name='n photons')
        self._tac_range = kwargs.get('tac_range', (0, 50))
        # decay reading_routine has to be either photons or curve
        self._decay_mode = decay_mode


class DyeDecay(Curve):

    def __init__(
            self,
            decay_parameter,
            diffusion_simulation,
            **kwargs
    ):
        self.fit = kwargs.get('fit', None)
        super().__init__(**kwargs)

        self.decay_parameter = decay_parameter
        self.diffusion = diffusion_simulation

        self._diffusion = None
        self._photon_trace = None
        self._decays = np.zeros(int(decay_parameter.n_tac), dtype=np.float64)
        self._structure = None
        self._y = None
        self._x = None

    @property
    def x_values(self):
        if self._x is None:
            self._x, self._y = self.get_histogram()
        return self._x

    @property
    def y_values(self):
        if self._y is None:
            self._x, self._y = self.get_histogram()
        return self._y

    @property
    def quantum_yield(self):
        tau0 = self.diffusion.dye.tauD0
        x = self.x_values
        yq = sum(self.y_values)
        y = sum(np.exp(-x / tau0)) * max(self.y_values)
        return float(yq / y)

    @property
    def photon_trace(self):
        return self._photon_trace

    @property
    def decays(self):
        return self._decays

    def save_photons(
            self,
            filename,
            mode: str = 'photons',
            group_title: str = 'dye_diffusion',
            hist_bins: int = 4096,
            hist_range=(0, 50),
            **kwargs
    ):
        verbose = kwargs.get('verbose', self.verbose)
        if mode == 'photons':
            dtTAC = self.diffusion.simulation_parameter.t_step
            dtMT = self.decay_parameter.dt_mt
            photons = self.photon_trace

            filters = tables.Filters(complib='blosc', complevel=9)
            h5 = tables.open_file(
                filename, mode="w", title=filename,
                filters=filters
            )
            h5.create_group("/", group_title)
            headertable = h5.createTable(
                '/' + group_title, 'header',
                description=chisurf.fio.photons.Header,
                filters=filters
            )
            headertable = h5.createTable(
                '/' + group_title, 'header',
                description=chisurf.fio.photons.Header,
                filters=filters
            )
            h5.close()
        elif mode == 'histogram':
            x, y, = self.get_histogram(hist_bins, hist_range)
            chisurf.fio.ascii.save_xy(
                filename,
                x,
                y,
                verbose,
                header_string="time\tcount"
            )

    def get_photons(
            self,
            **kwargs
    ):
        n_photons = kwargs.get('n_photons', self.decay_parameter.n_photons)
        verbose = kwargs.get('verbose', mfm.verbose)
        tau0 = kwargs.get('tau0', self.diffusion.dye.tauD0)
        kq_array = kwargs.get('quenching', self.diffusion.quenching_trajectory)
        t_step = kwargs.get('t_step', self.diffusion.simulation_parameter.t_step)

        if verbose:
            print("")
            print("Simulating decay:")
            print("----------------")
            print("Number of excitation photons: %s" % n_photons)

        dts, phs = chisurf.fluorescence.simulation.photon.simulate_photon_trace_rate(
            n_ph=n_photons,
            quench=kq_array,
            t_step=t_step,
            tau0=tau0,
            verbose=verbose
        )

        n_photons = phs.shape[0]
        n_f = phs.sum()
        if verbose or self.verbose:
            print("Number of absorbed photons: %s" % (n_photons))
            print("Number of fluorescent photons: %s" % (n_f))
            print("Quantum yield: %.2f" % (float(n_f) / n_photons))
        return dts.take(np.where(phs > 0)[0])

    def get_decay(
            self,
            **kwargs
    ):
        verbose = kwargs.get('verbose', self.verbose)
        n_curves = kwargs.get('n_curves', self.decay_parameter.n_curves)
        n_tac = kwargs.get('n_tac', self.decay_parameter.n_tac)
        kq_array = kwargs.get('quenching', self.diffusion.quenching_trajectory)
        tau0 = kwargs.get('tau0', self.diffusion.dye.tauD0)
        t_step = kwargs.get('t_step',
                            self.diffusion.simulation_parameter.t_step)
        dt_tac = kwargs.get('dt_tac', self.decay_parameter.dt_tac)
        decays = np.zeros(int(n_tac), dtype=np.float64)
        if verbose:
            print("Simulating decay:")
            print("----------------")
            print("Sum kq: %s" % kq_array)
        chisurf.fluorescence.simulation.photon.simulate_decay_quench(
            n_curves=n_curves,
            decay=decays,
            dt_tac=dt_tac,
            k_quench=kq_array,
            t_step=t_step,
            tau0=tau0,
            verbose=verbose
        )
        return decays

    def get_histogram(
            self,
            **kwargs
    ):
        normalize = kwargs.get('normalize', False)
        normalization_type = kwargs.get('normalization_type', 'area')
        verbose = kwargs.get('verbose', self.verbose)
        n_tac = kwargs.get('n_tac', self.decay_parameter.n_tac)
        dt_tac = kwargs.get('dt_tac', self.decay_parameter.dt_tac)
        tac_range = kwargs.get('tac_range', self.decay_parameter.tac_range)
        decay_mode = kwargs.get('decay_mode', self.decay_parameter.decay_mode)
        dts = kwargs.get('photons', self.photon_trace)

        if verbose:
            print("Making histogram")
            print("================")
            print("Decay-reading_routine: %s" % decay_mode)

        if decay_mode == 'photon':
            if verbose:
                print(tac_range)
                print("tac_range: (%.2f..%.2f)" % tac_range)
                print("dt_tac: %s" % dt_tac)
            y, x = np.histogram(dts, bins=np.arange(tac_range[0], tac_range[1], dt_tac))
            x = x[:-1]
        else:
            if verbose:
                print("nbins: %s" % n_tac)
                print("dt_tac: %s" % dt_tac)
            x = np.arange(n_tac) * dt_tac
            y = self._decays

        if normalize:
            if normalization_type == 'area':
                y /= sum(y)
            elif normalization_type == 'max':
                y /= max(y)

        return x.astype(np.float64), y.astype(np.float64)

    def update(self, **kwargs):
        decay_mode = kwargs.get('decay_mode', self.decay_parameter.decay_mode)
        if decay_mode != self.decay_parameter.decay_mode:
            self.decay_parameter.decay_mode = decay_mode

        if decay_mode == 'photon':
            self._photon_trace = self.get_photons()
        elif decay_mode == 'curve':
            self._decays = self.get_decay()


class FRETDecay(
    DyeDecay
):

    def __init__(
            self,
            donor_diffusion,
            acceptor_diffusion,
            fret_parameter,
            decay_parameter,
            **kwargs
    ):
        super().__init__(
            decay_parameter,
            donor_diffusion,
            **kwargs
        )
        self.verbose = kwargs.get('verbose', mfm.verbose)

        self.donor_diffusion = donor_diffusion
        self.acceptor_diffusion = acceptor_diffusion
        self.fret_parameter = fret_parameter
        self._dipole_center_atom_a = kwargs.get('dipole_center_atom_a', 0)
        self._dipole_center_atom_d = kwargs.get('dipole_center_atom_d', 0)
        self._acceptor_simulation_shift = kwargs.get('acceptor_simulation_shift', 0)
        self.decay_parameter = decay_parameter

        # Dummy initialization
        self._donor_photons = None

    @property
    def dipole_center_atom_a(self):
        return self._dipole_center_atom_a

    @dipole_center_atom_a.setter
    def dipole_center_atom_a(self, v):
        self._dipole_center_atom_a = v

    @property
    def dipole_center_atom_d(self):
        return self._dipole_center_atom_d

    @dipole_center_atom_d.setter
    def dipole_center_atom_d(self, v):
        self._dipole_center_atom_d = v

    @property
    def acceptor_simulation_shift(self):
        return self._acceptor_simulation_shift

    @acceptor_simulation_shift.setter
    def acceptor_simulation_shift(self, v):
        self._acceptor_simulation_shift = v

    @property
    def kFRET(self):
        kappa2 = self.fret_parameter.kappa2
        tau0 = self.fret_parameter.tauD0
        forster_radius = self.fret_parameter.forster_radius
        kfret = chisurf.fluorescence.general.distance_to_fret_rate_constant(
            self.dRDA,
            forster_radius,
            tau0,
            kappa2
        )
        return kfret

    @property
    def donor_photons(self):
        if self._donor_photons is None:
            self.update()
        return self._donor_photons

    @property
    def dRDA(self):
        d_coordinates = self.donor_diffusion.xyz[:, self.dipole_center_atom_d, :]
        a_coordinates = self.acceptor_diffusion.xyz[:, self.dipole_center_atom_a, :]
        # roll the coordinates by an arbitrary value
        acceptor_simulation_shift = self.acceptor_simulation_shift
        a_shift = np.roll(a_coordinates, acceptor_simulation_shift)
        d = ne.evaluate("sum((d_coordinates - a_shift)**2, axis=1)")
        d = ne.evaluate("sqrt(d)")
        return d.ravel()

    @property
    def donor_quench(self):
        kq_array = self.donor_diffusion.quenching_trajectory
        k_fret = self.kFRET
        kq = kq_array + k_fret
        return kq

    def plot_fret_trajectory(self, show=True):
        """
        This plots the FRET-trajectory of the current DA-pair
        be careful using this inside of PyQT (this is mainly for debugging)
        """
        import pylab as p

        r_da = self.dRDA
        p.subplot(2, 1, 1)
        p.plot(r_da, 'b')
        p.ylabel('RDA')

        kFRET = self.kFRET
        p.subplot(2, 1, 2)
        p.plot(kFRET, 'r')
        p.ylabel('kFRET')

        if show:
            p.show()

    def get_histogram(self, **kwargs):
        kwargs['photons'] = self.donor_photons
        return DyeDecay.get_histogram(self, **kwargs)

    def get_donor_photons(self, **kwargs):
        kwargs['tau0'] = self.donor_diffusion.dye.tauD0
        kwargs['quenching'] = self.donor_quench
        return DyeDecay.get_photons(self, **kwargs)

    def get_donor_decay(self, **kwargs):
        kwargs['quenching'] = self.donor_quench
        kwargs['tau0'] = self.donor_diffusion.dye.tauD0
        kwargs['t_step'] = self.donor_diffusion.simulation_parameter.t_step
        return self.get_decay(**kwargs)

    def update(self, **kwargs):
        decay_mode = kwargs.get('decay_mode', self.decay_parameter.decay_mode)
        if decay_mode != self.decay_parameter.decay_mode:
            self.decay_parameter.decay_mode = decay_mode
        if decay_mode == 'photon':
            self._donor_photons = self.get_donor_photons()
        elif decay_mode == 'curve':
            self._decays = self.get_donor_decay()
