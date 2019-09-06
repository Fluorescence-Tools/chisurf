from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic

import mfm
import mfm.fluorescence.anisotropy.kappa2
import mfm.math
import mfm.plots as plots
from mfm.models.model import ModelWidget
from mfm.models.tcspc.widgets import ConvolveWidget, CorrectionsWidget, GenericWidget, AnisotropyWidget
from mfm.models.tcspc.lifetime import Lifetime, LifetimeWidget, LifetimeModel, LifetimeModelWidgetBase
from mfm.fluorescence.general import distribution2rates, rates2lifetimes
from mfm.fluorescence import rda_axis
from mfm.fitting.parameter import FittingParameter, FittingParameterGroup
from mfm.fitting.widgets import FittingControllerWidget

fret_settings = mfm.settings.cs_settings['fret']


class FRETParameters(FittingParameterGroup):

    name = "FRET-parameters"

    @property
    def forster_radius(self) -> float:
        return self._forster_radius.value

    @forster_radius.setter
    def forster_radius(
            self,
            v: float
    ):
        self._forster_radius.value = v

    @property
    def tauD0(self) -> float:
        return self._tauD0.value

    @tauD0.setter
    def tauD0(
            self,
            v: float
    ):
        self._tauD0.value = v

    @property
    def kappa2(self) -> float:
        return self._kappa2.value

    @property
    def xDOnly(self) -> float:
        return np.sqrt(self._xDonly.value ** 2)

    @xDOnly.setter
    def xDOnly(
            self,
            v: float
    ):
        self._xDonly.value = v

    def __init__(self, **kwargs):
        forster_radius = kwargs.pop('forster_radius', fret_settings['forster_radius'])
        #kappa2 = kwargs.pop('kappa2', mfm.settings.cs_settings['fret']['kappa2'])
        t0 = kwargs.pop('tau0', mfm.settings.cs_settings['fret']['tau0'])
        xDOnly = kwargs.pop('x(D0)', 0.0)
        model = kwargs.get('models', None)

        self._tauD0 = FittingParameter(name='t0', label='&tau;<sub>0</sub>',
                                       value=t0, fixed=True, model=model)
        self._forster_radius = FittingParameter(name='R0', label='R<sub>0</sub>',
                                                value=forster_radius, fixed=True, model=model)
        #self._kappa2 = FittingParameter(name='k2', label='&kappa;<sup>2</sup>',
        #                                value=kappa2, fixed=True,
        #                                lb=0.0, ub=4.0,
        #                                bounds_on=False,
        #                                models=models)
        self._xDonly = FittingParameter(name='xDOnly', label='x<sup>(D,0)</sup>',
                                        value=xDOnly, fixed=False,
                                        lb=0.0, ub=1.0, bounds_on=False,
                                        model=model)

        func_calc_fret = kwargs.get('func_calc_fret', 'error')
        self._fret_efficiency = FittingParameter(name='E_FRET', label='E<sub>FRET</sub>',
                                                 value=func_calc_fret,
                                                 fixed=False, lb=0.0, ub=1.0,
                                                 bounds_on=True, model=model)
        parameters = [
            self._tauD0,
            self._forster_radius,
            #self._kappa2,
            self._xDonly,
            self._fret_efficiency
        ]
        FittingParameterGroup.__init__(self, parameters=parameters, **kwargs)
        self.name = "FRET-parameters"


class OrientationParameter(FittingParameterGroup):

    @property
    def orientation_spectrum(self):
        if self.mode == 'fast_isotropic':
            return self._k2_fast_iso
        elif self.mode == 'slow_isotropic':
            return self._k2_slow_iso
        return self._k2_fast_iso

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, v):
        self._mode = v

    def __init__(self, *args, **kwargs):
        self._mode = kwargs.get('orientation_mode', 'fast_isotropic')

        # fast isotropic
        self._k2_fast_iso = [1., 0.666]

        # slow isotropic
        k2s = np.linspace(0.01, 4, 50)
        p = mfm.fluorescence.anisotropy.kappa2.p_isotropic_orientation_factor(k2s)
        self._k2_slow_iso = mfm.fluorescence.general.two_column_to_interleaved(p, k2s)

        FittingParameterGroup.__init__(self, *args, **kwargs)


class Gaussians(FittingParameterGroup):

    name = "gaussians"

    @property
    def distribution(self) -> np.array:
        d = list()
        weights = self.amplitude
        if not self.is_distance_between_gaussians:
            args = zip(self.mean, self.sigma, self.shape)
            pdf = mfm.math.functions.distributions.generalized_normal_distribution
        else:
            args = zip(self.mean, self.sigma)
            pdf = mfm.math.functions.rdf.distance_between_gaussian
        p = mfm.math.functions.distributions.sum_distribution(rda_axis, pdf, args, weights, normalize=True)

        d.append([p, rda_axis])
        d = np.array(d)
        return d

    @property
    def mean(self):
        try:
            a = np.sqrt(np.array([g.value for g in self._gaussianMeans]) ** 2)
            return a
        except AttributeError:
            return np.array([])

    @property
    def shape(self):
        try:
            a = np.array([g.value for g in self._gaussianShape])
            return a
        except AttributeError:
            return np.array([])

    @property
    def sigma(self):
        try:
            return np.array([g.value for g in self._gaussianSigma])
        except AttributeError:
            return np.array([])

    @property
    def amplitude(self):
        try:
            a = np.sqrt(np.array([g.value for g in self._gaussianAmplitudes]) ** 2)
            a /= a.sum()
            return a
        except AttributeError:
            return np.array([])

    def finalize(self):
        a = self.amplitude
        for i, g in enumerate(self._gaussianAmplitudes):
            g.value = a[i]

    def append(
            self,
            mean: float,
            sigma: float,
            x: float,
            shape: float = 0.0
    ):
        """
        Adds/appends a new Gaussian/normal-distribution

        :param mean: float
            Mean of the new normal distribution
        :param sigma: float
            Sigma/width of the normal distribution
        :param x: float
            Amplitude of the normal distribution
        :param shape: float
            Shape of the Gaussian (generalized Gaussian, log(x)...)

        """
        n = len(self)
        m = FittingParameter(name='R(%s,%i)' % (self.short, n + 1), value=mean)
        x = FittingParameter(name='x(%s,%i)' % (self.short, n + 1), value=x)
        s = FittingParameter(name='s(%s,%i)' % (self.short, n + 1), value=sigma)
        shape = FittingParameter(name='k(%s,%i)' % (self.short, n + 1), value=shape, fixed=True)
        self._gaussianMeans.append(m)
        self._gaussianSigma.append(s)
        self._gaussianAmplitudes.append(x)
        self._gaussianShape.append(shape)

    def pop(self):
        """
        Removes the last appended Gaussian/normal-distribution
        """
        self._gaussianMeans.pop()
        self._gaussianSigma.pop()
        self._gaussianAmplitudes.pop()
        self._gaussianShape.pop()

    def __len__(self):
        return len(self._gaussianAmplitudes)

    def __init__(self, **kwargs):
        """
        This class keeps the necessary parameters to perform a fit with Gaussian/Normal-disitributed
        distances. New distance distributions are added using the methods append.

        :param donors: Lifetime
            The donor-only spectrum in form of a `Lifetime` object.
        :param forster_radius: float
            The Forster-radius of the FRET-pair in Angstrom. By default 52.0 Angstrom (FRET-pair Alexa488/Alexa647)
        :param kappa2: float
            Orientation factor. By default 2./3.
        :param t0: float
            Lifetime of the donor-fluorophore in absence of FRET.
        :param donor_only: float
            Donor-only fraction. The fraction of molecules without acceptor.
        :param no_donly: bool
            If this is True the donor-only fraction is not displayed/present.
        """
        FittingParameterGroup.__init__(self, **kwargs)
        self._name = kwargs.get('name', 'gaussians')

        self.donors = Lifetime(**kwargs)

        self._gaussianMeans = []
        self._gaussianSigma = []
        self._gaussianShape = []
        self._gaussianAmplitudes = []
        self.short = kwargs.get('short', 'G')

        self.is_distance_between_gaussians = True # If this is True than the fitted distance is the distance between two Gaussians


class GaussianWidget(Gaussians, QtWidgets.QWidget):

    def __init__(self, donors, model=None, **kwargs):

        Gaussians.__init__(self, donors=donors, model=model, **kwargs)
        QtWidgets.QWidget.__init__(self)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.gb = QtWidgets.QGroupBox()
        self.layout.addWidget(self.gb)
        self.gb.setTitle("Gaussian distances")
        self.lh = QtWidgets.QVBoxLayout()
        self.gb.setLayout(self.lh)

        self._gb = list()

        self.grid_layout = QtWidgets.QGridLayout()

        l = QtWidgets.QHBoxLayout()
        addGaussian = QtWidgets.QPushButton()
        addGaussian.setText("add")
        l.addWidget(addGaussian)

        removeGaussian = QtWidgets.QPushButton()
        removeGaussian.setText("del")
        l.addWidget(removeGaussian)
        self.lh.addLayout(l)

        self.lh.addLayout(self.grid_layout)

        addGaussian.clicked.connect(self.onAddGaussian)
        removeGaussian.clicked.connect(self.onRemoveGaussian)

        # add some initial distance
        self.append(1.0, 50.0, 6.0, 0.0)

    def onAddGaussian(self):
        t = "for f in cs.current_fit:\n" \
            "   f.models.%s.append()\n" \
            "   f.models.update()" % self.name
        mfm.run(t)

    def onRemoveGaussian(self):
        t = "for f in cs.current_fit:\n" \
            "   f.models.%s.pop()\n" \
            "   f.models.update()" % self.name
        mfm.run(t)

    def append(self, *args, **kwargs):
        Gaussians.append(self, 50.0, 6., 1., **kwargs)
        gb = QtWidgets.QGroupBox()
        n_gauss = len(self)
        gb.setTitle('G%i' % (n_gauss))
        l = QtWidgets.QVBoxLayout()

        m = self._gaussianMeans[-1].make_widget(layout=l)
        s = self._gaussianSigma[-1].make_widget(layout=l)
        shape = self._gaussianShape[-1].make_widget(layout=l)
        x = self._gaussianAmplitudes[-1].make_widget(layout=l)

        gb.setLayout(l)
        row = (n_gauss - 1) / 2 + 1
        col = (n_gauss - 1) % 2
        self.grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)

    def pop(self) -> None:
        #self._gaussianMeans.pop().close()
        #self._gaussianSigma.pop().close()
        #self._gaussianAmplitudes.pop().close()
        #self._gaussianShape.pop().close()
        self._gb.pop().close()


class DiscreteDistance(FittingParameterGroup):

    name = "discrete_distance"

    @property
    def distribution(self) -> np.array:
        distance = self.distance
        amplitude = self.amplitude
        count, bins = np.histogram(distance, bins=rda_axis, weights=amplitude)
        count.resize(count.shape[0] + 1)
        s = np.vstack([count, rda_axis])
        return np.array([s], dtype=np.float64)

    @property
    def distance(self) -> np.array:
        try:
            a = np.sqrt(np.array([g.value for g in self._distances]) ** 2)
            return a
        except AttributeError:
            return np.array([])

    @property
    def amplitude(self) -> np.array:
        try:
            a = np.sqrt(np.array([g.value for g in self._amplitudes]) ** 2)
            a /= a.sum()
            return a
        except AttributeError:
            return np.array([])

    def finalize(self):
        a = self.amplitude
        for i, g in enumerate(self._amplitudes):
            g.value = a[i]

    def append(
            self,
            mean: float,
            x: float
    ):
        n = len(self)
        m = FittingParameter(name='R(%s,%i)' % (self.short, n + 1), value=mean)
        x = FittingParameter(name='x(%s,%i)' % (self.short, n + 1), value=x)
        self._distances.append(m)
        self._amplitudes.append(x)

    def pop(self):
        """
        Removes the last appended Gaussian/normal-distribution
        """
        self._distances.pop()
        self._amplitudes.pop()

    def __len__(self):
        return len(self._amplitudes)

    def __init__(self, **kwargs):
        FittingParameterGroup.__init__(self, **kwargs)
        self.name = kwargs.get('name', 'fret_rate')
        self.short = kwargs.get('short', 'G')

        self.donors = Lifetime(**kwargs)

        self._distances = []
        self._amplitudes = []


class DiscreteDistanceWidget(DiscreteDistance, QtWidgets.QWidget):

    def __init__(self, donors, model=None, **kwargs):
        DiscreteDistance.__init__(self, donors=donors, model=model, **kwargs)
        QtWidgets.QWidget.__init__(self)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.gb = QtWidgets.QGroupBox()
        self.layout.addWidget(self.gb)
        self.gb.setTitle("FRET-rates")
        self.lh = QtWidgets.QVBoxLayout()
        self.gb.setLayout(self.lh)

        self._gb = list()

        self.grid_layout = QtWidgets.QGridLayout()

        l = QtWidgets.QHBoxLayout()
        addFRETrate = QtWidgets.QPushButton()
        addFRETrate.setText("add")
        l.addWidget(addFRETrate)

        removeFRETrate = QtWidgets.QPushButton()
        removeFRETrate.setText("del")
        l.addWidget(removeFRETrate)
        self.lh.addLayout(l)

        self.lh.addLayout(self.grid_layout)

        addFRETrate.clicked.connect(self.onAddFRETrate)
        removeFRETrate.clicked.connect(self.onRemoveFRETrate)

        # add some initial distance
        self.append(1.0, 50.0, False)

    def onAddFRETrate(self):
        t = """
for f in cs.current_fit:
    f.models.%s.append()
            """ % self.name
        mfm.run(t)

    def onRemoveFRETrate(self):
        t = """
for f in cs.current_fit:
    f.models.%s.pop()
            """ % self.name
        mfm.run(t)

    def append(self, x=None, distance=None, update=True):
        x = 1.0 if x is None else x
        m = 50.0 if distance is None else distance
        gb = QtWidgets.QGroupBox()
        n_rates = len(self)
        gb.setTitle('G%i' % (n_rates + 1))
        l = QtWidgets.QVBoxLayout()
        pm = FittingParameter(name='R(%s,%i)' % (self.short, n_rates + 1),
                              value=m, model=self.model, decimals=1,
                              bounds_on=False, lb=fret_settings['rda_min'], ub=fret_settings['rda_max'],
                              text='R', update_function=self.update)
        px = FittingParameter(name='x(%s,%i)' % (self.short, n_rates + 1), value=x,
                              model=self.model, decimals=3,
                              bounds_on=False, text='x', update_function=self.update)
        m = mfm.fitting.widgets.make_fitting_parameter_widget(pm, layout=l)
        x = mfm.fitting.widgets.make_fitting_parameter_widget(px, layout=l)

        gb.setLayout(l)
        row = n_rates / 2
        col = n_rates % 2
        self.grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)
        self._distances.append(m)
        self._amplitudes.append(x)
        mfm.run("cs.current_fit.update()")

    def pop(self):
        self._distances.pop().close()
        self._amplitudes.pop().close()
        self._gb.pop().close()
        mfm.run("cs.current_fit.update()")


class FRETModel(LifetimeModel):

    @property
    def distance_distribution(self):
        """
        The distribution of distances. The distribution should be 3D numpy array of the form

            gets distribution in form: (1,2,3)
            0: number of distribution
            1: amplitude
            2: distance

        """
        return np.array([[[1.0], [52.0]]], dtype=np.float64)

    @property
    def fret_rate_spectrum(self):
        """
        The FRET-rate spectrum. This takes the distance distribution of the models and calculated the resulting
        FRET-rate spectrum (excluding the donor-offset).
        """
        tauD0 = self.fret_parameters.tauD0
        #kappa2 = self.fret_parameters.kappa2
        forster_radius = self.fret_parameters.forster_radius
        kappa2s = self.orientation_parameter.orientation_spectrum
        rs = distribution2rates(self.distance_distribution, tauD0, kappa2s, forster_radius)
        #rs = distribution2rates(self.distance_distribution, tauD0, 2./3., forster_radius)
        r = np.hstack(rs).ravel([-1])
        return r

    @property
    def lifetime_spectrum(self) -> np.array:
        xDOnly = self.fret_parameters.xDOnly
        lt = rates2lifetimes(self.fret_rate_spectrum, self.donors.rate_spectrum, xDOnly)
        if mfm.settings.cs_settings['fret']['bin_lifetime']:
            n_lifetimes = mfm.settings.cs_settings['fret']['lifetime_bins']
            discriminate = mfm.settings.cs_settings['fret']['discriminate']
            discriminate_amplitude = mfm.settings.cs_settings['fret']['discriminate_amplitude']
            return mfm.fluorescence.tcspc.bin_lifetime_spectrum(lt, n_lifetimes=n_lifetimes,
                                                                discriminate=discriminate,
                                                                discriminator=discriminate_amplitude
                                                                )
        else:
            return lt

    @property
    def donor_lifetime_spectrum(self) -> np.array:
        """
        The donor lifetime spectrum in form amplitude, lifetime, amplitude, lifetime
        """
        return self.donors.lifetime_spectrum

    @donor_lifetime_spectrum.setter
    def donor_lifetime_spectrum(
            self,
            v: np.array
    ):
        self.model.donors.lifetime_spectrum = v

    @property
    def donor_species_averaged_lifetime(self) -> float:
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self.donors.species_averaged_lifetime

    @property
    def donor_fluorescence_averaged_lifetime(self) -> float:
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self.donors.fluorescence_averaged_lifetime

    @property
    def fret_species_averaged_lifetime(self) -> float:
        """
        The current species averages lifetime of the FRET sample xi * taui
        """
        return self.species_averaged_lifetime

    @property
    def fret_fluorescence_averaged_lifetime(self) -> float:
        return self.fluorescence_averaged_lifetime

    @property
    def fret_efficiency(self) -> float:
        return 1.0 - self.fret_species_averaged_lifetime / self.donor_species_averaged_lifetime

    @fret_efficiency.setter
    def fret_efficiency(
            self,
            v: float
    ):
        sdecay = self.fit.data.y.sum()
        tau0x = self.donor_species_averaged_lifetime
        n0 = sdecay/(tau0x*(1.-v))
        self.convolve.n0 = n0

    @property
    def donors(self):
        return self._donors

    @donors.setter
    def donors(
            self,
            v
    ):
        self._donors = v

    @property
    def reference(self):
        self._reference.update_model()
        return np.maximum(self._reference._y, 0)

    def calc_fret_efficiency(self) -> float:
        try:
            eff = 1.0 - self.fret_species_averaged_lifetime / self.donor_species_averaged_lifetime
            return eff
        except AttributeError:
            return 0.0

    def __str__(self):
        s = LifetimeModel.__str__(self)
        s += "\n"
        s += "FRET-parameter\n"
        s += "--------------\n"
        s += "FRET-efficiency: %s \n" % self.fret_efficiency
        s += "Donor tauX: %s \n" % self.donor_species_averaged_lifetime
        s += "Donor tauF: %s \n" % self.donor_fluorescence_averaged_lifetime
        return s

    def __init__(self, fit, **kwargs):
        LifetimeModel.__init__(self, fit, **kwargs)
        self.orientation_parameter = OrientationParameter(orientation_mode=mfm.settings.cs_settings['fret']['orientation_mode'])
        self.fret_parameters = kwargs.get(
            'fret_parameters',
            FRETParameters(
                func_calc_fret=self.calc_fret_efficiency,
                **kwargs
            )
        )
        self._donors = kwargs.get('lifetimes', Lifetime())

        self._reference = LifetimeModel(fit, **kwargs)
        self._reference.lifetimes = self.donors
        self._reference.convolve = self.convolve


class GaussianModel(FRETModel):
    """
    This fit models is uses multiple Gaussian/normal distributions to fit the FRET-decay. Here the donor lifetime-
    spectrum as well as the distances may be fitted. In this models it is assumed that each donor-species is fitted
    by the same FRET-rate distribution.

    References
    ----------

    .. [1]  Kalinin, S., and Johansson, L.B., Energy Migration and Transfer Rates
            are Invariant to Modeling the Fluorescence Relaxation by Discrete and Continuous
            Distributions of Lifetimes.
            J. Phys. Chem. B, 108 (2004) 3092-3097.

    """

    name = "FRET: FD (Gaussian)"

    @property
    def distance_distribution(self) -> np.array:
        dist = self.gaussians.distribution
        return dist

    def append(
            self,
            mean: float,
            sigma: float,
            species_fraction: float
    ):
        self.gaussians.append(mean, sigma, species_fraction)

    def pop(self):
        return self.gaussians.pop()

    def finalize(self):
        super(FRETModel, self).finalize()
        self.gaussians.finalize()

    def __init__(self, fit, **kwargs):
        FRETModel.__init__(self, fit, **kwargs)
        self.gaussians = kwargs.get('gaussians', Gaussians(**kwargs))


class GaussianModelWidget(GaussianModel, LifetimeModelWidgetBase):

    plot_classes = [
                       (plots.LinePlot, {'d_scalex': 'lin', 'd_scaley': 'log', 'r_scalex': 'lin', 'r_scaley': 'lin',
                                         'x_label': 'x', 'y_label': 'y', 'plot_irf': True}),
                       (plots.FitInfo, {}), (plots.DistributionPlot, {}), (plots.ParameterScanPlot, {})
                    ]

    def __init__(self, fit, **kwargs):
        donors = LifetimeWidget(parent=self, model=self, title='Donor(0)')
        gaussians = GaussianWidget(donors=donors, parent=self, model=self, short='G', **kwargs)
        GaussianModel.__init__(self, fit=fit, lifetimes=donors, gaussians=gaussians)

        LifetimeModelWidgetBase.__init__(self, fit=fit, **kwargs)
        self.lifetimes = donors

        self.layout_parameter.addWidget(donors)

        self.layout_parameter.addWidget(
            mfm.fitting.widgets.make_fitting_parameter_group_widget(self.fret_parameters)
        )

        self.layout_parameter.addWidget(gaussians)


class FRETrateModel(FRETModel):

    name = "FRET: FD (Discrete)"

    @property
    def fret_rate_spectrum(self) -> np.array:
        fret_rates = mfm.fluorescence.general.distance_to_fret_rate_constant(self.fret_rates.distance,
                                                                             self.fret_parameters.forster_radius,
                                                                             self.fret_parameters.tauD0,
                                                                             self.fret_parameters.kappa2
                                                                             )
        amplitudes = self.fret_rates.amplitude
        r = np.ravel(np.column_stack((amplitudes, fret_rates)))
        return r

    @property
    def distance_distribution(self) -> np.array:
        dist = self.fret_rates.distribution
        return dist

    def append(
            self,
            mean: float,
            species_fraction: float
    ):
        self.fret_rates.append(mean, species_fraction)

    def pop(self):
        return self.fret_rates.pop()

    def finalize(self):
        super(FRETModel, self).finalize()
        self.fret_rates.finalize()

    def __init__(self, fit, **kwargs):
        FRETModel.__init__(self, fit, **kwargs)
        self.fret_rates = kwargs.get('fret_rates', DiscreteDistance(**kwargs))


class FRETrateModelWidget(FRETrateModel, LifetimeModelWidgetBase):

    plot_classes = [
                       (plots.LinePlot, {'d_scalex': 'lin', 'd_scaley': 'log', 'r_scalex': 'lin', 'r_scaley': 'lin',
                                         'x_label': 'x', 'y_label': 'y', 'plot_irf': True}),
                       (plots.FitInfo, {}), (plots.DistributionPlot, {}), (plots.ParameterScanPlot, {})
                    ]

    def __init__(
            self,
            fit: mfm.fitting.fit.FitGroup,
            **kwargs
    ):
        donors = LifetimeWidget(parent=self, model=self, title='Donor(0)')
        fret_rates = DiscreteDistanceWidget(donors=donors, parent=self, model=self, short='G', **kwargs)
        FRETrateModel.__init__(self, fit=fit, lifetimes=donors, fret_rates=fret_rates)

        LifetimeModelWidgetBase.__init__(self, fit=fit, **kwargs)
        self.lifetimes = donors

        self.layout_parameter.addWidget(donors)
        self.layout_parameter.addWidget(self.fret_parameters.to_widget())
        self.layout_parameter.addWidget(fret_rates)


class WormLikeChainModel(FRETModel):

    name = "FD(A): Worm-like chain"

    @property
    def distance_distribution(self):
        chain_length = self._chain_length.value
        kappa = self._persistence_length.value / chain_length
        if not self.use_dye_linker:
            prob = mfm.math.functions.rdf.worm_like_chain(rda_axis, kappa, chain_length)
        else:
            sigma_linker = self._sigma_linker.value
            prob = mfm.math.functions.rdf.worm_like_chain_linker(rda_axis, kappa,
                                                                 chain_length,
                                                                 sigma_linker)
        dist = np.array([prob, rda_axis]).reshape([1, 2, fret_settings['rda_resolution']])
        return dist

    @property
    def use_dye_linker(self):
        return self._use_dye_linker

    @use_dye_linker.setter
    def use_dye_linker(self, v):
        self._use_dye_linker = v

    def __init__(self, fit, **kwargs):
        FRETModel.__init__(self, fit, **kwargs)
        self._chain_length = FittingParameter(name='length', value=100.0, model=self,
                                              decimals=1, fixed=False, text='l')
        self._use_dye_linker = kwargs.get('use_dye_linker', False)
        self._sigma_linker = FittingParameter(name='link_width', value=6.0, model=self,
                                              decimals=1, fixed=False, text='lw')
        self._persistence_length = FittingParameter(name='persistence', value=30.0,
                                                    model=self, decimals=4,
                                                    fixed=False, text='lp')


class WormLikeChainModelWidget(WormLikeChainModel, LifetimeModelWidgetBase):

    plot_classes = [
        (plots.LinePlot,
         {'d_scalex': 'lin',
          'd_scaley': 'log',
          'r_scalex': 'lin',
          'r_scaley': 'lin',
          'x_label': 'x',
          'y_label': 'y',
          'plot_irf': True}
         ),
        (plots.FitInfo, {}),
        (plots.DistributionPlot, {}),
        (plots.ParameterScanPlot, {})
    ]

    @property
    def use_dye_linker(self):
        return bool(self._use_dye_linker.isChecked())

    @use_dye_linker.setter
    def use_dye_linker(self, v):
        self._use_dye_linker.setChecked(v)

    def __init__(self, fit, **kwargs):
        donors = LifetimeWidget(parent=self, model=self, title='Donor(0)', name='donors')
        WormLikeChainModel.__init__(self, fit=fit, lifetimes=donors, **kwargs)

        LifetimeModelWidgetBase.__init__(self, fit, **kwargs)
        self.lifetimes = donors

        l = QtWidgets.QHBoxLayout()
        self._use_dye_linker = QtWidgets.QCheckBox()
        self._use_dye_linker.setText('Use linker')
        l.addWidget(self._use_dye_linker)
        self._sigma_linker = self._sigma_linker.make_widget(layout=l)

        self.layout_parameter.addWidget(self.fret_parameters.to_widget())
        self.layout_parameter.addWidget(donors)
        self.layout_parameter.addLayout(l)
        self._chain_length = self._chain_length.make_widget(layout=self.layout_parameter)
        self._persistence_length = self._persistence_length.make_widget(layout=self.layout_parameter)


class SingleDistanceModel(FRETModel):

    name = "Fixed distance distribution"

    @property
    def xDOnly(self):
        return self._xDOnly.value

    @property
    def distance_distribution(self):
        n_points = self.n_points_dist
        r = np.vstack([self.prda, self.rda]).reshape([1, 2,  n_points])
        return r

    @property
    def n_points_dist(self):
        """
        The number of points in the distribution
        """
        return self.prda.shape[0]

    @property
    def rda(self):
        return self._rda

    @rda.setter
    def rda(self, v):
        self._rda = v

    @property
    def prda(self):
        p = self._prda
        p /= sum(p)
        return p

    @prda.setter
    def prda(self, v):
        self._prda = v

    def __init__(self, **kwargs):
        FRETModel.__init__(self, **kwargs)
        self._rda = kwargs.get('rda', np.array([100.0]))
        self._prda = kwargs.get('prda', np.array([100.0]))


class SingleDistanceModelWidget(ModelWidget, SingleDistanceModel):

    def __init__(self, fit, **kwargs):
        self.anisotropy = AnisotropyWidget(model=self, short='rL', **kwargs)
        self.convolve = ConvolveWidget(fit=fit, model=self, **kwargs)
        self.donors = LifetimeWidget(parent=self, model=self, title='Donor(0)')
        self.generic = GenericWidget(fit=fit, model=self, parent=self, **kwargs)
        self.fitting_widget = QtWidgets.QLabel() if kwargs.get('disable_fit', False) else FittingControllerWidget(fit=fit, **kwargs)
        #self.errors = ErrorWidget(fit, **kwargs)
        self.corrections = CorrectionsWidget(fit, model=self, **kwargs)

        ModelWidget.__init__(self, fit=fit, icon=QtGui.QIcon(":/icons/icons/TCSPC.png"), **kwargs)

        SingleDistanceModel.__init__(self, fit=fit, convolve=self.convolve, corrections=self.corrections,
                                     generic=self.generic, lifetimes=self.donors, anisotropy=self.anisotropy)

        self._donly = self._donly.make_widget()

        uic.loadUi('mfm/ui/fitting/models/tcspc/load_distance_distibution.ui', self)
        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")
        self.actionOpen_distirbution.triggered.connect(self.load_distance_distribution)

        self.verticalLayout.addWidget(self.fitting_widget)
        self.verticalLayout.addWidget(self.convolve)
        self.verticalLayout.addWidget(self.generic)
        self.verticalLayout.addWidget(self._donly)
        self.verticalLayout.addWidget(self.donors)
        self.verticalLayout.addWidget(self.anisotropy)
        self.verticalLayout.addWidget(self.corrections)
        self.verticalLayout.addWidget(self.errors)

    def load_distance_distribution(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        #print "load_distance_distribution"
        verbose = kwargs.get('verbose', self.verbose)
        #filename = kwargs.get('filename', str(QtGui.QFileDialog.getOpenFileName(self, 'Open File')))
        filename = mfm.widgets.get_filename('Open distance distribution', 'CSV-files (*.csv)')
        self.lineEdit.setText(filename)
        csv = mfm.io.ascii.Csv(filename)
        ar = csv.data.T
        if verbose:
            print("Opening distribution")
            print("Filename: %s" % filename)
            print("Shape: %s" % ar.shape)
        self.rda = ar[0]
        self.prda = ar[1]
        self.update_model()
