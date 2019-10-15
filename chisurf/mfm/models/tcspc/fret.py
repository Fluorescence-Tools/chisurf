from __future__ import annotations

import numpy as np

import mfm
import mfm.fluorescence.anisotropy.kappa2
import mfm.math
import mfm.math.datatools
from mfm.models.tcspc.lifetime import Lifetime, LifetimeModel
from mfm.fluorescence.general import distribution2rates, rates2lifetimes
from fitting.parameter import FittingParameter, FittingParameterGroup

rda_axis = np.linspace(
    mfm.settings.fret['rda_min'],
    mfm.settings.fret['rda_max'],
    mfm.settings.fret['rda_resolution'], dtype=np.float64
)


class FRETParameters(FittingParameterGroup):
    """

    """

    name = "FRET-parameters"

    @property
    def forster_radius(
            self
    ) -> float:
        return self._forster_radius.value

    @forster_radius.setter
    def forster_radius(
            self,
            v: float
    ):
        self._forster_radius.value = v

    @property
    def tauD0(
            self
    ) -> float:
        return self._tauD0.value

    @tauD0.setter
    def tauD0(
            self,
            v: float
    ):
        self._tauD0.value = v

    @property
    def kappa2(
            self
    ) -> float:
        return self._kappa2.value

    @property
    def xDOnly(
            self
    ) -> float:
        return np.sqrt(self._xDonly.value ** 2)

    @xDOnly.setter
    def xDOnly(
            self,
            v: float
    ):
        self._xDonly.value = v

    def __init__(
            self,
            forster_radius: float = mfm.settings.fret['forster_radius'],
            tau0: float = mfm.settings.fret['tau0'],
            **kwargs
    ):
        #kappa2 = kwargs.pop('kappa2', mfm.settings.cs_settings['fret']['kappa2'])
        t0 = tau0
        xDOnly = kwargs.pop('x(D0)', 0.0)
        model = kwargs.get('models', None)

        self._tauD0 = FittingParameter(
            name='t0',
            label='&tau;<sub>0</sub>',
            value=t0,
            fixed=True,
            model=model
        )
        self._forster_radius = FittingParameter(
            name='R0',
            label='R<sub>0</sub>',
            value=forster_radius,
            fixed=True,
            model=model
        )
        #self._kappa2 = FittingParameter(name='k2', label='&kappa;<sup>2</sup>',
        #                                value=kappa2, fixed=True,
        #                                lb=0.0, ub=4.0,
        #                                bounds_on=False,
        #                                models=models)
        self._xDonly = FittingParameter(
            name='xDOnly',
            label='x<sup>(D,0)</sup>',
            value=xDOnly,
            fixed=False,
            lb=0.0,
            ub=1.0,
            bounds_on=False,
            model=model
        )

        func_calc_fret = kwargs.get('func_calc_fret', 'error')
        self._fret_efficiency = FittingParameter(
            name='E_FRET',
            label='E<sub>FRET</sub>',
            value=func_calc_fret,
            fixed=False,
            lb=0.0,
            ub=1.0,
            bounds_on=True,
            model=model
        )
        parameters = [
            self._tauD0,
            self._forster_radius,
            #self._kappa2,
            self._xDonly,
            self._fret_efficiency
        ]
        super().__init__(
            parameters=parameters,
            **kwargs
        )


class OrientationParameter(FittingParameterGroup):
    """

    """

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

    def __init__(
            self,
            *args,
            **kwargs
    ):
        self._mode = kwargs.get('orientation_mode', 'fast_isotropic')

        # fast isotropic
        self._k2_fast_iso = [1., 0.666]

        # slow isotropic
        k2s = np.linspace(0.01, 4, 50)
        p = mfm.fluorescence.anisotropy.kappa2.p_isotropic_orientation_factor(
            k2s
        )
        self._k2_slow_iso = mfm.math.datatools.two_column_to_interleaved(
            p, k2s
        )

        FittingParameterGroup.__init__(self, *args, **kwargs)


class Gaussians(FittingParameterGroup):
    """

    """

    name = "gaussians"

    @property
    def distribution(
            self
    ) -> np.array:
        d = list()
        weights = self.amplitude
        if not self.is_distance_between_gaussians:
            args = zip(self.mean, self.sigma, self.shape)
            pdf = mfm.math.functions.distributions.generalized_normal_distribution
        else:
            args = zip(self.mean, self.sigma)
            pdf = mfm.math.functions.rdf.distance_between_gaussian
        p = mfm.math.functions.distributions.sum_distribution(
            rda_axis,
            pdf,
            args,
            weights,
            normalize=True
        )

        d.append([p, rda_axis])
        d = np.array(d)
        return d

    @property
    def mean(
            self
    ):
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
            a = np.sqrt(
                np.array([g.value for g in self._gaussianAmplitudes]) ** 2
            )
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
        m = FittingParameter(
            name='R(%s,%i)' % (self.short, n + 1),
            value=mean
        )
        x = FittingParameter(
            name='x(%s,%i)' % (self.short, n + 1),
            value=x
        )
        s = FittingParameter(
            name='s(%s,%i)' % (self.short, n + 1),
            value=sigma
        )
        shape = FittingParameter(
            name='k(%s,%i)' % (self.short, n + 1),
            value=shape,
            fixed=True
        )
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

    def __init__(
            self,
            short: str = 'G',
            is_distance_between_gaussians: bool = True,
            name: str = 'gaussians',
            **kwargs
    ):
        """
        This class keeps the necessary parameters to perform a fit with
        Gaussian/Normal-disitributed distances. New distance distributions
        are added using the methods append.

        :param donors: Lifetime
            The donor-only spectrum in form of a `Lifetime` object.
        :param forster_radius: float
            The Forster-radius of the FRET-pair in Angstrom. By default 52.0
            Angstrom (FRET-pair Alexa488/Alexa647)
        :param kappa2: float
            Orientation factor. By default 2./3.
        :param t0: float
            Lifetime of the donor-fluorophore in absence of FRET.
        :param donor_only: float
            Donor-only fraction. The fraction of molecules without acceptor.
        :param no_donly: bool
            If this is True the donor-only fraction is not displayed/present.
        """
        super().__init__(
            name=name,
            **kwargs
        )
        self.donors = Lifetime(
            **kwargs
        )
        self.short = short
        self._gaussianMeans = list()
        self._gaussianSigma = list()
        self._gaussianShape = list()
        self._gaussianAmplitudes = list()
        self.is_distance_between_gaussians = is_distance_between_gaussians


class DiscreteDistance(FittingParameterGroup):
    """

    """

    name = "discrete_distance"

    @property
    def distribution(
            self
    ) -> np.array:
        distance = self.distance
        amplitude = self.amplitude
        count, bins = np.histogram(distance, bins=rda_axis, weights=amplitude)
        count.resize(count.shape[0] + 1)
        s = np.vstack([count, rda_axis])
        return np.array([s], dtype=np.float64)

    @property
    def distance(
            self
    ) -> np.array:
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
        self._distances.append(
            FittingParameter(
                name='R(%s,%i)' % (self.short, n + 1),
                value=mean
            )
        )
        self._amplitudes.append(
            FittingParameter(
                name='x(%s,%i)' % (self.short, n + 1),
                value=x
            )
        )

    def pop(self):
        """
        Removes the last appended Gaussian/normal-distribution
        """
        self._distances.pop()
        self._amplitudes.pop()

    def __len__(self):
        return len(self._amplitudes)

    def __init__(
            self,
            name: str = 'fret_rate',
            short: str = 'G',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.short = short
        self.donors = Lifetime(**kwargs)
        self._distances = list()
        self._amplitudes = list()


class FRETModel(LifetimeModel):
    """

    """

    @property
    def distance_distribution(
            self
    ) -> np.array:
        """
        The distribution of distances. The distribution should be 3D numpy array of the form

            gets distribution in form: (1,2,3)
            0: number of distribution
            1: amplitude
            2: distance

        """
        return np.array([[[1.0], [52.0]]], dtype=np.float64)

    @property
    def fret_rate_spectrum(
            self
    ) -> np.array:
        """
        The FRET-rate spectrum. This takes the distance distribution of the models and calculated the resulting
        FRET-rate spectrum (excluding the donor-offset).
        """
        tauD0 = self.fret_parameters.tauD0
        #kappa2 = self.fret_parameters.kappa2
        forster_radius = self.fret_parameters.forster_radius
        kappa2s = self.orientation_parameter.orientation_spectrum
        rs = distribution2rates(
            self.distance_distribution,
            tauD0,
            kappa2s,
            forster_radius
        )
        #rs = distribution2rates(self.distance_distribution, tauD0, 2./3., forster_radius)
        r = np.hstack(rs).ravel([-1])
        return r

    @property
    def lifetime_spectrum(
            self
    ) -> np.array:
        xDOnly = self.fret_parameters.xDOnly
        lt = rates2lifetimes(
            self.fret_rate_spectrum,
            self.donors.rate_spectrum,
            xDOnly
        )
        if mfm.settings.cs_settings['fret']['bin_lifetime']:
            n_lifetimes = mfm.settings.cs_settings['fret']['lifetime_bins']
            discriminate = mfm.settings.cs_settings['fret']['discriminate']
            discriminate_amplitude = mfm.settings.cs_settings['fret'][
                'discriminate_amplitude']
            return mfm.fluorescence.tcspc.bin_lifetime_spectrum(
                lt, n_lifetimes=n_lifetimes,
                discriminate=discriminate,
                discriminator=discriminate_amplitude
            )
        else:
            return lt

    @property
    def donor_lifetime_spectrum(
            self
    ) -> np.array:
        """
        The donor lifetime spectrum in form amplitude, lifetime, amplitude,
        lifetime.
        """
        return self.donors.lifetime_spectrum

    @donor_lifetime_spectrum.setter
    def donor_lifetime_spectrum(
            self,
            v: np.array
    ):
        self.model.donors.lifetime_spectrum = v

    @property
    def donor_species_averaged_lifetime(
            self
    ) -> float:
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self.donors.species_averaged_lifetime

    @property
    def donor_fluorescence_averaged_lifetime(
            self
    ) -> float:
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self.donors.fluorescence_averaged_lifetime

    @property
    def fret_species_averaged_lifetime(
            self
    ) -> float:
        """
        The current species averages lifetime of the FRET sample xi * taui
        """
        return self.species_averaged_lifetime

    @property
    def fret_fluorescence_averaged_lifetime(
            self
    ) -> float:
        return self.fluorescence_averaged_lifetime

    @property
    def fret_efficiency(
            self
    ) -> float:
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
    def donors(
            self
    ) -> Lifetime:
        return self._donors

    @donors.setter
    def donors(
            self,
            v: Lifetime
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

    def __init__(
            self,
            fit: fitting.fit.FitGroup,
            lifetimes: Lifetime = None,
            **kwargs
    ):
        super().__init__(
            fit,
            **kwargs
        )
        self.orientation_parameter = OrientationParameter(
            orientation_mode=mfm.settings.cs_settings['fret']['orientation_mode']
        )
        self.fret_parameters = kwargs.get(
            'fret_parameters',
            FRETParameters(
                func_calc_fret=self.calc_fret_efficiency,
                **kwargs
            )
        )
        if lifetimes is None:
            lifetimes = Lifetime()

        self._donors = lifetimes
        self._reference = LifetimeModel(
            fit,
            **kwargs
        )
        self._reference.lifetimes = self.donors
        self._reference.convolve = self.convolve


class GaussianModel(FRETModel):
    """
    This fit models is uses multiple Gaussian/normal distributions to fit
    the FRET-decay. Here the donor lifetime-
    spectrum as well as the distances may be fitted. In this models it is
    assumed that each donor-species is fitted
    by the same FRET-rate distribution.

    References
    ----------

    .. [1]  Kalinin, S., and Johansson, L.B., Energy Migration and Transfer Rates
            are Invariant to Modeling the Fluorescence Relaxation by Discrete
            and Continuous Distributions of Lifetimes.
            J. Phys. Chem. B, 108 (2004) 3092-3097.

    """

    name = "FRET: FD (Gaussian)"

    @property
    def distance_distribution(
            self
    ) -> np.array:
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
        super().finalize()
        self.gaussians.finalize()

    def __init__(
            self,
            fit: fitting.fit.FitGroup,
            **kwargs
    ):
        super().__init__(
            fit,
            **kwargs
        )
        self.gaussians = kwargs.get('gaussians', Gaussians(**kwargs))


class FRETrateModel(FRETModel):
    """

    """

    name = "FRET: FD (Discrete)"

    @property
    def fret_rate_spectrum(
            self
    ) -> np.array:
        fret_rates = mfm.fluorescence.general.distance_to_fret_rate_constant(
            self.fret_rates.distance,
            self.fret_parameters.forster_radius,
            self.fret_parameters.tauD0,
            self.fret_parameters.kappa2
        )
        amplitudes = self.fret_rates.amplitude
        r = np.ravel(np.column_stack((amplitudes, fret_rates)))
        return r

    @property
    def distance_distribution(
            self
    ) -> np.array:
        dist = self.fret_rates.distribution
        return dist

    def append(
            self,
            mean: float,
            species_fraction: float
    ):
        self.fret_rates.append(mean, species_fraction)

    def pop(
            self
    ):
        return self.fret_rates.pop()

    def finalize(
            self
    ):
        super().finalize()
        self.fret_rates.finalize()

    def __init__(
            self,
            fit: fitting.fit.FitGroup,
            fret_rates: DiscreteDistance = None,
            **kwargs
    ):
        """

        :param fit:
        :param fret_rates:
        :param kwargs:
        """
        FRETModel.__init__(
            self,
            fit,
            **kwargs
        )
        if fret_rates is None:
            fret_rates = DiscreteDistance(**kwargs)
        self.fret_rates = fret_rates


class WormLikeChainModel(FRETModel):
    """

    """

    name = "FD(A): Worm-like chain"

    @property
    def distance_distribution(self):
        chain_length = self._chain_length.value
        kappa = self._persistence_length.value / chain_length
        if not self.use_dye_linker:
            prob = mfm.math.functions.rdf.worm_like_chain(
                rda_axis,
                kappa,
                chain_length
            )
        else:
            sigma_linker = self._sigma_linker.value
            prob = mfm.math.functions.rdf.worm_like_chain_linker(
                rda_axis, kappa,
                chain_length,
                sigma_linker
            )
        dist = np.array([prob, rda_axis]).reshape(
            [1, 2, mfm.settings.fret['rda_resolution']]
        )
        return dist

    @property
    def use_dye_linker(self):
        return self._use_dye_linker

    @use_dye_linker.setter
    def use_dye_linker(self, v):
        self._use_dye_linker = v

    def __init__(
            self,
            fit: fitting.fit.FitGroup,
            use_dye_linker: bool = False,
            **kwargs
    ):
        super().__init__(
            fit,
            **kwargs
        )
        self._chain_length = FittingParameter(
            name='length',
            value=100.0,
            model=self,
            fixed=False,
            text='l'
        )
        self._use_dye_linker = use_dye_linker
        self._sigma_linker = FittingParameter(
            name='link_width',
            value=6.0,
            model=self,
            fixed=False,
            text='lw'
        )
        self._persistence_length = FittingParameter(
            name='persistence',
            value=30.0,
            model=self,
            fixed=False,
            text='lp'
        )


class SingleDistanceModel(FRETModel):
    """

    """

    name = "Fixed distance distribution"

    @property
    def xDOnly(
            self
    ) -> float:
        return self._xDOnly.value

    @property
    def distance_distribution(
            self
    ) -> np.array:
        n_points = self.n_points_dist
        r = np.vstack(
            [self.prda, self.rda]
        ).reshape([1, 2,  n_points])
        return r

    @property
    def n_points_dist(
            self
    ) -> int:
        """
        The number of points in the distribution
        """
        return self.prda.shape[0]

    @property
    def rda(
            self
    ) -> np.array:
        return self._rda

    @rda.setter
    def rda(self, v):
        self._rda = v

    @property
    def prda(
            self
    ) -> np.array:
        p = self._prda
        p /= sum(p)
        return p

    @prda.setter
    def prda(
            self,
            v: np.array
    ):
        self._prda = v

    def __init__(
            self,
            fit: fitting.fit.FitGroup,
            **kwargs
    ):
        super().__init__(
            fit=fit,
            **kwargs
        )
        self._rda = kwargs.get('rda', np.array([100.0]))
        self._prda = kwargs.get('prda', np.array([100.0]))


