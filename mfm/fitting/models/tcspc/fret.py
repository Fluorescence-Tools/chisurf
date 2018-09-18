import numpy as np
import pandas as pd
from PyQt4 import QtCore, QtGui, uic

import mfm
import mfm.math
from mfm import plots, rda_axis
#from mfm.fitting.error_estimate import ErrorWidget
from mfm.fitting.fit import FittingControllerWidget
from mfm.fitting.models import ModelWidget
from mfm.fitting.models.tcspc.nusiance import GenericWidget, CorrectionsWidget
from mfm.fitting.models.tcspc.tcspc import LifetimeModel
from mfm.fluorescence import distribution2rates, rates2lifetimes
from mfm.fluorescence.anisotropy import AnisotropyWidget
from mfm.fluorescence.intensity import transfer_efficency2fdfa
from mfm.fluorescence.tcspc.convolve import ConvolveWidget
from mfm.parameter import FittingParameterWidget, FittingParameter, AggregatedParameters
from tcspc import Lifetime, LifetimeWidget

fret_settings = mfm.settings['fret']


class Gaussians(AggregatedParameters):

    name = "gaussians"

    @property
    def distribution(self):
        d = list()
        p = np.zeros_like(rda_axis)
        for i in range(len(self)):
            #p += self.amplitude[i] * scipy.stats.norm.pdf(rda_axis, loc=self.mean[i], scale=self.sigma[i])
            if not self.is_distance_between_gaussians:
                p += self.amplitude[i] * \
                     mfm.math.functions.distributions.generalized_normal_distribution(
                         rda_axis, self.mean[i], scale=self.sigma[i], shape=self.shape[i],
                         norm=True)
            else:
                p += mfm.math.functions.rdf.distance_between_gaussian(rda_axis, self.mean[i], self.sigma[i])
        p /= p.sum()
        threshold = max(p) * mfm.settings['tcspc']['threshold']
        p = np.where(p >= threshold, p, 0)
        d.append([p, rda_axis])
        d = np.array(d)
        return d

    @property
    def forster_radius(self):
        return self._R0.value

    @forster_radius.setter
    def forster_radius(self, v):
        self._R0.value = v

    @property
    def tau0(self):
        return self._t0.value

    @tau0.setter
    def tau0(self, v):
        self._t0.value = v

    @property
    def kappa2(self):
        return self._kappa2.value

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

    @property
    def donly(self):
        return np.sqrt(self._donly.value ** 2) # 1. / (1. + )

    @donly.setter
    def donly(self, v):
        self._donly.value = v

    def finalize(self):
        """
        This updates the values of the fitting parameters
        """
        # update amplitudes (sum of amplitudes is one)
        a = self.amplitude
        for i, g in enumerate(self._gaussianAmplitudes):
            g.value = a[i]
        # update means (only positive distances)
        a = self.mean
        for i, g in enumerate(self._gaussianMeans):
            g.value = a[i]
        #self._donly.value = self.donly

    def append(self, mean, sigma, x, shape=0.0):
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
        x = FittingParameter(name='x(%s,%i)' % (self.short, n + 1), value=x, lb=fret_settings['rda_min'],
                             ub=fret_settings['rda_max'])
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

    def __init__(self, donor_only=0.5, no_donly=False, **kwargs):
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
        kwargs['name'] = "Gaussians"
        AggregatedParameters.__init__(self, **kwargs)

        self.donors = Lifetime(**kwargs)
        self.no_donly = no_donly
        if no_donly:
            donor_only = 0.0

        self._gaussianMeans = []
        self._gaussianSigma = []
        self._gaussianShape = []
        self._gaussianAmplitudes = []
        self.short = kwargs.get('short', 'G')

        forster_radius = kwargs.pop('forster_radius', fret_settings['forster_radius'])
        kappa2 = kwargs.pop('kappa2', mfm.settings['fret']['kappa2'])
        t0 = kwargs.pop('tau0', mfm.settings['fret']['tau0'])

        model = self.model
        self._t0 = FittingParameter(name='t0', value=t0, fixed=True, model=model)
        self._R0 = FittingParameter(name='R0', value=forster_radius, fixed=True, model=model)
        self._kappa2 = FittingParameter(name='k2', value=kappa2, fixed=True, lb=0.0, ub=4.0, bounds_on=False, model=model)
        self._name = kwargs.get('name', 'gaussians')
        self._donly = FittingParameter(name='DOnly', value=donor_only, fixed=False, lb=0.0, ub=1.0, bounds_on=False,
                                       model=model)
        self.is_distance_between_gaussians = False # If this is True than the fitted distance is the distance between two Gaussians


class GaussianWidget(Gaussians, QtGui.QWidget):

    def __init__(self, donors, model=None, donly=0.2, no_donly=False, **kwargs):
        hide_donly = kwargs.get('hide_donly', False)

        Gaussians.__init__(self, donors=donors, donor_only=donly,
                           no_donly=no_donly, model=model, **kwargs)
        QtGui.QWidget.__init__(self)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)

        self.gb = QtGui.QGroupBox()
        self.layout.addWidget(self.gb)
        self.gb.setTitle("Gaussian distances")
        self.lh = QtGui.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setMargin(0)
        self.gb.setLayout(self.lh)

        lh = QtGui.QHBoxLayout()
        lh.setMargin(0)
        lh.setSpacing(0)
        self._R0 = self._R0.widget(text='R<sub>0</sub>', layout=lh)
        self._t0 = self._t0.widget(text='&tau;<sub>0</sub>', layout=lh)
        self.lh.addLayout(lh)

        lh = QtGui.QHBoxLayout()
        lh.setMargin(0)
        lh.setSpacing(0)
        self._kappa2 = self._kappa2.widget(text='&kappa;<sup>2</sup>', layout=lh)
        self._donly = self._donly.widget(text='x<sup>(D,0)</sup>', layout=lh)
        self._donly.setDisabled(self.no_donly)
        self.lh.addLayout(lh)

        if self.no_donly:
            self._donly.fixed = True
        if hide_donly:
            self._donly.hide()

        self._gb = list()

        self.gaus_grid_layout = QtGui.QGridLayout()

        l = QtGui.QHBoxLayout()
        addGaussian = QtGui.QPushButton()
        addGaussian.setText("add")
        l.addWidget(addGaussian)

        removeGaussian = QtGui.QPushButton()
        removeGaussian.setText("del")
        l.addWidget(removeGaussian)
        self.lh.addLayout(l)

        self.lh.addLayout(self.gaus_grid_layout)

        addGaussian.clicked.connect(self.onAddGaussian)
        removeGaussian.clicked.connect(self.onRemoveGaussian)

        # add some initial distance
        self.append(1.0, 50.0, 6.0, 0.0, False)

    def onAddGaussian(self):
        t = """
for f in cs.current_fit:
    f.model.%s.append()
            """ % self.name
        mfm.run(t)
        mfm.run("cs.current_fit.update()")

    def onRemoveGaussian(self):
        t = """
for f in cs.current_fit:
    f.model.%s.pop()
            """ % self.name
        mfm.run(t)
        mfm.run("cs.current_fit.update()")

    def append(self, x=None, mean=None, sigma=None, shape=0.0, update=True):
        x = 1.0 if x is None else x
        m = self._R0.value * 0.9 if mean is None else mean
        s = 6.0 if sigma is None else sigma
        gb = QtGui.QGroupBox()
        n_gauss = len(self)
        gb.setTitle('G%i' % (n_gauss + 1))
        l = QtGui.QVBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        m = FittingParameterWidget(name='R(%s,%i)' % (self.short, n_gauss + 1),
                                   value=m, layout=l, model=self.model, decimals=1,
                                   bounds_on=False,
                                   lb=fret_settings['rda_min'], ub=fret_settings['rda_max'],
                                   text='R', update_function=self.update)
        s = FittingParameterWidget(name='s(%s,%i)' % (self.short, n_gauss + 1), value=s, layout=l, model=self.model,
                                   decimals=1, fixed=True, bounds_on=False, lb=0.0, ub=40.0, hide_bounds=True,
                                   text='<b>&sigma;</b>', update_function=self.update)
        x = FittingParameterWidget(name='x(%s,%i)' % (self.short, n_gauss + 1), value=x, layout=l, model=self.model, decimals=3,
                                   bounds_on=False, text='x', update_function=self.update)
        shape = FittingParameterWidget(name='k(%s,%i)' % (self.short, n_gauss + 1), value=shape, layout=l, model=self.model,
                                       decimals=3, bounds_on=False, text='<b>&kappa;</b>',
                                       update_function=self.update, fixed=True)

        gb.setLayout(l)
        row = n_gauss / 2
        col = n_gauss % 2
        self.gaus_grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)
        self._gaussianMeans.append(m)
        self._gaussianSigma.append(s)
        self._gaussianShape.append(shape)
        self._gaussianAmplitudes.append(x)
        self.update()

    def pop(self):
        self._gaussianMeans.pop().close()
        self._gaussianSigma.pop().close()
        self._gaussianAmplitudes.pop().close()
        self._gaussianShape.pop().close()
        self._gb.pop().close()
        self.update()


class FRETModel(LifetimeModel):

    @property
    def forster_radius(self):
        """
        The Forster-radius of the FRET-pair
        """
        return self._forster_radius.value

    @forster_radius.setter
    def forster_radius(self, v):
        """
        The Forster-radius of the FRET-pair
        """
        self._forster_radius.value = v

    @property
    def tau0(self):
        """
        The lifetime of the donor in absence of additional quenching
        """
        return self._tau0.value

    @tau0.setter
    def tau0(self, v):
        """
        The lifetime of the donor in absence of additional quenching
        """
        self._tau0.value = v

    @property
    def kappa2(self):
        """
        The mean orientation factor
        """
        return self._kappa2.value

    @kappa2.setter
    def kappa2(self, v):
        """
        The mean orientation factor
        """
        self._kappa2.value = v

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
    def donly(self):
        """
        The fractions of donor-only (No-FRET) species. By default no donor-only is assumed. This has to be
        implemented by the model anyway.
        """
        return self._donly.value

    @donly.setter
    def donly(self, v):
        """
        The fractions of donor-only (No-FRET) species. By default no donor-only is assumed. This has to be
        implemented by the model anyway.
        """
        self._donly.value = v

    @property
    def fret_rate_spectrum(self):
        """
        The FRET-rate spectrum. This takes the distance distribution of the model and calculated the resulting
        FRET-rate spectrum (excluding the donor-offset).
        """
        rs = distribution2rates(self.distance_distribution, self.tau0, self.kappa2, self.forster_radius)
        r = np.hstack(rs).ravel([-1])
        return r

    @property
    def lifetime_spectrum(self):
        """
        Slightly slower than new version
        This returns the lifetime-spectrum of the model including the donor-only offset (donor-only fraction)
        """
        lt = rates2lifetimes(self.fret_rate_spectrum, self.donors.rate_spectrum, self.donly)
        #lt = rates2lifetimes(self.fret_rate_spectrum, self.donors.lifetime_spectrum, self.donly)
        if mfm.settings['fret']['bin_lifetime']:
            n_lifetimes = mfm.settings['fret']['lifetime_bins']
            discriminate = mfm.settings['fret']['discriminate']
            discriminate_amplitude = mfm.settings['fret']['discriminate_amplitude']
            return mfm.fluorescence.tcspc.bin_lifetime_spectrum(lt, n_lifetimes=n_lifetimes,
                                                                discriminate=discriminate,
                                                                discriminator=discriminate_amplitude
                                                                )
        else:
            return lt

    @property
    def donor_lifetime_spectrum(self):
        """
        The donor lifetime spectrum in form amplitude, lifetime, amplitude, lifetime
        """
        return self.donors.lifetime_spectrum

    @donor_lifetime_spectrum.setter
    def donor_lifetime_spectrum(self, v):
        self.model.donors.lifetime_spectrum = v

    @property
    def donor_species_averaged_lifetime(self):
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self.donors.species_averaged_lifetime

    @property
    def donor_fluorescence_averaged_lifetime(self):
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self.donors.fluorescence_averaged_lifetime

    @property
    def fret_species_averaged_lifetime(self):
        """
        The current species averages lifetime of the FRET sample xi * taui
        """
        return self.species_averaged_lifetime

    @property
    def fret_fluorescence_averaged_lifetime(self):
        """
        The current fluorescence averaged lifetime of the FRET-sample = xi*taui**2 / species_averaged_lifetime
        """
        return self.fluorescence_averaged_lifetime

    @property
    def transfer_efficiency(self):
        """
        The current transfer efficency of the model (this includes donor-only)
        """
        if self.convolve._n0.fixed:
            return 1.0 - self.fret_species_averaged_lifetime / self.donor_species_averaged_lifetime
        else:
            return 1.0 - self.fit.data.y.sum() / self.reference.sum()

    @transfer_efficiency.setter
    def transfer_efficiency(self, v):
        sdecay = self.fit.data.y.sum()
        tau0x = self.donor_species_averaged_lifetime
        n0 = sdecay/(tau0x*(1.-v))
        self.convolve.n0 = n0

    @property
    def donors(self):
        return self._donors

    @donors.setter
    def donors(self, v):
        self._donors = v

    @property
    def reference(self):
        self._reference.update_model()
        return np.maximum(self._reference._y, 0)

    def get_transfer_efficency(self, phiD, phiA):
        """ Get the current donor-acceptor fluorescence intensity ratio
        :param phiD: float
            donor quantum yield
        :param phiA:
            acceptor quantum yield
        :return: float, transfer efficency
        """
        return transfer_efficency2fdfa(self.transfer_efficiency, phiD, phiA)

    def update_model(self, **kwargs):
        LifetimeModel.update_model(self, **kwargs)
        verbose = kwargs.get('verbose', mfm.verbose)
        if verbose:
            print "Transfer-efficency: %s " % self.transfer_efficiency
            print "FRET tauX: %s " % self.fret_species_averaged_lifetime
            print "FRET tauF: %s " % self.fret_fluorescence_averaged_lifetime
            print "Donor tauX: %s " % self.donor_species_averaged_lifetime
            print "Donor tauF: %s " % self.donor_fluorescence_averaged_lifetime

    def __init__(self, fit, **kwargs):
        LifetimeModel.__init__(self, fit, autoscale=False, **kwargs)

        self._forster_radius = FittingParameter(value=fret_settings['forster_radius'], name="R0", fixed=True)
        self._tau0 = FittingParameter(value=fret_settings['tau0'], name="tau0", fixed=True)
        self._kappa2 = FittingParameter(value=fret_settings['kappa2'], name="kappa2", fixed=True)
        self._donly = FittingParameter(value=0.2, name="donly", lb=0.0, ub=1.0, bounds_on=True)

        self._donors = kwargs.get('lifetimes', Lifetime())

        self._reference = LifetimeModel(fit, **kwargs)
        self._reference.lifetimes = self.donors
        self._reference.convolve = self.convolve


class GaussianModel(FRETModel):
    """
    This fit model is uses multiple Gaussian/normal distributions to fit the FRET-decay. Here the donor lifetime-
    spectrum as well as the distances may be fitted. In this model it is assumed that each donor-species is fitted
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
    def kappa2(self):
        return self.gaussians.kappa2

    @property
    def distance_distribution(self):
        dist = self.gaussians.distribution
        return dist

    def append(self, mean, sigma, species_fraction):
        self.gaussians.append(mean, sigma, species_fraction)

    def pop(self):
        return self.gaussians.pop()

    def finalize(self):
        super(FRETModel, self).finalize()
        self.gaussians.finalize()
        #self.gaussians.update()

    def __init__(self, fit, **kwargs):
        FRETModel.__init__(self, fit, **kwargs)
        self.gaussians = kwargs.get('gaussians', Gaussians(**kwargs))
        self._donly = self.gaussians._donly
        self._tau0 = self.gaussians._t0
        self._forster_radius = self.gaussians._R0


class GaussianModelWidget(GaussianModel, ModelWidget):

    plot_classes = [
                       (plots.LinePlot, {'d_scalex': 'lin', 'd_scaley': 'log', 'r_scalex': 'lin', 'r_scaley': 'lin',
                                         'x_label': 'x', 'y_label': 'y', 'plot_irf': True}),
                       (plots.FitInfo, {}), (plots.DistributionPlot, {}), (plots.ParameterScanPlot, {})
                    ]

    def __init__(self, fit, **kwargs):
        ModelWidget.__init__(self, fit, icon=QtGui.QIcon(":/icons/icons/TCSPC.png"), **kwargs)

        convolve = ConvolveWidget(fit=fit, model=self, **kwargs)
        donors = LifetimeWidget(parent=self, model=self, title='Donor(0)')
        gaussians = GaussianWidget(donors=donors, parent=self, model=self, short='G', **kwargs)
        anisotropy = AnisotropyWidget(model=self, short='rL', **kwargs)
        generic = GenericWidget(fit=fit, model=self, parent=self, **kwargs)
        corrections = CorrectionsWidget(fit=fit, model=self, **kwargs)

        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout = layout

        self.layout.addWidget(convolve)
        self.layout.addWidget(generic)
        self.layout.addWidget(donors)
        self.layout.addWidget(gaussians)

        self.layout.addWidget(anisotropy)
        self.layout.addWidget(corrections)

        GaussianModel.__init__(self,
                               fit=fit,
                               convolve=convolve,
                               generic=generic,
                               lifetimes=donors,
                               gaussians=gaussians,
                               corrections=corrections,
                               anisotropy=anisotropy)


class FretRate(AggregatedParameters):

    @property
    def distribution(self):
        """
        a = np.array(self.amplitude)
        a /= sum(a)
        d = np.array(self.distances)
        n_rates = len(self)
        d = np.vstack([a, d]).reshape([1, 2, n_rates])
        """
        d = list()
        px = np.array(self.distances)
        py = np.array(self.amplitude)
        p = mfm.math.functions.distributions.linear_dist(rda_axis, px, py)
        d.append([p, rda_axis])
        d = np.array(d)

        return d

    @property
    def R0(self):
        return self._R0.value

    @R0.setter
    def R0(self, v):
        self._R0.value = v

    @property
    def tau0(self):
        return self._t0.value

    @tau0.setter
    def tau0(self, v):
        self._t0.value = v

    @property
    def kappa2(self):
        return self._kappa2.value

    @kappa2.setter
    def kappa2(self, v):
        self._kappa2.value = v

    @property
    def distances(self):
        try:
            a = np.sqrt(np.array([g.value for g in self._distances]) ** 2)
            for i, g in enumerate(self._distances):
                g.value = a[i]
            return a
        except AttributeError:
            return np.array([])

    @property
    def amplitude(self):
        try:
            a = np.sqrt(np.array([g.value for g in self._amplitudes]) ** 2)
            a /= a.sum()
            for i, g in enumerate(self._amplitudes):
                g.value = a[i]
            return a
        except AttributeError:
            return np.array([])

    @property
    def donly(self):
        return np.sqrt(self._donly.value ** 2)

    @donly.setter
    def donly(self, v):
        self._donly.value = v

    def append(self, distance, x):
        """
        Adds/appends a new FRET-rate

        :param distance: float
            Mean of the new normal distribution
        :param x: float
            Amplitude of the normal distribution
        """
        n = len(self)
        m = FittingParameter(name='R(%s,%i)' % (self.short, n + 1), value=distance)
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

    def __init__(self, no_donly=False, **kwargs):
        """
        This class keeps the necessary parameters to perform a fit with single (discrete) FRET-rates in form of
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
        AggregatedParameters.__init__(self, **kwargs)

        forster_radius = kwargs.get('forster_radius', mfm.settings['fret']['forster_radius'])
        kappa2 = kwargs.get('kappa2', mfm.settings['fret']['kappa2'])
        t0 = kwargs.get('tau0', mfm.settings['fret']['tau0'])

        self.donors = Lifetime(**kwargs)
        self._name = kwargs.get('name', 'fret_rate')
        self.no_donly = kwargs.get('no_donly', False)

        self._distances = []
        self._amplitudes = []
        self.short = 'F'

        self._t0 = FittingParameter(name='t0', value=t0, fixed=True)
        self._R0 = FittingParameter(name='R0', value=forster_radius, fixed=True)
        self._kappa2 = FittingParameter(name='k2', value=kappa2, fixed=True, lb=0.0, ub=4.0, bounds_on=False)
        if not no_donly:
            self._donly = self._donly.widget()


class FretRateWidget(FretRate, QtGui.QWidget):

    def __init__(self, donors, model=None, short='F', **kwargs):
        hide_donly = kwargs.get('hide_donly', False)

        self.model = model
        self.short = short
        FretRate.__init__(self, donors=donors)
        QtGui.QWidget.__init__(self)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)

        lh = QtGui.QHBoxLayout()
        lh.setMargin(0)
        lh.setSpacing(0)
        self._R0 = self._R0.widget(layout=lh, text='R<sub>0</sub>')
        self._t0 = self._t0.widget(layout=lh, text='&tau;<sub>0</sub>')
        self.layout.addLayout(lh)

        lh = QtGui.QHBoxLayout()
        lh.setMargin(0)
        lh.setSpacing(0)
        self._kappa2 = self._kappa2.widget(layout=lh, text='&kappa;<sup>2</sup>')
        self.layout.addLayout(lh)

        self.gb = QtGui.QGroupBox()
        self.layout.addWidget(self.gb)
        self.gb.setTitle("FRET-rates")
        self.lh = QtGui.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setMargin(0)
        self.gb.setLayout(self.lh)

        l = QtGui.QHBoxLayout()

        add_FretRate = QtGui.QPushButton()
        add_FretRate.setText("add")
        l.addWidget(add_FretRate)

        remove_FRETrate = QtGui.QPushButton()
        remove_FRETrate.setText("del")
        l.addWidget(remove_FRETrate)

        spacerItem = QtGui.QSpacerItem(20, 0, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        l.addItem(spacerItem)

        lh.addLayout(l)

        l = QtGui.QHBoxLayout()
        self._donly = self._donly.widget(layout=l, text='x<sup>(D,0)</sup>')
        self._donly.setDisabled(self.no_donly)
        if hide_donly:
            self._donly.hide()

        self.lh.addLayout(l)
        self._gb = list()

        self.fret_grid_layout = QtGui.QGridLayout()
        self.lh.addLayout(self.fret_grid_layout)

        self.connect(add_FretRate, QtCore.SIGNAL("clicked()"), self.onAddRate)
        self.connect(remove_FRETrate, QtCore.SIGNAL("clicked()"), self.onRemoveRate)
        # add some initial distance
        self.append(1.0, 50.0, False)

    def update(self, *__args):
        QtGui.QWidget.update(self, *__args)
        self.model.update()
        #self.model.updatePlots()

    def onAddRate(self):
        t = "for f in cs.current_fit:\n" \
            "   f.model.%s.append()" % self.name
        mfm.run(t)
        mfm.run("cs.current_fit.update()")

    def onRemoveRate(self):
        t = "for f in cs.current_fit:\n" \
            "   f.model.%s.pop()" % self.name
        mfm.run(t)
        mfm.run("cs.current_fit.update()")

    def append(self, x=None, mean=None, update=True):
        x = 1.0 if x is None else x
        m = np.random.normal(self._R0.value, self._R0.value * 0.6, 1)[0] if mean is None else mean
        gb = QtGui.QGroupBox()
        n_fret = len(self)
        gb.setTitle('G%i' % (n_fret + 1))
        l = QtGui.QVBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        m = FittingParameterWidget(name='R(%s,%i)' % (self.short, n_fret + 1), value=m, layout=l, model=self.model, decimals=1,
                                   text='R', update_function=self.update)
        x = FittingParameterWidget(name='x(%s,%i)' % (self.short, n_fret + 1), value=x, layout=l, model=self.model, decimals=2,
                                   bounds_on=False, text='x', update_function=self.update)
        gb.setLayout(l)
        row = n_fret / 2
        col = n_fret % 2
        self.fret_grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)

        self._distances.append(m)
        self._amplitudes.append(x)

        if update:
            self.update()

    def pop(self):
        self._amplitudes.pop().close()
        self._distances.pop().close()
        self._gb.pop().close()
        self.update()


class FRETrateModel(FRETModel):
    """
    This fit model is uses multiple discrete FRET rates to fit the Donor-decay. Here the donor lifetime-
    spectrum as well as the distances may be fitted. In this model it is assumed that each donor-species is fitted
    by the same FRET-rate distribution.

    References
    ----------

    .. [1]  Kalinin, S., and Johansson, L.B., Energy Migration and Transfer Rates
            are Invariant to Modeling the Fluorescence Relaxation by Discrete and Continuous
            Distributions of Lifetimes.
            J. Phys. Chem. B, 108 (2004) 3092-3097.

    """

    name = "FRET: FD (Discrete)"

    @property
    def forster_radius(self):
        return self.fret_rates.R0

    @forster_radius.setter
    def forster_radius(self, v):
        self.fret_rates.R0 = v

    @property
    def tau0(self):
        return self.fret_rates.tau0

    @tau0.setter
    def tau0(self, v):
        self.fret_rates.tau0 = v

    @property
    def kappa2(self):
        return self.fret_rates.kappa2

    @kappa2.setter
    def kappa2(self, v):
        self.fret_rates.kappa2 = v

    @property
    def donly(self):
        return self.fret_rates.donly

    @donly.setter
    def donly(self, v):
        self.fret_rates.donly = v

    @property
    def distance_distribution(self):
        dist = self.fret_rates.distribution
        return dist

    def append(self, mean, species_fraction):
        self.fret_rates.append(mean, species_fraction)

    def pop(self):
        return self.fret_rates.pop()

    def __init__(self, fit, **kwargs):
        FRETModel.__init__(self, fit, **kwargs)
        self.fret_rates = kwargs.get('fret_rates', FretRate(**kwargs))


class FRETrateModelWidget(FRETrateModel, ModelWidget):

    plot_classes = [
                       (plots.LinePlot, {'d_scalex': 'lin', 'd_scaley': 'log', 'r_scalex': 'lin', 'r_scaley': 'lin',
                                         'x_label': 'x', 'y_label': 'y', 'plot_irf': True}),
                       (plots.FitInfo, {}), (plots.DistributionPlot, {}, (plots.ParameterScanPlot, {}))
                    ]

    def __init__(self, fit, **kwargs):
        ModelWidget.__init__(self, fit=fit, icon=QtGui.QIcon(":/icons/icons/TCSPC.png"), **kwargs)

        convolve = ConvolveWidget(fit=fit, model=self, **kwargs)
        donors = LifetimeWidget(parent=self, model=self, title='Donor(0)', name='donors')
        fret_rates = FretRateWidget(donors=donors, parent=self, model=self, short='G', name='fret_rates', **kwargs)
        anisotropy = AnisotropyWidget(model=self, short='rL', **kwargs)
        generic = GenericWidget(fit=fit, model=self, parent=self, **kwargs)
        corrections = CorrectionsWidget(fit, model=self, **kwargs)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.layout.addWidget(convolve)
        self.layout.addWidget(generic)
        self.layout.addWidget(donors)
        self.layout.addWidget(fret_rates)

        self.layout.addWidget(anisotropy)
        self.layout.addWidget(corrections)

        FRETrateModel.__init__(self, fit=fit, fret_rates=fret_rates)
        self.convolve = convolve
        self.donors = donors
        self.fret_rates = fret_rates
        self.anisotropy = anisotropy
        self.generic = generic
        self.corrections = corrections


class WormLikeChainModel(FRETModel):

    name = "FD(A): Worm-like chain"

    @property
    def distance_distribution(self):
        kappa = self.persistence_length / self.chain_length
        if not self.use_dye_linker:
            prob = mfm.math.functions.rdf.worm_like_chain(rda_axis, kappa, self.chain_length)
        else:
            prob = mfm.math.functions.rdf.worm_like_chain_linker(rda_axis, kappa,
                                                                 self.chain_length,
                                                                 self.sigma_linker)
        dist = np.array([prob, rda_axis]).reshape([1, 2, fret_settings['rda_resolution']])
        return dist

    @property
    def donly(self):
        return self._donly.value

    @property
    def chain_length(self):
        return self._chain_length.value

    @property
    def persistence_length(self):
        return self._persistence_length.value

    @property
    def forster_radius(self):
        return self._R0.value

    @property
    def tau0(self):
        return self._tau0.value

    @property
    def sigma_linker(self):
        return self._sigma_linker.value

    @sigma_linker.setter
    def sigma_linker(self, v):
        self._sigma_linker.value = v

    def __init__(self, fit, **kwargs):
        self._R0 = FittingParameter(value=52.0, name="R0", fixed=True)
        self._tau0 = FittingParameter(value=4.0, name="tau0", fixed=True)
        self._donly = FittingParameter(value=0.0, name="donly")
        FRETModel.__init__(self, fit, **kwargs)
        self._chain_length = FittingParameter(name='length', value=100.0, model=self, decimals=1, fixed=False, text='l')
        self.use_dye_linker = False
        self._sigma_linker = FittingParameter(name='link_width', value=6.0, model=self, decimals=1, fixed=False, text='lw')


class WormLikeChainModelWidget(WormLikeChainModel, ModelWidget):

    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                  'd_scaley': 'log',
                                                  'r_scalex': 'lin',
                                                  'r_scaley': 'lin',
                                                  'x_label': 'x',
                                                  'y_label': 'y',
                                                  'plot_irf': True}
                     )
        ,(plots.FitInfo, {})
                    , (plots.DistributionPlot, {})
        #,(plots.SurfacePlot, {})
    ]

    @property
    def use_dye_linker(self):
        return bool(self._use_dye_linker.isChecked())

    @use_dye_linker.setter
    def use_dye_linker(self, v):
        self._use_dye_linker.setChecked(v)

    def __init__(self, fit, **kwargs):
        ModelWidget.__init__(self, fit, icon=QtGui.QIcon(":/icons/icons/TCSPC.png"), **kwargs)
        convolve = ConvolveWidget(fit=fit, model=self, hide_curve_convolution=True, **kwargs)
        donors = LifetimeWidget(parent=self, model=self, title='Donor(0)', name='donors')
        generic = GenericWidget(fit=fit, model=self, parent=self, **kwargs)
        self._use_dye_linker = QtGui.QCheckBox()
        self._use_dye_linker.setText('Use linker')

        corrections = CorrectionsWidget(fit, model=self, **kwargs)
        WormLikeChainModel.__init__(self, fit=fit, lifetimes=donors, generic=generic,
                                    corrections=corrections, convolve=convolve)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.layout.addWidget(convolve)
        self.layout.addWidget(generic)
        self.layout.addWidget(donors)

        self.layout.addWidget(self._sigma_linker.widget())
        self.layout.addWidget(self._use_dye_linker)

        self._donly = self._donly.widget(text='dOnly', layout=self.layout)
        self._chain_length = self._chain_length.widget(layout=self.layout)

        self._persistence_length = FittingParameterWidget(name='persistence',
                                                          value=6.0, layout=self.layout,
                                                          model=self, decimals=4,
                                                          fixed=False, text='lp')

        self._R0 = FittingParameterWidget(name='R0 [A]', value=52.0,
                                          layout=self.layout,
                                          model=self, decimals=1, fixed=True,
                                          text='R0[A]')
        self._tau0 = FittingParameterWidget(name='tau0', value=4.1, layout=self.layout,
                                            model=self, decimals=2, fixed=True,
                                            text='tau0')

        self.layout.addWidget(self._chain_length)
        self.layout.addWidget(self._persistence_length)
        self.layout.addWidget(self._R0)
        self.layout.addWidget(self._tau0)
        self.layout.addWidget(corrections)


class SingleDistanceModel(FRETModel):

    name = "Fixed distance distribution"

    @property
    def donly(self):
        return self._donly.value

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
        self.fitting_widget = QtGui.QLabel() if kwargs.get('disable_fit', False) else FittingControllerWidget(fit=fit, **kwargs)
        #self.errors = ErrorWidget(fit, **kwargs)
        self.corrections = CorrectionsWidget(fit, model=self, **kwargs)

        ModelWidget.__init__(self, fit=fit, icon=QtGui.QIcon(":/icons/icons/TCSPC.png"), **kwargs)

        SingleDistanceModel.__init__(self, fit=fit, convolve=self.convolve, corrections=self.corrections,
                                     generic=self.generic, lifetimes=self.donors, anisotropy=self.anisotropy)

        self._donly = self._donly.widget()

        uic.loadUi('mfm/ui/fitting/models/tcspc/load_distance_distibution.ui', self)
        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")
        self.connect(self.actionOpen_distirbution, QtCore.SIGNAL('triggered()'), self.load_distance_distribution)

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
        print "load_distance_distribution"
        verbose = kwargs.get('verbose', self.verbose)
        #filename = kwargs.get('filename', str(QtGui.QFileDialog.getOpenFileName(self, 'Open File')))
        filename = mfm.widgets.open_file('Open distance distribution', 'CSV-files (*.csv)')
        self.lineEdit.setText(filename)
        ar = np.array(pd.read_csv(filename, sep='\t')).T
        if verbose:
            print "Opening distribution"
            print "Filename: %s" % filename
            print "Shape: %s" % ar.shape
        self.rda = ar[0]
        self.prda = ar[1]
        self.update_model()