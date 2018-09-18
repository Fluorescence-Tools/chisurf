import math

import numpy as np
from PyQt4 import QtGui, QtCore

import mfm
import mfm.fluorescence.tcspc
import mfm.math.signal
from mfm.fitting.models import Model, ModelWidget, ModelCurve
from mfm.fitting.models.tcspc.nusiance import Generic, GenericWidget, Corrections, CorrectionsWidget
from mfm.fluorescence.anisotropy import Anisotropy, AnisotropyWidget
from mfm.fluorescence.general import species_averaged_lifetime, fluorescence_averaged_lifetime
from mfm.fluorescence.tcspc.convolve import Convolve, ConvolveWidget
from mfm.parameter import FittingParameterWidget, FittingParameter, AggregatedParameters


class Lifetime(AggregatedParameters):

    @property
    def absolute_amplitudes(self):
        return self._abs_amplitudes

    @absolute_amplitudes.setter
    def absolute_amplitudes(self, v):
        self._abs_amplitudes = v

    @property
    def normalize_amplitudes(self):
        return self._normalize_amplitudes

    @normalize_amplitudes.setter
    def normalize_amplitudes(self, v):
        self._normalize_amplitudes = v

    @property
    def species_averaged_lifetime(self):
        decay = np.empty(2 * len(self), dtype=np.float64)
        a = self.amplitudes
        a /= a.sum()
        decay[0::2] = a
        decay[1::2] = self.lifetimes
        return species_averaged_lifetime(decay)

    @property
    def fluorescence_averaged_lifetime(self):
        decay = np.empty(2 * len(self), dtype=np.float64)
        a = self.amplitudes
        a /= a.sum()
        decay[0::2] = a
        decay[1::2] = self.lifetimes
        return fluorescence_averaged_lifetime(decay)

    @property
    def amplitudes(self):
        vs = np.array([x.value for x in self._amplitudes])
        if self.absolute_amplitudes:
            vs = np.sqrt(vs**2)
        if self.normalize_amplitudes:
            vs /= vs.sum()
        return vs

    @amplitudes.setter
    def amplitudes(self, vs):
        for i, v in enumerate(vs):
            self._amplitudes[i].value = v

    @property
    def lifetimes(self):
        vs = np.array([math.sqrt(x.value ** 2) for x in self._lifetimes])
        for i, v in enumerate(vs):
            self._lifetimes[i].value = v
        return vs

    @lifetimes.setter
    def lifetimes(self, vs):
        for i, v in enumerate(vs):
            self._lifetimes[i].value = v

    @property
    def lifetime_spectrum(self):
        if self._link is None:
            if self._lifetime_spectrum is None:
                decay = np.empty(2 * len(self), dtype=np.float64)
                decay[0::2] = self.amplitudes
                decay[1::2] = self.lifetimes
                return decay
            else:
                return self._lifetime_spectrum
        else:
            return self._link.lifetime_spectrum

    @lifetime_spectrum.setter
    def lifetime_spectrum(self, v):
        self._lifetime_spectrum = v
        for p in self.parameters_all:
            p.fixed = True

    @property
    def rate_spectrum(self):
        return mfm.fluorescence.general.ilt(self.lifetime_spectrum)

    @property
    def n(self):
        return len(self._amplitudes)

    @property
    def link(self):
        return self._link

    @link.setter
    def link(self, v):
        if isinstance(v, Lifetime) or v is None:
            self._link = v

    def update(self):
        amplitudes = self.amplitudes
        for i, a in enumerate(self._amplitudes):
            a.value = amplitudes[i]

    def finalize(self):
        self.update()

    def append(self, amplitude, lifetime, **kwargs):
        lower_bound_amplitude = kwargs.get('lower_bound_amplitude', None)
        upper_bound_amplitude = kwargs.get('upper_bound_amplitude', None)
        fixed = kwargs.get('upper_bound_amplitude', False)
        bound_on = kwargs.get('bound_on', False)
        lower_bound_lifetime = kwargs.get('lower_bound_lifetime', None)
        upper_bound_lifetime = kwargs.get('upper_bound_lifetime', None)

        n = len(self)
        a = FittingParameter(lb=lower_bound_amplitude, ub=upper_bound_amplitude,
                             value=amplitude, name='x%s%i' % (self.short, n + 1),
                             fixed=fixed, bounds_on=bound_on)
        t = FittingParameter(lb=lower_bound_lifetime, ub=upper_bound_lifetime,
                             value=lifetime, name='t%s%i' % (self.short, n + 1),
                             fixed=fixed, bounds_on=bound_on)
        self._amplitudes.append(a)
        self._lifetimes.append(t)

    def pop(self):
        a = self._amplitudes.pop()
        l = self._lifetimes.pop()
        return a, l

    def __init__(self, **kwargs):
        AggregatedParameters.__init__(self, **kwargs)
        self.short = kwargs.get('short', 'L')
        self._lifetime_spectrum = None
        self._abs_amplitudes = kwargs.get('absolute_amplitudes', True)
        self._normalize_amplitudes = kwargs.get('normalize_amplitudes', True)
        self._amplitudes = kwargs.get('amplitudes', [])
        self._lifetimes = kwargs.get('lifetimes', [])
        self._name = kwargs.get('name', 'lifetimes')
        self._link = kwargs.get('link', None)

    def __len__(self):
        return self.n


class LifetimeWidget(Lifetime, QtGui.QWidget):

    def update(self, *__args):
        QtGui.QWidget.update(self, *__args)
        Lifetime.update(self)

    def read_values(self, target):
        def linkcall():
            for key in self.parameter_dict:
                v = target.parameters_all_dict[key].value
                mfm.run("cs.current_fit.model.parameters_all_dict['%s'].value = %s" % (key, v))
            mfm.run("cs.current_fit.update()")
        return linkcall

    def read_menu(self):
        menu = QtGui.QMenu()
        for f in mfm.fits:
            for fs in f:
                submenu = QtGui.QMenu(menu)
                submenu.setTitle(fs.name)
                for a in fs.model.aggregated_parameters:
                    if isinstance(a, LifetimeWidget):
                        Action = submenu.addAction(a.name)
                        Action.triggered.connect(self.read_values(a))
                menu.addMenu(submenu)
        self.readFrom.setMenu(menu)
        #menu.exec_(event.globalPos())

    def link_values(self, target):
        def linkcall():
            self._link = target
            self.setEnabled(False)
        return linkcall

    def link_menu(self):
        menu = QtGui.QMenu()
        for f in mfm.fits:
            for fs in f:
                submenu = QtGui.QMenu(menu)
                submenu.setTitle(fs.name)
                for a in fs.model.aggregated_parameters:
                    if isinstance(a, LifetimeWidget):
                        Action = submenu.addAction(a.name)
                        Action.triggered.connect(self.link_values(a))
                menu.addMenu(submenu)
        self.linkFrom.setMenu(menu)
        #menu.exec_(event.globalPos())

    def __init__(self, title='', **kwargs):
        QtGui.QWidget.__init__(self)
        Lifetime.__init__(self, **kwargs)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)

        self.gb = QtGui.QGroupBox()
        self.gb.setTitle(title)
        self.lh = QtGui.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setMargin(0)
        self.gb.setLayout(self.lh)
        self.layout.addWidget(self.gb)

        lh = QtGui.QHBoxLayout()
        lh.setSpacing(0)
        lh.setMargin(0)

        addDonor = QtGui.QPushButton()
        addDonor.setText("add")
        addDonor.clicked.connect(self.onAddLifetime)
        lh.addWidget(addDonor)

        removeDonor = QtGui.QPushButton()
        removeDonor.setText("del")
        removeDonor.clicked.connect(self.onRemoveLifetime)
        lh.addWidget(removeDonor)

        spacerItem = QtGui.QSpacerItem(20, 0, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        lh.addItem(spacerItem)

        readFrom = QtGui.QToolButton()
        readFrom.setText("read")
        readFrom.setPopupMode(QtGui.QToolButton.InstantPopup)
        readFrom.clicked.connect(self.read_menu)
        lh.addWidget(readFrom)
        self.readFrom = readFrom

        linkFrom = QtGui.QToolButton()
        linkFrom.setText("link")
        linkFrom.setPopupMode(QtGui.QToolButton.InstantPopup)
        linkFrom.clicked.connect(self.link_menu)
        lh.addWidget(linkFrom)
        self.linkFrom = linkFrom

        normalize_amplitude = QtGui.QCheckBox("Norm.")
        normalize_amplitude.setChecked(True)
        normalize_amplitude.setToolTip("Normalize amplitudes to unity.\nThe sum of all amplitudes equals one.")
        normalize_amplitude.clicked.connect(self.onNormalizeAmplitudes)
        self.normalize_amplitude = normalize_amplitude

        absolute_amplitude = QtGui.QCheckBox("Abs.")
        absolute_amplitude.setChecked(True)
        absolute_amplitude.setToolTip("Take absolute value of amplitudes\nNo negative amplitudes")
        absolute_amplitude.clicked.connect(self.onAbsoluteAmplitudes)
        self.absolute_amplitude = absolute_amplitude

        lh.addWidget(absolute_amplitude)
        lh.addWidget(normalize_amplitude)
        self.lh.addLayout(lh)

        self.append()

    #def update_link_menu(self, *__args):

    def onNormalizeAmplitudes(self):
        norm_amp = self.normalize_amplitude.isChecked()
        mfm.run("cs.current_fit.model.lifetimes.normalize_amplitudes = %s" % norm_amp)
        mfm.run("cs.current_fit.update()")

    def onAbsoluteAmplitudes(self):
        abs_amp = self.absolute_amplitude.isChecked()
        mfm.run("cs.current_fit.model.lifetimes.absolute_amplitudes = %s" % abs_amp)
        mfm.run("cs.current_fit.update()")

    def onAddLifetime(self):
        t = "for f in cs.current_fit:\n" \
            "   f.model.%s.append()\n" \
            "cs.current_fit.update()" % self.name
        mfm.run(t)

    def onRemoveLifetime(self):
        t = "for f in cs.current_fit:\n" \
            "   f.model.%s.pop()" % self.name
        mfm.run(t)

    def append(self, x=1.0, l=None, **kwargs):
        try:
            fwhm = self.fit.model.convolve.fwhm
        except AttributeError:
            fwhm = 1.0
        #fwhm = self.model.convolve.fwhm
        lt = fwhm * 4.0

        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)

        n = len(self)
        self._amplitudes.append(
            FittingParameterWidget(name='x%s%i' % (self.short, n + 1), value=x, layout=l, model=self.model,
                                   text='x(%s,%i)' % (self.short, n + 1))
        )
        self._lifetimes.append(
            FittingParameterWidget(name='t%s%i' % (self.short, n + 1), value=lt, layout=l, model=self.model,
                                   text='<b>&tau;</b>(%s,%i)' % (self.short, n + 1))
        )
        self.lh.addLayout(l)

    def pop(self):
        self._amplitudes.pop().close()
        self._lifetimes.pop().close()
        self.model.update()


class LifetimeModel(ModelCurve):

    name = "Lifetime fit"

    def __str__(self):
        s = Model.__str__(self)
        s += "\n------------------\n"
        s += "\nAverage Lifetimes:\n"
        s += "<tau>x: %.3f\n<tau>F: %.3f\n" % (self.species_averaged_lifetime, self.fluorescence_averaged_lifetime)
        s += "\n------------------\n"
        return s

    def __init__(self, fit, **kwargs):
        ModelCurve.__init__(self, fit, **kwargs)
        self.generic = kwargs.get('generic', Generic(name='generic', fit=fit, **kwargs))
        self.corrections = kwargs.get('corrections', Corrections(name='corrections', fit=fit, model=self, **kwargs))
        self.anisotropy = kwargs.get('anisotropy', Anisotropy(name='anisotropy', **kwargs))
        self.lifetimes = kwargs.get('lifetimes', Lifetime(name='lifetimes', model=self, fit=fit, **kwargs))
        self.convolve = kwargs.get('convolve', Convolve(name='convolve', fit=fit, **kwargs))

    @property
    def species_averaged_lifetime(self):
        return species_averaged_lifetime(self.lifetime_spectrum)

    @property
    def var_lifetime(self):
        lx = self.species_averaged_lifetime
        lf = self.fluorescence_averaged_lifetime
        return lx*(lf-lx)

    @property
    def fluorescence_averaged_lifetime(self):
        return fluorescence_averaged_lifetime(self.lifetime_spectrum, self.species_averaged_lifetime)

    @property
    def lifetime_spectrum(self):
        return self.lifetimes.lifetime_spectrum

    def finalize(self):
        Model.finalize(self)
        self.lifetimes.update()

    def decay(self, time):
        x, l = self.lifetime_spectrum.reshape((self.lifetime_spectrum.shape[0]/2), 2).T
        f = np.array([np.dot(x, np.exp(- t / l)) for t in time])
        return f

    def update_model(self, **kwargs):
        verbose = kwargs.get('verbose', mfm.verbose)
        lifetime_spectrum = kwargs.get('lifetime_spectrum', self.lifetime_spectrum)
        scatter = kwargs.get('scatter', self.generic.scatter)
        background = kwargs.get('background', self.generic.background)
        shift_bg_with_irf = kwargs.get('shift_bg_with_irf', mfm.settings['tcspc']['shift_bg_with_irf'])

        lt = self.anisotropy.get_decay(lifetime_spectrum)
        #print "-------"
        decay = self.convolve.convolve(lt, verbose=verbose, scatter=scatter)
        #print decay

        # Calculate background curve from reference measurement
        background_curve = kwargs.get('background_curve', self.generic.background_curve)
        if isinstance(background_curve, mfm.curve.Curve):
            if shift_bg_with_irf:
                background_curve = background_curve << self.convolve.timeshift

            bg_y = background_curve.y.copy()
            bg_y /= bg_y.sum()
            bg_y *= self.generic.n_ph_bg

            decay *= self.generic.n_ph_fl
            decay += bg_y

        self.convolve.scale(decay)
        #print decay
        self.corrections.pileup(decay)
        decay += background
        self.corrections.linearize(decay)

        self._y = np.maximum(decay, 0)

        if verbose:
            print "<tau>x: %s " % self.species_averaged_lifetime
            print "<tau>f: %s " % self.fluorescence_averaged_lifetime


class LifetimeModelWidget(ModelWidget, LifetimeModel):

    def __init__(self, fit, **kwargs):
        ModelWidget.__init__(self, fit=fit, icon=QtGui.QIcon(":/icons/icons/TCSPC.png"))

        corrections = CorrectionsWidget(fit=fit, model=self, **kwargs)
        generic = GenericWidget(fit=fit, parent=self, model=self, **kwargs)
        anisotropy = AnisotropyWidget(name='anisotropy', model=self, short='rL', **kwargs)
        lifetimes = LifetimeWidget(name='lifetimes', model=self, parent=self, title='Lifetimes', short='L', fit=fit)
        convolve = ConvolveWidget(name='convolve',
                                  fit=fit,
                                  model=self,
                                  show_convolution_mode=False,
                                  **kwargs)

        LifetimeModel.__init__(self, fit)
        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)
        layout.setAlignment(QtCore.Qt.AlignTop)

        ## add widgets
        layout.addWidget(convolve)
        layout.addWidget(generic)
        layout.addWidget(lifetimes)
        layout.addWidget(anisotropy)
        layout.addWidget(corrections)
        #layout.addWidget(error_widget)
        self.setLayout(layout)
        self.layout = layout

        self.generic = generic
        self.corrections = corrections
        self.anisotropy = anisotropy
        self.lifetimes = lifetimes
        self.convolve = convolve
