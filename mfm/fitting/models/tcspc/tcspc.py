import math

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import mfm
import mfm.fluorescence.tcspc
import mfm.math.signal
from mfm.fitting.models import Model, ModelWidget, ModelCurve
from mfm.fitting.models.tcspc.nusiance import Generic, GenericWidget, Corrections, CorrectionsWidget
from mfm.fluorescence.anisotropy import Anisotropy
from mfm.fluorescence.general import species_averaged_lifetime, fluorescence_averaged_lifetime
from mfm.fluorescence.tcspc.convolve import Convolve, ConvolveWidget
from mfm.fluorescence.widgets import AnisotropyWidget
from mfm.parameter import FittingParameter, FittingParameterGroup


class Lifetime(FittingParameterGroup):

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

    def append(self, *args, **kwargs):
        amplitude = args[0] if len(args) > 2 else 1.0
        lifetime = args[1] if len(args) > 2 else 4.0
        amplitude = kwargs.get('amplitude', amplitude)
        lifetime = kwargs.get('amplitude', lifetime)
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
        FittingParameterGroup.__init__(self, **kwargs)
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


class LifetimeWidget(Lifetime, QtWidgets.QWidget):

    def update(self, *__args):
        QtWidgets.QWidget.update(self, *__args)
        Lifetime.update(self)

    def read_values(self, target):
        def linkcall():
            for key in self.parameter_dict:
                v = target.parameters_all_dict[key].value
                mfm.run("cs.current_fit.model.parameters_all_dict['%s'].value = %s" % (key, v))
            mfm.run("cs.current_fit.update()")
        return linkcall

    def read_menu(self):
        menu = QtWidgets.QMenu()
        for f in mfm.fits:
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
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
        menu = QtWidgets.QMenu()
        for f in mfm.fits:
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)
                for a in fs.model.aggregated_parameters:
                    if isinstance(a, LifetimeWidget):
                        Action = submenu.addAction(a.name)
                        Action.triggered.connect(self.link_values(a))
                menu.addMenu(submenu)
        self.linkFrom.setMenu(menu)
        #menu.exec_(event.globalPos())

    def __init__(self, title='', **kwargs):
        QtWidgets.QWidget.__init__(self)
        Lifetime.__init__(self, **kwargs)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.gb = QtWidgets.QGroupBox()
        self.gb.setTitle(title)
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.gb.setLayout(self.lh)
        self.layout.addWidget(self.gb)
        self._amp_widgets = list()
        self._lifetime_widgets = list()

        lh = QtWidgets.QHBoxLayout()
        lh.setSpacing(0)
        lh.setContentsMargins(0, 0, 0, 0)

        addDonor = QtWidgets.QPushButton()
        addDonor.setText("add")
        addDonor.clicked.connect(self.onAddLifetime)
        lh.addWidget(addDonor)

        removeDonor = QtWidgets.QPushButton()
        removeDonor.setText("del")
        removeDonor.clicked.connect(self.onRemoveLifetime)
        lh.addWidget(removeDonor)

        spacerItem = QtWidgets.QSpacerItem(20, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        lh.addItem(spacerItem)

        readFrom = QtWidgets.QToolButton()
        readFrom.setText("read")
        readFrom.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        readFrom.clicked.connect(self.read_menu)
        lh.addWidget(readFrom)
        self.readFrom = readFrom

        linkFrom = QtWidgets.QToolButton()
        linkFrom.setText("link")
        linkFrom.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        linkFrom.clicked.connect(self.link_menu)
        lh.addWidget(linkFrom)
        self.linkFrom = linkFrom

        normalize_amplitude = QtWidgets.QCheckBox("Norm.")
        normalize_amplitude.setChecked(True)
        normalize_amplitude.setToolTip("Normalize amplitudes to unity.\nThe sum of all amplitudes equals one.")
        normalize_amplitude.clicked.connect(self.onNormalizeAmplitudes)
        self.normalize_amplitude = normalize_amplitude

        absolute_amplitude = QtWidgets.QCheckBox("Abs.")
        absolute_amplitude.setChecked(True)
        absolute_amplitude.setToolTip("Take absolute value of amplitudes\nNo negative amplitudes")
        absolute_amplitude.clicked.connect(self.onAbsoluteAmplitudes)
        self.absolute_amplitude = absolute_amplitude

        lh.addWidget(absolute_amplitude)
        lh.addWidget(normalize_amplitude)
        self.lh.addLayout(lh)

        self.append()

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
            "   f.model.update()" % self.name
        mfm.run(t)

    def onRemoveLifetime(self):
        t = "for f in cs.current_fit:\n" \
            "   f.model.%s.pop()\n" \
            "   f.model.update()" % self.name
        mfm.run(t)

    def append(self, *args, **kwargs):
        Lifetime.append(self, *args, **kwargs)
        l = QtWidgets.QHBoxLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)
        a = self._amplitudes[-1].make_widget(layout=l)
        t = self._lifetimes[-1].make_widget(layout=l)
        self._amp_widgets.append(a)
        self._lifetime_widgets.append(t)
        self.lh.addLayout(l)

    def pop(self):
        self._amplitudes.pop()
        self._lifetimes.pop()
        self._amp_widgets.pop().close()
        self._lifetime_widgets.pop().close()


class LifetimeModel(ModelCurve):

    name = "Lifetime fit"

    def __str__(self):
        s = Model.__str__(self)
        s += "\nLifetimes"
        s += "\n------------------\n"
        s += "\nAverage Lifetimes:\n"
        s += "<tau>x: %.3f\n<tau>F: %.3f\n" % (self.species_averaged_lifetime, self.fluorescence_averaged_lifetime)
        return s

    def __init__(self, fit, **kwargs):
        ModelCurve.__init__(self, fit, **kwargs)
        self.generic = kwargs.get('generic', Generic(name='generic', fit=fit, **kwargs))
        self.corrections = kwargs.get('corrections', Corrections(name='corrections', fit=fit, **kwargs))
        self.anisotropy = kwargs.get('anisotropy', Anisotropy(name='anisotropy', **kwargs))
        self.lifetimes = kwargs.get('lifetimes', Lifetime(name='lifetimes', fit=fit, **kwargs))
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
        shift_bg_with_irf = kwargs.get('shift_bg_with_irf', mfm.cs_settings['tcspc']['shift_bg_with_irf'])
        lt = self.anisotropy.get_decay(lifetime_spectrum)
        decay = self.convolve.convolve(lt, verbose=verbose, scatter=scatter)

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

        self.convolve.scale(decay, bg=self.generic.background)
        self.corrections.pileup(decay)
        decay += background
        decay = self.corrections.linearize(decay)
        self.y = np.maximum(decay, 0)


class LifetimeModelWidgetBase(ModelWidget, LifetimeModel):

    def __init__(self, fit, icon=None, **kwargs):
        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/TCSPC.png")
        ModelWidget.__init__(self, fit=fit, icon=icon)
        hide_nuisances = kwargs.get('hide_nuisances', False)

        corrections = CorrectionsWidget(fit=fit, **kwargs)
        generic = GenericWidget(fit=fit, parent=self, **kwargs)
        anisotropy = AnisotropyWidget(name='anisotropy', short='rL', **kwargs)
        convolve = ConvolveWidget(name='convolve', fit=fit, show_convolution_mode=False, **kwargs)

        LifetimeModel.__init__(self, fit)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(QtCore.Qt.AlignTop)

        ## add widgets
        if not hide_nuisances:
            layout.addWidget(convolve)
            layout.addWidget(generic)

        self.layout_parameter = QtWidgets.QVBoxLayout()
        self.layout_parameter.setContentsMargins(0, 0, 0, 0)
        self.layout_parameter.setSpacing(0)

        layout.addLayout(self.layout_parameter)
        if not hide_nuisances:
            layout.addWidget(anisotropy)
            layout.addWidget(corrections)

        self.setLayout(layout)
        self.layout = layout

        self.generic = generic
        self.corrections = corrections
        self.anisotropy = anisotropy
        self.convolve = convolve


class LifetimeModelWidget(LifetimeModelWidgetBase):

    def __init__(self, fit, **kwargs):
        LifetimeModelWidgetBase.__init__(self, fit=fit, **kwargs)
        self.lifetimes = LifetimeWidget(name='lifetimes', parent=self, title='Lifetimes', short='L', fit=fit)
        self.layout_parameter.addWidget(self.lifetimes)


class DecayModel(ModelCurve):

    name = "Fluorescence decay model"

    def __init__(self, fit, **kwargs):
        ModelCurve.__init__(self, fit, **kwargs)
        self.generic = kwargs.get('generic', Generic(name='generic', fit=fit, **kwargs))
        self.corrections = kwargs.get('corrections', Corrections(name='corrections', fit=fit, **kwargs))
        self.convolve = kwargs.get('convolve', Convolve(name='convolve', fit=fit, **kwargs))
        self._decay = None

    @property
    def species_averaged_lifetime(self):
        return species_averaged_lifetime([self.times, self.decay], is_lifetime_spectrum=False)

    @property
    def fluorescence_averaged_lifetime(self):
        return fluorescence_averaged_lifetime([self.times, self.decay], is_lifetime_spectrum=False)

    @property
    def times(self):
        return self.fit.data.x

    @property
    def decay(self):
        return self._decay

    def update_model(self, **kwargs):
        verbose = kwargs.get('verbose', mfm.verbose)
        scatter = kwargs.get('scatter', self.generic.scatter)
        background = kwargs.get('background', self.generic.background)
        shift_bg_with_irf = kwargs.get('shift_bg_with_irf', mfm.cs_settings['tcspc']['shift_bg_with_irf'])
        decay = self.decay
        convolved_decay = self.convolve.convolve(decay, verbose=verbose, scatter=scatter, mode='full')

        # Calculate background curve from reference measurement
        background_curve = kwargs.get('background_curve', self.generic.background_curve)
        if isinstance(background_curve, mfm.curve.Curve):
            if shift_bg_with_irf:
                background_curve = background_curve << self.convolve.timeshift

            bg_y = background_curve.y.copy()
            bg_y /= bg_y.sum()
            bg_y *= self.generic.n_ph_bg

            convolved_decay *= self.generic.n_ph_fl
            decay += bg_y

        convolved_decay = self.convolve.scale(convolved_decay, bg=self.generic.background)
        self.corrections.pileup(convolved_decay)
        convolved_decay += background
        convolved_decay = self.corrections.linearize(convolved_decay)
        self._y = np.maximum(convolved_decay, 0)

