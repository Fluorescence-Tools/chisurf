import numpy as np

import mfm
import mfm.fitting
import mfm.fitting.parameter

__author__ = 'Thomas'


class Anisotropy(mfm.fitting.parameter.FittingParameterGroup):

    @property
    def r0(self):
        return self._r0.value

    @r0.setter
    def r0(self, v):
        self._r0.value = v

    @property
    def l1(self):
        return self._l1.value

    @l1.setter
    def l1(self, v):
        self._r0.value = v

    @property
    def l2(self):
        return self._l2.value

    @l2.setter
    def l2(self, v):
        self._l2.value = v

    @property
    def g(self):
        return self._g.value

    @g.setter
    def g(self, v):
        self._g.value = v

    @property
    def rho(self):
        r = np.array([rho.value for rho in self._rhos], dtype=np.float64)
        r *= r
        r = np.sqrt(r)
        for i, v in enumerate(r):
            self._rhos[i].value = v
        return r

    @property
    def b(self):
        a = np.sqrt(np.array([g.value for g in self._bs]) ** 2)
        a /= a.sum()
        a *= self.r0
        for i, g in enumerate(self._bs):
            g.value = a[i]
        return a

    @property
    def rotation_spectrum(self):
        rot = np.empty(2 * len(self), dtype=np.float64)
        rot[0::2] = self.b
        rot[1::2] = self.rho
        return rot

    @property
    def polarization_type(self):
        return self._polarization_type

    @polarization_type.setter
    def polarization_type(self, v):
        self._polarization_type = v

    def get_decay(self, lifetime_spectrum):
        pt = self.polarization_type.upper()
        a = self.rotation_spectrum
        f = lifetime_spectrum
        if pt == 'VH' or pt == 'VV':
            d = mfm.fluorescence.general.elte2(a, f)
            vv = np.hstack([f, mfm.fluorescence.generale1tn(d, 2)])
            vh = mfm.fluorescence.e1tn(np.hstack([f, mfm.fluorescence.e1tn(d, -1)]), self.g)
        if self.polarization_type.upper() == 'VH':
            return np.hstack([mfm.fluorescence.e1tn(vv, self.l2), mfm.fluorescence.e1tn(vh, 1 - self.l2)])
        elif self.polarization_type.upper() == 'VV':
            r = np.hstack([mfm.fluorescence.e1tn(vv, 1 - self.l1), mfm.fluorescence.e1tn(vh, self.l1)])
            return r
        else:
            return f

    def __len__(self):
        return len(self._bs)

    def add_rotation(self, **kwargs):
        b_value = kwargs.get('b', 0.2)
        rho_value = kwargs.get('rho', 1.0)
        lb = kwargs.get('lb', None)
        ub = kwargs.get('ub', None)
        fixed = kwargs.get('fixed', False)
        bound_on = kwargs.get('bound_on', False)

        b = mfm.fitting.parameter.FittingParameter(lb=lb, ub=ub,
                                                   value=rho_value,
                                                   name='b(%i)' % (len(self) + 1),
                                                   fixed=fixed,
                                                   bounds_on=bound_on)
        rho = mfm.fitting.parameter.FittingParameter(lb=lb, ub=ub,
                                                     value=b_value,
                                                     name='rho(%i)' % (len(self) + 1),
                                                     fixed=fixed, bounds_on=bound_on)
        self._rhos.append(rho)
        self._bs.append(b)

    def remove_rotation(self):
        self._rhos.pop().close()
        self._bs.pop().close()

    def __init__(self, **kwargs):
        kwargs['name'] = 'Anisotropy'
        mfm.fitting.parameter.FittingParameterGroup.__init__(self, **kwargs)
        self._rhos = []
        self._bs = []
        self._polarization_type = kwargs.get('polarization', mfm.settings.cs_settings['tcspc']['polarization'])

        self._r0 = mfm.fitting.parameter.FittingParameter(name='r0', value=0.38, fixed=True)
        self._g = mfm.fitting.parameter.FittingParameter(name='g', value=1.00, fixed=True)
        self._l1 = mfm.fitting.parameter.FittingParameter(name='l1', value=0.0308, fixed=True)
        self._l2 = mfm.fitting.parameter.FittingParameter(name='l2', value=0.0368, fixed=True)