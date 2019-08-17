from __future__ import annotations
from typing import List

import os
import emcee
import numba as nb
import numpy as np
import scipy.linalg
import scipy.stats
from PyQt5 import QtCore, QtWidgets, uic


import mfm
import mfm.base
import mfm.fitting.models
from mfm.math.optimization.leastsqbound import leastsqbound

eps = np.sqrt(np.finfo(float).eps)


class Fit(mfm.base.Base):

    def __init__(
            self,
            model_class: mfm.fitting.models.Model = object,
            **kwargs
    ):
        mfm.base.Base.__init__(self, **kwargs)
        self._model = None
        self.results = None

        self._data = kwargs.get(
            'data',
            mfm.curve.DataCurve(x=np.arange(10), y=np.arange(10))
        )
        self.plots = list()
        self._xmin, self._xmax = kwargs.get('xmin', 0), kwargs.get('xmax', 0)
        self._model_kw = kwargs.get('model_kw', {})
        self.model = model_class

    def __str__(self):
        s = "\nFitting:\n"
        s += "Dataset:\n"
        s += "--------\n"
        s += str(self.data)
        s += "\n\nFit-result: \n"
        s += "----------\n"
        s += "fitrange: %i..%i\n" % (self.xmin, self.xmax)
        s += "chi2:\t%.4f \n" % self.chi2r
        s += "----------\n"
        s += str(self.model)
        return s

    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, v):
        self._xmin = max(0, v)

    @property
    def xmax(self):
        return self._xmax

    @xmax.setter
    def xmax(self, v):
        try:
            self._xmax = min(len(self.data.y) - 1, v)
        except AttributeError:
            self._xmax = v

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, v):
        self._data = v

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_class):
        if issubclass(model_class, mfm.fitting.models.Model):
            kw = self._model_kw
            self._model = model_class(self, **kw)

    @property
    def weighted_residuals(self):
        return self.model.weighted_residuals

    @property
    def chi2(self):
        """Unreduced Chi2
        """
        return get_chi2(self.model.parameter_values, model=self.model, reduced=False)

    @property
    def chi2r(self):
        """Reduced Chi2
        """
        pv = self.model.parameter_values
        return get_chi2(None, model=self.model)

    @property
    def name(self):
        try:
            return self._kw['name']
        except KeyError:
            try:
                return self.model.name + " - " + self._data.name
            except AttributeError:
                return "no name"

    @name.setter
    def name(self, n):
        self._name = n

    @property
    def fit_range(self):
        return self.xmin, self.xmax

    @fit_range.setter
    def fit_range(self, v):
        self.xmin, self.xmax = v

    @property
    def grad(self):
        """Get the approximate gradient at the current parameter values
        :return:
        """
        f0, grad = approx_grad(self.model.parameter_values, self, eps)
        return grad

    @property
    def covariance_matrix(self):
        """Returns the covariance matrix of the fit given the current
        model parameter values and returns a list of the 'relevant' used
        parameters.

        :return:
        """
        return covariance_matrix(self)

    @property
    def n_free(self):
        """The number of free parameters of the model
        """
        return self.model.n_free

    def get_chi2(self, parameter=None, model=None, reduced=True):
        if model is None:
            model = self.model
        return get_chi2(parameter, model, reduced)

    def get_wres(self, parameter=None, **kwargs):
        model = kwargs.get('model', self.model)
        if parameter is not None:
            model.parameter_values = parameter
            model.update_model()
        return model.get_wres(self, **kwargs)

    def save(
            self,
            path: str,
            name: str,
            mode: str = 'txt'
    ):
        filename = os.path.join(path, name)
        self.model.save(filename + '.json')
        if mode == 'txt':
            csv = mfm.io.ascii.Csv()
            wr = self.weighted_residuals
            xmin, xmax = self.xmin, self.xmax
            x, m = self.model[xmin:xmax]
            csv.save(np.vstack([x, wr]), filename+'_wr.txt')
            csv.save(self.model[:], filename+'_fit.txt')
            if isinstance(self.data, mfm.curve.Curve):
                self.data.save(filename+'_data.txt', mode='txt')
                self.data.save(filename+'_data.json', mode='json')
                with open(filename+'_info.txt', 'w') as fp:
                    fp.write(str(self))

    def run(self, **kwargs):
        self.model.find_parameters(parameter_type=mfm.parameter.FittingParameter)
        fitting_options = mfm.cs_settings['fitting']['leastsq']
        self.model.find_parameters()
        self.results = leastsqbound(get_wres,
                                    self.model.parameter_values,
                                    args=(self.model, ),
                                    bounds=self.model.parameter_bounds,
                                    **fitting_options)
        self.model.finalize()
        self.update()

    def update(self):
        self.model.update()
        # Estimate errors based on gradient
        try:
            fit = self
            cov_m, used_parameters = fit.covariance_matrix
            err = np.sqrt(np.diag(cov_m))
            for p, e in zip(used_parameters, err):
                fit.model.parameters[p].error_estimate = e
        except ValueError:
            print("Problems calculating the covariances")

    def chi2_scan(self, parameter_name, **kwargs):
        """Perform a chi2-scan on a parameter of the fit.

        :param parameter_name: the parameter name
        :param kwargs:
        :return: an list containing arrays of the chi2 and the parameter-values
        """
        parameter = self.model.parameters_all_dict[parameter_name]
        rel_range = max(parameter.error_estimate * 3.0 / parameter.value, 0.25)
        kwargs['rel_range'] = kwargs.get('rel_range', (rel_range, rel_range))
        parameter.parameter_scan = mfm.fitting.scan_parameter(self, parameter_name, **kwargs)
        return parameter.parameter_scan


class FitGroup(list, Fit):

    @property
    def selected_fit(self):
        return self[self._selected_fit]

    @selected_fit.setter
    def selected_fit(self, v):
        self._selected_fit = v

    @property
    def data(self):
        return self.selected_fit.data

    @data.setter
    def data(self, v):
        self.selected_fit.data = v

    @property
    def model(self):
        return self.selected_fit.model

    @model.setter
    def model(self, v):
        self.selected_fit.model = v

    @property
    def weighted_residuals(self):
        return [self.selected_fit.weighted_residuals]

    @property
    def chi2r(self):
        return self.selected_fit.chi2r

    @property
    def fit_range(self):
        return self.xmin, self.xmax

    @fit_range.setter
    def fit_range(self, v):
        for f in self:
            f.xmin, f.xmax = v
        self.xmin, self.xmax = v

    @property
    def xmin(self):
        return self.selected_fit.xmin

    @xmin.setter
    def xmin(self, v):
        for f in self:
            f.xmin = v
        self._xmin = v

    @property
    def xmax(self):
        return self.selected_fit.xmax

    @xmax.setter
    def xmax(self, v):
        for f in self:
            f.xmax = v
        self._xmax = v

    def update(self):
        self.global_model.update()
        for p in self.plots:
            p.update_all()

    def run(self, **kwargs):
        if kwargs.get('local_first', mfm.cs_settings['fitting']['global']['fit_local_first']):
            for f in self:
                f.run(**kwargs)
        for f in self:
            f.model.find_parameters()
        self.global_model.find_parameters()
        fitting_options = mfm.cs_settings['fitting']['leastsq']
        bounds = [pi.bounds for pi in self.global_model.parameters]
        self.results = leastsqbound(get_wres,
                                    self.global_model.parameter_values,
                                    args=(self.global_model, ),
                                    bounds=bounds,
                                    **fitting_options)
        self.update()
        self.global_model.finalize()

    def __init__(self, data, model_class, **kwargs):

        self._selected_fit = 0
        self._fits = list()
        for d in data:
            model_kw = kwargs.get('model_kw', {})
            fit = Fit(model_class=model_class, data=d, model_kw=model_kw)
            #fit = Fit(model_class=model_class, model_kw=model_kw)
            self._fits.append(fit)

        list.__init__(self, self._fits)
        Fit.__init__(self, data=data, **kwargs)

        self.global_model = mfm.fitting.models.GlobalFitModel(self)
        self.global_model.fits = self._fits

    def __str__(self):
        s = ""
        for f in self:
            s += str(f)
            s += "---\n"
        return s

    def next(self):
        if self._selected_fit > len(self._fits):
            raise StopIteration
        else:
            self._selected_fit += 1
            return self._fits[self._selected_fit - 1]


def walk_mcmc(
        fit: Fit,
        steps: int,
        step_size: float,
        chi2max: float,
        temp: float,
        thin: int
):
    """

    :param fit:
    :param steps:
    :param step_size:
    :param chi2max:
    :param temp:
    :param thin:
    :return:
    """
    dim = fit.model.n_free
    state_initial = fit.model.parameter_values
    n_samples = steps // thin
    # initialize arrays
    lnp = np.empty(n_samples)
    parameter = np.zeros((n_samples, dim))
    n_accepted = 0
    state_prev = np.copy(state_initial)
    lnp_prev = np.array(fit.lnprob(state_initial))
    while n_accepted < n_samples:
        state_next = state_prev + np.random.normal(0.0, step_size, dim) * state_initial
        lnp_next = fit.lnprob(state_next, chi2max)
        if not np.isfinite(lnp_next):
            continue
        if (-lnp_next + lnp_prev)/temp > np.log(np.random.rand()):
            # save results
            parameter[n_accepted] = state_next
            lnp[n_accepted] = lnp_next
            # switch previous and next
            np.copyto(state_prev, state_next)
            np.copyto(lnp_prev, lnp_next)
            n_accepted += 1
    return [lnp, parameter]


def sample_emcee(
        fit: Fit,
        steps: int,
        nwalkers: int,
        thin: int = 10,
        std: float = 1e-4,
        chi2max: float = np.inf):
    """Sample the parameter space by emcee using a number of 'walkers'

    :param fit: the fit to be samples
    :param steps: the number of steps of each walker
    :param thin: an integer (only every ith step is saved)
    :param nwalkers: the number of walkers
    :param chi2max: maximum allowed chi2
    :param std: the standard deviation of the parameters used to randomize the initial set of the walkers
    :return: a list containing the chi2 and the parameter values
    """
    model = fit.model
    ndim = fit.n_free # Number of free parameters to be sampled (number of dimensions)
    kw = {
        'parameters': fit.model.parameters,
        'bounds': fit.model.parameter_bounds,
        'chi2max': chi2max
    }
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[fit], kwargs=kw)
    std = np.array(fit.model.parameter_values) * std
    pos = [fit.model.parameter_values + std * np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(pos, steps, thin=thin)
    chi2 = -2. * sampler.flatlnprobability / float(model.n_points - model.n_free - 1.0)
    return chi2, sampler.flatchain


def sample_fit(
        fit: Fit,
        filename: str,
        **kwargs
):
    method = kwargs.pop('method', 'emcee')
    steps = kwargs.pop('steps', 1000)
    thin = kwargs.pop('thin', 1)
    chi2max = kwargs.pop('chi2max', np.inf)
    n_runs = kwargs.pop('n_runs', 10)
    # save initial parameter values
    pv = fit.model.parameter_values
    for i_run in range(n_runs):
        fn = os.path.splitext(filename)[0] + "_" + str(i_run) + '.er4'
        #if method == 'emcee':
        n_walkers = int(fit.n_free * 2)
        #try:
        chi2, para = sample_emcee(fit, steps=steps, nwalkers=n_walkers, thin=thin, chi2max=chi2max, **kwargs)
        #except ValueError:
        #    fit.model.parameter_values = pv
        #    fit.model.update()
        mask = np.where(np.isfinite(chi2))
        scan = np.vstack([chi2[mask], para[mask].T])
        header = "chi2\t"
        header += "\t".join(fit.model.parameter_names)
        mfm.io.ascii.Csv().save(scan, fn, delimiter='\t', mode='txt', header=header)
    # restore initial parameter values
    fit.model.parameter_values = pv
    fit.model.update()


#@nb.jit#(nopython=True)
def approx_grad(
        xk,
        fit: Fit,
        epsilon: float,
        args=(),
        f0=None
):
    """Approximate the derivative of a fit with respect to the parameters
    xk. The return value of the function is an array.

    :param xk: values around which gradient is estimated. These values should be an array of length of the
    free fitting parameters
    :param fit: object of type 'Fit'
    :param epsilon: differential change
    :param args: additional arguments passed to 'Fit.get_wres'
    :param f0: weighted residuals for the values xk
    :return:
    """
    p0 = fit.model.parameter_values
    f = fit.get_wres
    n_xk = len(xk)
    if f0 is None:
        f0 = f(*((xk,) + args))
    i = len(f0)
    grad = np.zeros((n_xk, i, ), float)
    ei = np.zeros((n_xk, ), float)

    for k in range(n_xk):
        ei[k] = 1.0
        d = epsilon * ei
        grad[k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0

    fit.model.parameter_values = p0
    return f0, grad


def covariance_matrix(
        fit: mfm.fitting.fit.Fit,
        **kwargs
):
    """Calculate the covariance matrix

    :param fit:
    :param kwargs:
    :return:
    """
    model = fit.model
    epsilon = kwargs.get('epsilon', mfm.eps)
    xk = np.array(model.parameter_values)

    fi_v, partial_derivatives = approx_grad(xk, fit, epsilon)

    # find parameters which do not change the model
    # use only parameters which change the model
    important_parameters = list()
    for k, pd_k in enumerate(partial_derivatives):
        if (pd_k**2).sum() > 0.0:
            important_parameters.append(k)
    pdi = partial_derivatives[important_parameters]
    n_important_parameters = len(important_parameters)
    m = np.zeros((n_important_parameters, n_important_parameters), float)

    for i_alpha in range(n_important_parameters):
        da_alpha = pdi[i_alpha]
        for i_beta in range(n_important_parameters):
            da_beta = pdi[i_beta]
            m[i_alpha, i_beta] = ((da_alpha * da_beta)).sum()
    try:
        cov_m = scipy.linalg.pinvh(0.5 * m)
    except ValueError:
        cov_m = np.zeros_like((n_important_parameters, n_important_parameters), dtype=float)
    return cov_m, important_parameters


@nb.jit(nopython=True)
def durbin_watson(
        residuals: np.array
):
    """Durbin-Watson parameter (1950,1951)

    :param residuals:  array
    :return:
    """
    n_res = len(residuals)
    nom = 0.0
    denom = np.sum(residuals ** 2)
    for i in range(1, n_res):
        nom += (residuals[i] - residuals[i - 1]) ** 2
    return nom / max(1, denom)


def get_wres(
        parameter: List[float],
        model: mfm.fitting.models.Model
):
    """Returns the weighted residuals for a list of parameters of a model

    :param parameter: a list of the parameter values / or None. If None the model is not updated
    :param model:
    :return:
    """
    if parameter is not None:
        model.parameter_values = parameter
        model.update_model()
    return model.weighted_residuals


def get_chi2(
        parameter: List[float],
        model: mfm.fitting.models.Model,
        reduced: bool = True
) -> float:
    """Returns either the reduced chi2 or the sum of squares (chi2)

    :param parameter: a list of the parameter values or None. If None the model is not updated
    :param model:
    :param reduced:
    :return:
    """
    chi2 = (get_wres(parameter, model)**2.0).sum()
    chi2 = np.inf if np.isnan(chi2) else chi2
    chi2r = chi2 / float(model.n_points - model.n_free - 1.0)
    if reduced:
        return chi2r
    else:
        return chi2


def chi2_max(
        chi2_value: float = 1.0,
        number_of_parameters: int = 1,
        nu: int = 1,
        conf_level: float = 0.95
) -> float:
    """Calculate the maximum chi2r of a fit given a certain confidence level

    :param chi2_value: the chi2 value
    :param number_of_parameters: the number of parameters of the model
    :param conf_level: the confidence level that is used to calculate the maximum chi2
    :param nu: the number of free degrees of freedom (number of observations - number of model parameters)
    """
    return chi2_value * (1.0 + float(number_of_parameters) / nu * scipy.stats.f.isf(1. - conf_level, number_of_parameters, nu))


def lnprior(
        parameter_values: List[float],
        fit: mfm.fitting.fit.Fit,
        **kwargs
) -> float:
    """The probability determined by the prior which is given by the bounds of the model parameters.
    If the model parameters leave the bounds, the ln of the probability is minus infinity otherwise it
    is zero.
    """
    bounds = kwargs.get('bounds', fit.model.parameter_bounds)
    for (bound, value) in zip(bounds, parameter_values):
        lb, ub = bound
        if lb is not None:
            if value < lb:
                return -np.inf
        if ub is not None:
            if value > ub:
                return -np.inf
    return 0.0


def lnprob(parameter_values, fit, chi2max=np.inf, **kwargs):
    """

    :param parameter_values:
    :param fit:
    :param chi2max:
    :return:
    """
    lp = lnprior(parameter_values, fit, **kwargs)
    if not np.isfinite(lp):
        return -np.inf
    else:
        chi2 = get_chi2(parameter_values, model=fit.model, reduced=False)
        lnlike = -0.5 * chi2 if chi2 < chi2max else -np.inf
        return lnlike + lp


def scan_parameter(
        fit: mfm.fitting.fit.Fit,
        parameter_name: str,
        scan_range=(None, None),
        rel_range: float = 0.2,
        n_steps: int = 30
):
    """Performs a chi2-scan for the parameter

    :param fit: the fit of type 'mfm.fitting.Fit'
    :param parameter_name: the name of the parameter (in the parameter dictionary)
    :param scan_range: the range within the parameter is scanned if not provided 'rel_range' is used
    :param rel_range: defines +- values for scanning
    :param n_steps: number of steps between +-
    :return:
    """
    # Store initial values before varying the parameter
    initial_parameter_values = fit.model.parameter_values

    varied_parameter = fit.model.parameters_all_dict[parameter_name]
    is_fixed = varied_parameter.fixed

    varied_parameter.fixed = True
    chi2r_array = np.zeros(n_steps, dtype=float)

    # Determine range within the parameter is varied
    parameter_value = varied_parameter.value
    p_min, p_max = scan_range
    if p_min is None or p_max is None:
        p_min = parameter_value * (1. - rel_range[0])
        p_max = parameter_value * (1. + rel_range[1])
    parameter_array = np.linspace(p_min, p_max, n_steps)

    for i, p in enumerate(parameter_array):
        varied_parameter.fixed = is_fixed
        fit.model.parameter_values = initial_parameter_values
        varied_parameter.fixed = True
        varied_parameter.value = p
        fit.run()
        chi2r_array[i] = fit.chi2r

    varied_parameter.fixed = is_fixed
    fit.model.parameter_values = initial_parameter_values
    fit.update()

    return parameter_array, chi2r_array


class FittingParameter(Parameter):

    def __init__(
            self,
            link=None,
            model: mfm.fitting.models.Model = None,
            lb: float = -10000,
            ub: float = 10000,
            fixed: bool = False,
            bounds_on: bool = False,
            error_estimate: bool = None,
            **kwargs
    ):
        super(FittingParameter, self).__init__(**kwargs)
        self._link = link
        self.model = model

        self._lb = lb
        self._ub = ub
        self._fixed = fixed
        self._bounds_on = bounds_on
        self._error_estimate = error_estimate

        self._chi2s = None
        self._values = None

    @property
    def parameter_scan(self):
        return self._values, self._chi2s

    @parameter_scan.setter
    def parameter_scan(self, v):
        self._values, self._chi2s = v

    @property
    def error_estimate(self):
        if self.is_linked:
            return self._link.error_estimate
        else:
            if isinstance(self._error_estimate, float):
                return self._error_estimate
            else:
                return "NA"

    @error_estimate.setter
    def error_estimate(self, v):
        self._error_estimate = v

    @property
    def bounds(self):
        if self.bounds_on:
            return self._lb, self._ub
        else:
            return None, None

    @bounds.setter
    def bounds(
            self,
            b: List[float]
    ):
        self._lb, self._ub = b

    @property
    def bounds_on(self) -> bool:
        return self._bounds_on

    @bounds_on.setter
    def bounds_on(
            self,
            v: bool
    ):
        self._bounds_on = bool(v)

    @property
    def value(self):
        v = self._value
        if callable(v):
            return v()
        else:
            if self.is_linked:
                return self.link.value
            else:
                if self.bounds_on:
                    lb, ub = self.bounds
                    if lb is not None:
                        v = max(lb, v)
                    if ub is not None:
                        v = min(ub, v)
                    return v
                else:
                    return v

    @value.setter
    def value(self, value):
        self._value = float(value)
        if self.is_linked:
            self.link.value = value

    @property
    def fixed(self) -> bool:
        return self._fixed

    @fixed.setter
    def fixed(
            self,
            v: bool
    ):
        self._fixed = v

    @property
    def link(self):
        return self._link

    @link.setter
    def link(
            self,
            link: mfm.parameter.FittingParameter
    ):
        if isinstance(link, FittingParameter):
            self._link = link
        elif link is None:
            try:
                self._value = self._link.value
            except AttributeError:
                pass
            self._link = None

    @property
    def is_linked(self) -> bool:
        return isinstance(self._link, FittingParameter)

    def to_dict(self) -> dict:
        d = Parameter.to_dict(self)
        d['lb'], d['ub'] = self.bounds
        d['fixed'] = self.fixed
        d['bounds_on'] = self.bounds_on
        d['error_estimate'] = self.error_estimate
        return d

    def from_dict(
            self,
            d: dict
    ):
        Parameter.from_dict(self, d)
        self._lb, self._ub = d['lb'], d['ub']
        self._fixed = d['fixed']
        self._bounds_on = d['bounds_on']
        self._error_estimate = d['error_estimate']

    def scan(
            self,
            fit: mfm.fitting.fit.Fit,
            **kwargs):
        fit.chi2_scan(self.name, **kwargs)

    def __str__(self):
        s = "\nVariable\n"
        s += "name: %s\n" % self.name
        s += "internal-value: %s\n" % self._value
        if self.bounds_on:
            s += "bounds: %s\n" % self.bounds
        if self.is_linked:
            s += "linked to: %s\n" % self.link.name
            s += "link-value: %s\n" % self.value
        return s

    def make_widget(
            self,
            **kwargs
    ) -> mfm.parameter.FittingParameterWidget:
        text = kwargs.get('text', self.name)
        layout = kwargs.get('layout', None)
        update_widget = kwargs.get('update_widget', lambda x: x)
        decimals = kwargs.get('decimals', self.decimals)
        kw = {
            'text': text,
            'decimals': decimals,
            'layout': layout
        }
        widget = FittingParameterWidget(self, **kw)
        self.controller = widget
        return widget


class FittingParameterWidget(QtWidgets.QWidget):

    def make_linkcall(self, fit_idx, parameter_name):

        def linkcall():
            tooltip = " linked to " + parameter_name
            mfm.run(
                "cs.current_fit.model.parameters_all_dict['%s'].link = mfm.fits[%s].model.parameters_all_dict['%s']" %
                (self.name, fit_idx, parameter_name)
            )
            self.widget_link.setToolTip(tooltip)
            self.widget_link.setChecked(True)
            self.widget_value.setEnabled(False)

        self.update()
        return linkcall

    def contextMenuEvent(self, event):

        menu = QtWidgets.QMenu(self)
        menu.setTitle("Link " + self.fitting_parameter.name + " to:")

        for fit_idx, f in enumerate(mfm.fits):
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)

                # Sorted by "Aggregation"
                for a in fs.model.aggregated_parameters:
                    action_submenu = QtWidgets.QMenu(submenu)
                    action_submenu.setTitle(a.name)
                    ut = a.parameters
                    ut.sort(key=lambda x: x.name, reverse=False)
                    for p in ut:
                        if p is not self:
                            Action = action_submenu.addAction(p.name)
                            Action.triggered.connect(self.make_linkcall(fit_idx, p.name))
                    submenu.addMenu(action_submenu)
                action_submenu = QtWidgets.QMenu(submenu)

                # Simply all parameters
                action_submenu.setTitle("All parameters")
                for p in fs.model.parameters_all:
                    if p is not self:
                        Action = action_submenu.addAction(p.name)
                        Action.triggered.connect(self.make_linkcall(fit_idx, p.name))
                submenu.addMenu(action_submenu)

                menu.addMenu(submenu)
        menu.exec_(event.globalPos())

    def __str__(self):
        return ""

    def __init__(self, fitting_parameter, **kwargs):
        layout = kwargs.pop('layout', None)
        label_text = kwargs.pop('text', self.__class__.__name__)
        hide_bounds = kwargs.pop('hide_bounds', parameter_settings['hide_bounds'])
        hide_link = kwargs.pop('hide_link', parameter_settings['hide_link'])
        fixable = kwargs.pop('fixable', parameter_settings['fixable'])
        hide_fix_checkbox = kwargs.pop('hide_fix_checkbox', parameter_settings['fixable'])
        hide_error = kwargs.pop('hide_error', parameter_settings['hide_error'])
        hide_label = kwargs.pop('hide_label', parameter_settings['hide_label'])
        decimals = kwargs.pop('decimals', parameter_settings['decimals'])

        super(FittingParameterWidget, self).__init__(**kwargs)
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('mfm/ui/variable_widget.ui', self)
        self.fitting_parameter = fitting_parameter
        self.widget_value = pg.SpinBox(dec=True, decimals=decimals)
        self.widget_value.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.widget_value.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addWidget(self.widget_value)

        self.widget_lower_bound = pg.SpinBox(dec=True, decimals=decimals)
        self.widget_lower_bound.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addWidget(self.widget_lower_bound)

        self.widget_upper_bound = pg.SpinBox(dec=True, decimals=decimals)
        self.widget_upper_bound.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addWidget(self.widget_upper_bound)

        # Hide and disable widgets
        self.label.setVisible(not hide_label)
        self.lineEdit.setVisible(not hide_error)
        self.widget_bounds_on.setDisabled(hide_bounds)
        self.widget_fix.setVisible(fixable or not hide_fix_checkbox)
        self.widget.setHidden(hide_bounds)
        self.widget_link.setDisabled(hide_link)

        # Display of values
        self.widget_value.setValue(float(self.fitting_parameter.value))

        self.label.setText(label_text.ljust(5))

        # variable bounds
        if not self.fitting_parameter.bounds_on:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Checked)

        # variable fixed
        if self.fitting_parameter.fixed:
            self.widget_fix.setCheckState(QtCore.Qt.Checked)
        else:
            self.widget_fix.setCheckState(QtCore.Qt.Unchecked)
        self.widget.hide()

        # The variable value
        self.widget_value.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].value = %s\n"
            "cs.current_fit.update()" %
            (self.fitting_parameter.name, self.widget_value.value()))
        )

        self.widget_fix.toggled.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].fixed = %s" %
            (self.fitting_parameter.name, self.widget_fix.isChecked()))
        )

        # Variable is bounded
        self.widget_bounds_on.toggled.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds_on = %s" %
            (self.fitting_parameter.name, self.widget_bounds_on.isChecked()))
        )

        self.widget_lower_bound.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds = (%s, %s)" %
            (self.fitting_parameter.name, self.widget_lower_bound.value(), self.widget_upper_bound.value()))
        )

        self.widget_upper_bound.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds = (%s, %s)" %
            (self.fitting_parameter.name, self.widget_lower_bound.value(), self.widget_upper_bound.value()))
        )

        self.widget_link.clicked.connect(self.onLinkFitGroup)

        if isinstance(layout, QtWidgets.QLayout):
            layout.addWidget(self)

    def onLinkFitGroup(self):
        self.blockSignals(True)
        cs = self.widget_link.checkState()
        if cs == 2:
            t = """
s = cs.current_fit.model.parameters_all_dict['%s']
for f in cs.current_fit:
   try:
       p = f.model.parameters_all_dict['%s']
       if p is not s:
           p.link = s
   except KeyError:
       pass
            """ % (self.fitting_parameter.name, self.fitting_parameter.name)
            mfm.run(t)
            self.widget_link.setCheckState(QtCore.Qt.Checked)
            self.widget_value.setEnabled(False)
        elif cs == 0:
            t = """
s = cs.current_fit.model.parameters_all_dict['%s']
for f in cs.current_fit:
   try:
       p = f.model.parameters_all_dict['%s']
       p.link = None
   except KeyError:
       pass
            """ % (self.fitting_parameter.name, self.fitting_parameter.name)
            mfm.run(t)
        self.widget_value.setEnabled(True)
        self.widget_link.setCheckState(QtCore.Qt.Unchecked)
        self.blockSignals(False)

    def finalize(self, *args):
        QtWidgets.QWidget.update(self, *args)
        self.blockSignals(True)
        # Update value of widget
        self.widget_value.setValue(float(self.fitting_parameter.value))
        if self.fitting_parameter.fixed:
            self.widget_fix.setCheckState(QtCore.Qt.Checked)
        else:
            self.widget_fix.setCheckState(QtCore.Qt.Unchecked)
        # Tooltip
        s = "bound: (%s,%s)\n" % self.fitting_parameter.bounds if self.fitting_parameter.bounds_on else "bounds: off\n"
        if self.fitting_parameter.is_linked:
            s += "linked to: %s" % self.fitting_parameter.link.name
        self.widget_value.setToolTip(s)

        # Error-estimate
        value = self.fitting_parameter.value
        if self.fitting_parameter.fixed or not isinstance(self.fitting_parameter.error_estimate, float):
            self.lineEdit.setText("NA")
        else:
            rel_error = abs(self.fitting_parameter.error_estimate / (value + 1e-12) * 100.0)
            self.lineEdit.setText("%.0f%%" % rel_error)
        #link
        if self.fitting_parameter.link is not None:
            tooltip = " linked to " + self.fitting_parameter.link.name
            self.widget_link.setToolTip(tooltip)
            self.widget_link.setChecked(True)
            self.widget_value.setEnabled(False)

        self.blockSignals(False)

class GlobalFittingParameter(FittingParameter):

    @property
    def value(self):
        g = self.g
        f = self.f
        r = eval(self.formula)
        return r.value

    @value.setter
    def value(self, v):
        pass

    @property
    def name(self):
        return self.formula

    @name.setter
    def name(self, v):
        pass

    def __init__(self, f, g, formula, **kwargs):
        #FittingParameter.__init__(self, **kwargs)
        args = [f, g, formula]
        super(GlobalFittingParameter, self).__init__(*args, **kwargs)
        self.f, self.g = f, g
        self.formula = formula
