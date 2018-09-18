import os
import numpy as np
from PyQt4 import QtCore, QtGui, uic
from scipy.linalg import pinvh
from scipy.stats import f as fdist
import mfm
from mfm import widgets
from mfm.fitting.optimization.leastsqbound import leastsqbound
import numba as nb
import emcee


#### Unused at the moment
def walk_mcmc(fit, steps, step_size, chi2max, temp, thin):
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


def sample_emcee(fit, steps, nwalkers, thin=10, std=1e-4, chi2max=np.inf):
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


def sample_fit(fit, filename, **kwargs):
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
def approx_grad(xk, fit, epsilon, args=(), f0=None):
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


def covariance_matrix(fit, **kwargs):
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
        cov_m = pinvh(0.5 * m)
    except ValueError:
        cov_m = np.zeros_like((n_important_parameters, n_important_parameters), dtype=float)
    return cov_m, important_parameters


@nb.jit(nopython=True)
def durbin_watson(residuals):
    """Durbin-Watson parameter (1950,1951)

    :param residuals:  array
    :return:
    """
    n_res = len(residuals)
    nom = 0.0
    denom = max(1.0, np.sum(residuals ** 2))
    for i in range(1, n_res):
        nom += (residuals[i] - residuals[i - 1]) ** 2
    return nom / denom


def get_wres(parameter, model):
    """Returns the weighted residuals for a list of parameters of a model

    :param parameter:
    :param model:
    :return:
    """
    model.parameter_values = parameter
    model.update_model()
    return model.weighted_residuals


def get_chi2(parameter, model, reduced=True):
    """Returns either the reduced chi2 or the sum of squares (chi2)

    :param parameter:
    :param model:
    :param reduced:
    :return:
    """
    chi2 = (get_wres(parameter, model)**2).sum()
    chi2 = np.inf if np.isnan(chi2) else chi2
    chi2r = chi2 / float(model.n_points - model.n_free - 1.0)
    if reduced:
        return chi2r
    else:
        return chi2


def chi2_max(fit=None, conf_level=0.95, **kwargs):
    """Calculate the maximum chi2r of a fit given a certain confidence level

    :param fit:
    :return:
    """
    try:
        npars = kwargs.get('npars', len(fit.model.parameter_values))
        nu = kwargs.get('nu', fit.model.n_points)
        chi2_min = kwargs.get('chi2_min', fit.chi2r)
        return chi2_min * (1.0 + float(npars) / nu * fdist.isf(1. - conf_level, npars, nu))
    except:
        return np.inf


def lnprior(parameter_values, fit, **kwargs):
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


def scan_parameter(fit, parameter_name, scan_range=(None, None), rel_range=0.2, n_steps=30):
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


class Fit(mfm.Base):

    def __init__(self, data=None, model_class=object, **kwargs):
        mfm.Base.__init__(self, **kwargs)
        self.results = None
        self._data = data
        self.plots = list()
        self._model = None
        self._xmin = kwargs.get('xmin', 0)
        self._xmax = kwargs.get('xmax', 0)

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
            self._model = model_class(self)

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
        return get_chi2(self.model.parameter_values, model=self.model)

    @property
    def name(self):
        if self._name is not None:
            return self._name
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
        f0, grad = approx_grad(self.model.parameter_values, self, mfm.eps)
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

    def save(self, path, name, mode='txt'):
        filename = os.path.join(path, name)
        self.model.save(filename+'.json')
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
        fitting_options = mfm.settings['fitting']['leastsq']

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
            cov_m, used_parameters = self.covariance_matrix
            err = np.sqrt(np.diag(cov_m)) * 3.0
            for p, e in zip(used_parameters, err):
                self.model.parameters[p].error_estimate = e
        except ValueError:
            print "Problems calculating the covariances"

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
    def current_fit(self):
        return self[self._current_fit]

    @current_fit.setter
    def current_fit(self, v):
        self._current_fit = v

    @property
    def data(self):
        return self.current_fit.data

    @data.setter
    def data(self, v):
        self.current_fit.data = v

    @property
    def model(self):
        return self.current_fit.model

    @model.setter
    def model(self, v):
        self.current_fit.model = v

    @property
    def weighted_residuals(self):
        return [self.current_fit.weighted_residuals]

    @property
    def chi2r(self):
        return self.current_fit.chi2r

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
        return self.current_fit.xmin

    @xmin.setter
    def xmin(self, v):
        for f in self:
            f.xmin = v
        self._xmin = v

    @property
    def xmax(self):
        return self.current_fit.xmax

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
        if kwargs.get('local_first', mfm.settings['fitting']['global']['fit_local_first']):
            for f in self:
                f.run(**kwargs)

        self.global_model.find_parameters()
        fitting_options = mfm.settings['fitting']['leastsq']
        bounds = [pi.bounds for pi in self.global_model.parameters]
        self.results = leastsqbound(get_wres,
                                    self.global_model.parameter_values,
                                    args=(self.global_model, ),
                                    bounds=bounds,
                                    **fitting_options)
        self.update()
        self.global_model.finalize()

    def __init__(self, data, model_class, **kwargs):
        self._current_fit = 0
        self._fits = list()
        for d in data:
            fit = Fit(data=d, model_class=model_class)
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
        if self._current_fit > len(self._fits):
            raise StopIteration
        else:
            self._current_fit += 1
            return self._fits[self._current_fit - 1]


class FittingControllerWidget(QtGui.QWidget):

    @property
    def selected_fit(self):
        return int(self.comboBox.currentIndex())
        #return int(self.spinBox.value())

    @selected_fit.setter
    def selected_fit(self, v):
        self.comboBox.setCurrentIndex(int(v))

    @property
    def current_fit_type(self):
        return str(self.comboBox.currentText())

    def change_dataset(self):
        dataset = self.curve_select.selected_dataset
        self.fit.data = dataset
        self.fit.update()
        self.lineEdit.clear()
        self.lineEdit.setText(dataset[0].name)

    def show_selector(self):
        self.curve_select.show()
        self.curve_select.update()

    def __init__(self, fit=None, **kwargs):
        QtGui.QWidget.__init__(self)
        self.curve_select = widgets.CurveSelector(parent=None, fit=self,
                                                  change_event=self.change_dataset,
                                                  setup=fit.data.setup.__class__)
        uic.loadUi("mfm/ui/fittingWidget.ui", self)

        self.curve_select.hide()
        hide_fit_button = kwargs.get('hide_fit_button', False)
        hide_range = kwargs.get('hide_range', False)
        hide_fitting = kwargs.get('hide_fitting', False)
        self.fit = fit
        fit_names = [os.path.basename(f.data.name) for f in fit]
        self.comboBox.addItems(fit_names)

        # decorate the update method of the fit
        # after decoration it should also call the update of
        # the fitting widget
        def wrapper(f):
            def update_new(*args, **kwargs):
                f(*args, **kwargs)
                self.update(*args)
                self.fit.model.update_plots(only_fit_range=True)
            return update_new

        self.fit.run = wrapper(self.fit.run)


        self.connect(self.actionFit, QtCore.SIGNAL('triggered()'), self.onRunFit)
        self.connect(self.actionFit_range_changed, QtCore.SIGNAL('triggered()'), self.onAutoFitRange)
        self.connect(self.actionChange_dataset, QtCore.SIGNAL('triggered()'), self.show_selector)
        self.connect(self.actionSelectionChanged, QtCore.SIGNAL('triggered()'), self.onDatasetChanged)
        self.connect(self.actionErrorEstimate, QtCore.SIGNAL('triggered()'), self.onErrorEstimate)

        if hide_fit_button:
            self.pushButton_fit.hide()
        if hide_range:
            self.toolButton_2.hide()
            self.spinBox.hide()
            self.spinBox_2.hide()
        if hide_fitting:
            self.hide()

        #self.spinBox.setMaximum(len(fit) - 1)
        #self.lineEdit.setText(fit.data.name)
        self.onAutoFitRange()

    def onDatasetChanged(self):
        mfm.run("cs.current_fit.model.hide()")
        mfm.run("cs.current_fit.current_fit = %i" % self.selected_fit)
        mfm.run("cs.current_fit.update()")
        mfm.run("cs.current_fit.model.show()")
        #name = self.fit.data.name
        #self.lineEdit.setText(name)

    def onErrorEstimate(self):
        filename = mfm.widgets.save_file('Error estimate', '*.er4')
        kw = mfm.settings['fitting']['sampling']
        sample_fit(self.fit, filename, **kw)

    def onRunFit(self):
        mfm.run("cs.current_fit.run()")

    def onAutoFitRange(self):
        try:
            self.fit.fit_range = self.fit.data.setup.autofitrange(self.fit.data)
        except AttributeError:
            self.fit.fit_range = 0, len(self.fit.data.x)
        self.fit.update()


class FitSubWindow(QtGui.QMdiSubWindow):

    def update(self, *__args):
        self.setWindowTitle(self.fit.name)
        QtGui.QMdiSubWindow.update(self, *__args)
        self.tw.update(self, *__args)

    def __init__(self, fit, control_layout, **kwargs):
        QtGui.QMdiSubWindow.__init__(self, kwargs.get('parent', None))
        self.setWindowTitle(fit.name)
        l = self.layout()

        self.tw = QtGui.QTabWidget()
        self.tw.setTabShape(QtGui.QTabWidget.Triangular)
        self.tw.setTabPosition(QtGui.QTabWidget.South)
        self.tw.currentChanged.connect(self.on_change_plot)

        l.addWidget(self.tw)
        self.close_confirm = kwargs.get('close_confirm', mfm.settings['gui']['confirm_close_fit'])
        self.fit = fit
        self.fit_widget = kwargs.get('fit_widget')

        self.current_plt_ctrl = QtGui.QWidget(self)
        self.current_plt_ctrl.hide()

        plots = list()
        for plot_class, kwargs in fit.model.plot_classes:
            plot = plot_class(fit, **kwargs)
            plot.pltControl.hide()
            plots.append(plot)
            self.tw.addTab(plot, plot.name)
            control_layout.addWidget(plot.pltControl)

        fit.plots = plots
        for f in fit:
            f.plots = plots

        self.on_change_plot()

    def on_change_plot(self):
        idx = self.tw.currentIndex()
        self.current_plt_ctrl.hide()
        self.current_plt_ctrl = self.fit.plots[idx].pltControl
        self.current_plt_ctrl.show()

    def updateStatusBar(self, msg):
        self.statusBar().showMessage(msg)

    def closeEvent(self, event):
        if self.close_confirm:
            reply = QtGui.QMessageBox.question(self, 'Message',
                                               "Are you sure to close this fit?:\n%s" % self.fit.name,
                                               QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

            if reply == QtGui.QMessageBox.Yes:
                mfm.console.execute('cs.close_fit()')
            else:
                event.ignore()
        else:
            event.accept()
