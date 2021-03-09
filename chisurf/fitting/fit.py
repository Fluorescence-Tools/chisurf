"""

"""
from __future__ import annotations
from chisurf import typing

import os
import numpy as np
import scipy.linalg
import scipy.stats

import chisurf.settings
import chisurf.base
import chisurf.fio
import chisurf.curve
import chisurf.experiments
import chisurf.data
import chisurf.fitting.parameter
import chisurf.fitting.sample
import chisurf.fitting.support_plane
import chisurf.models
import chisurf.math.statistics
import chisurf.math.optimization


class Fit(
    chisurf.base.Base
):

    def __init__(
            self,
            model_class: typing.Type[chisurf.models.Model] = type,
            data: chisurf.data.DataCurve = None,
            xmin: int = 0,
            xmax: int = 0,
            model_kw: typing.Dict = None,
            **kwargs
    ):
        """

        :param model_class:
        :param xmin:
        :param xmax:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._model = None
        self.results = None
        if data is None:
            data = chisurf.data.DataCurve(
                x=np.arange(10),
                y=np.arange(10)
            )
        self._data = data
        self.plots = list()
        self._xmin, self._xmax = xmin, xmax
        if model_kw is None:
            model_kw = {}
        self._model_kw = model_kw
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
    def xmin(
            self
    ) -> int:
        return self._xmin

    @xmin.setter
    def xmin(
            self,
            v: int
    ):
        self._xmin = max(0, v)

    @property
    def xmax(
            self
    ) -> int:
        return self._xmax

    @xmax.setter
    def xmax(
            self,
            v: int
    ):
        try:
            self._xmax = min(len(self.data.y) - 1, v)
        except AttributeError:
            self._xmax = v

    @property
    def data(
            self
    ) -> chisurf.data.DataCurve:
        return self._data

    @data.setter
    def data(
            self,
            v: chisurf.data.DataCurve
    ):
        self._data = v

    @property
    def model(
            self
    ) -> chisurf.models.ModelCurve:
        return self._model

    @model.setter
    def model(
            self,
            model_class: typing.Type[
                chisurf.models.model.ModelCurve
            ]
    ):
        if issubclass(model_class, chisurf.models.Model):
            self._model = model_class(
                self,
                **self._model_kw
            )

    @property
    def weighted_residuals(
            self
    ) -> chisurf.curve.Curve:
        wres_x, _ = self.model[self.xmin:self.xmax]
        wres_y = self.model.weighted_residuals
        return chisurf.curve.Curve(
            x=wres_x,
            y=wres_y,
            copy_array=False
        )

    @property
    def autocorrelation(self):
        wres = self.weighted_residuals
        return chisurf.curve.Curve(
            x=wres.x[1:],
            y=chisurf.math.signal.autocorr(wres.y)[1:],
            copy_array=False
        )

    @property
    def chi2(
            self
    ) -> float:
        """Unreduced Chi2
        """
        return get_chi2(
            self.model.parameter_values,
            model=self.model,
            reduced=False
        )

    @property
    def chi2r(
            self
    ) -> float:
        """Reduced Chi2
        """
        return get_chi2(
            list(),
            model=self.model
        )

    @property
    def name(
            self
    ) -> str:
        try:
            return self.model.name + " - " + self._data.name
        except AttributeError:
            return "no name"

    @property
    def fit_range(
            self
    ) -> typing.Tuple[int, int]:
        return self.xmin, self.xmax

    @fit_range.setter
    def fit_range(
            self,
            v: typing.Tuple[int, int]
    ):
        self.xmin, self.xmax = v

    @property
    def grad(
            self
    ) -> np.array:
        """Get the approximate gradient at the current parameter values
        :return:
        """
        _, grad = approx_grad(
            self.model.parameter_values,
            self,
            chisurf.settings.eps
        )
        return grad

    @property
    def covariance_matrix(
            self
    ) -> typing.Tuple[np.array, typing.typing.List[int]]:
        """Returns the covariance matrix of the fit given the current
        models parameter values and returns a list of the 'relevant' used
        parameters.

        :return:
        """
        return covariance_matrix(self)

    @property
    def n_free(
            self
    ) -> int:
        """The number of free parameters of the models
        """
        return self.model.n_free

    def get_curves(
            self,
            copy_curves: bool = False
    ) -> typing.Dict[str, chisurf.curve.Curve]:
        d = self.model.get_curves(
            copy_curves=copy_curves
        )
        d.update(
            {
                'data': self.data,
                'weighted residuals': self.weighted_residuals,
                'autocorrelation': self.autocorrelation
            }
        )
        return d

    def get_chi2(
            self,
            parameter=None,
            model: chisurf.models.Model = None,
            reduced: bool = True
    ) -> float:
        if model is None:
            model = self.model
        return get_chi2(
            parameter,
            model,
            reduced
        )

    def get_wres(
            self,
            parameter=None,
            model=None,
            **kwargs
    ) -> np.ndarray:
        if model is None:
            model = self.model
        if parameter is not None:
            model.parameter_values = parameter
            model.update_model()
        return model.get_wres(
            self,
            **kwargs
        )

    def save(
            self,
            filename: str,
            file_type: str = 'txt',
            verbose: bool = False,
            **kwargs
    ) -> None:
        super().save(
            filename=filename,
            file_type=file_type,
            verbose=verbose
        )
        # Save fit as txt (for plotting)
        curve_dict = self.get_curves()
        with open(filename+'_info.txt', mode='w') as fp:
            fp.write(str(self))
        for curve_key in curve_dict:
            curve = curve_dict[curve_key]
            curve_file_root = filename + "_%s" % curve_key
            curve.save(
                filename=curve_file_root,
                file_type='txt'
            )
            curve.save(
                filename=curve_file_root,
                file_type='json'
            )
            curve.save(
                filename=curve_file_root,
                file_type='yaml'
            )

    def run(
            self,
            *args,
            **kwargs
    ) -> None:
        fitting_options = chisurf.settings.cs_settings['optimization']['leastsq']
        self.model.find_parameters(
            parameter_type=chisurf.fitting.parameter.FittingParameter
        )
        self.results = chisurf.math.optimization.leastsqbound(
            get_wres,
            self.model.parameter_values,
            args=(self.model,),
            bounds=self.model.parameter_bounds,
            **fitting_options
        )
        self.model.finalize()
        self.update()

    def update(
            self
    ) -> None:
        self.model.update()
        # Estimate errors based on gradient
        try:
            fit = self
            cov_m, used_parameters = fit.covariance_matrix
            err = np.sqrt(np.diag(cov_m))
            for p, e in zip(used_parameters, err):
                fit.model.parameters[p].error_estimate = e
        except ValueError:
            chisurf.logging.warning("Problems calculating the covariances")

    def chi2_scan(
            self,
            parameter_name: str,
            rel_range: float = None,
            scan_range: typing.Tuple[float, float] = (None, None),
            n_steps: int = 30
    ) -> typing.Tuple[np.array, np.array]:
        """Perform a chi2-scan on a parameter of the fit.

        :param parameter_name: the parameter name
        :param rel_range: defines the scanning range as a fraction of the
        current value, e.g., for a value of 2.0 a rel_range of 0.5 scans
        from (2.0 - 2.0*0.5) to (2.0 + 2.0*0.5)
        :param kwargs:
        :return: an list containing arrays of the chi2 and the parameter-values
        """
        parameter = self.model.parameters_all_dict[parameter_name]
        if rel_range is None:
            rel_range = max(
                parameter.error_estimate * 3.0 / parameter.value,
                0.25
            )
        r = chisurf.fitting.support_plane.scan_parameter(
            fit=self,
            parameter_name=parameter_name,
            rel_range=rel_range,
            scan_range=scan_range,
            n_steps=n_steps
        )
        parameter.parameter_scan = r['parameter_values'], r['chi2r']
        return parameter.parameter_scan


class FitGroup(
    Fit
):

    @property
    def selected_fit(
            self
    ) -> Fit:
        return self.grouped_fits[self.selected_fit_index]

    @property
    def selected_fit_index(
            self
    ) -> int:
        return self._selected_fit_index

    @selected_fit.setter
    def selected_fit(
            self,
            v: int
    ):
        self._selected_fit_index = v

    @property
    def data(
            self
    ) -> chisurf.data.DataCurve:
        return self.selected_fit.data

    @data.setter
    def data(
            self,
            v: chisurf.base.Data
    ):
        self.selected_fit.data = v

    @property
    def model(
            self
    ) -> chisurf.models.Model:
        return self.selected_fit.model

    @model.setter
    def model(
            self,
            v: typing.Type[chisurf.models.Model]
    ):
        self.selected_fit.model = v

    @property
    def weighted_residuals(
            self
    ) -> chisurf.curve.Curve:
        return self.selected_fit.weighted_residuals

    @property
    def chi2r(
            self
    ) -> float:
        return self.selected_fit.chi2r

    @property
    def durbin_watson(
            self
    ) -> float:
        return chisurf.math.statistics.durbin_watson(
            self.weighted_residuals.y
        )

    @property
    def fit_range(
            self
    ) -> typing.Tuple[int, int]:
        return self.xmin, self.xmax

    @fit_range.setter
    def fit_range(
            self,
            v: typing.Tuple[int, int]
    ):
        for f in self:
            f.xmin, f.xmax = v
        self.xmin, self.xmax = v

    @property
    def xmin(
            self
    ) -> int:
        return self.selected_fit.xmin

    @xmin.setter
    def xmin(
            self,
            v: int
    ):
        for f in self:
            f.xmin = v

    @property
    def xmax(
            self
    ) -> int:
        return self.selected_fit.xmax

    @xmax.setter
    def xmax(
            self,
            v: int
    ):
        for f in self:
            f.xmax = v

    def get_curves(
            self,
            copy_curves: bool = False
    ) -> typing.Dict[str, chisurf.curve.Curve]:
        all_curves = super().get_curves()
        for i, f in enumerate(self.grouped_fits):
            fit_curves = f.get_curves(
                copy_curves=copy_curves
            )
            for curve_key in fit_curves:
                new_curve_key = curve_key + "_%02d" % i
                all_curves[new_curve_key] = fit_curves[curve_key]
        return all_curves

    def save(
            self,
            filename: str,
            file_type: str = 'txt',
            verbose: bool = False,
            **kwargs
    ) -> None:
        root, ext = os.path.splitext(filename)
        for i, fit in enumerate(self):
            root += "_%02d" % i
            fit.save(
                filename=filename,
                file_type=file_type,
                verbose=verbose,
                **kwargs
            )

    def finalize(self):
        self._model.finalize()

    def update(
            self
    ) -> None:
        self._model.update()
        for f in self.grouped_fits:
            f.model.update()

    def run(
            self,
            local_first: bool = None,
            **kwargs
    ):
        """Optimizes the free parameters

        :param local_first: if True the local parameters of a global-fit in a
        fit group are optimized first
        :param kwargs:
        :return:
        """
        fit: FitGroup = self
        if local_first is None:
            local_first = chisurf.settings.fitting['global_optimize_local_first']
        if local_first:
            for f in fit:
                f.run(**kwargs)
        for f in fit:
            f.model.find_parameters()
        fit._model.find_parameters()
        fitting_options = chisurf.settings.fitting['leastsq']
        bounds = [pi.bounds for pi in fit._model.parameters]
        results = chisurf.math.optimization.leastsqbound(
            func=get_wres,
            x0=fit._model.parameter_values,
            args=(fit._model,),
            bounds=bounds,
            **fitting_options
        )
        self.results = results
        self.update()

    def __init__(
            self,
            data: chisurf.data.DataGroup,
            model_class: typing.Type[chisurf.models.Model] = type,
            model_kw: typing.Dict = None
    ):
        """

        :param data:
        :param model_class:
        :param kwargs:
        """
        self._selected_fit_index = 0
        self.grouped_fits = list()

        for d in data:
            if model_kw is None:
                model_kw = dict()
            fit = Fit(
                model_class=model_class,
                data=d,
                model_kw=model_kw
            )
            self.grouped_fits.append(fit)

        super().__init__(
            data=data
        )
        self._model = chisurf.models.global_model.GlobalFitModel(
            fit=self,
            fits=self.grouped_fits
        )

    def to_dict(
            self,
            remove_protected: bool = False,
            copy_values: bool = True,
            convert_values_to_elementary: bool = False
    ) -> typing.Dict:
        d = super().to_dict(
            remove_protected=remove_protected,
            copy_values=copy_values,
            convert_values_to_elementary=convert_values_to_elementary
        )
        d['grouped_fits'] = [
            f.to_dict(
                remove_protected=remove_protected,
                copy_values=copy_values,
                convert_values_to_elementary=convert_values_to_elementary
            ) for f in self.grouped_fits
        ]
        return d

    def __str__(self):
        s = ""
        for f in self:
            s += str(f)
            s += "---\n"
        return s

    def next(self):
        if self._selected_fit_index > len(self.grouped_fits):
            raise StopIteration
        else:
            self._selected_fit_index += 1
            return self.grouped_fits[self._selected_fit_index - 1]

    def __getitem__(
            self,
            key
    ) -> typing.List[Fit]:
        if isinstance(key, int):
            return self.grouped_fits.__getitem__(key)
        else:
            start = 0 if key.start is None else key.start
            stop = len(self.grouped_fits) if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            key = slice(start, stop, step)
            return self.grouped_fits.__getitem__(key)


def sample_fit(
        fit: Fit,
        filename: str,
        method: str = 'emcee',
        steps: int = 1000,
        thin: int = 1,
        chi2max: float = float("inf"),
        n_runs: int = 10,
        step_size: float = 0.1,
        temp: float = 1.0,
        **kwargs
):
    """Samples the free paramter of a fit and writes the parameters with
    corresponding chi2 to a text file.

    :param fit:
    :param filename:
    :param method:
    :param steps:
    :param thin:
    :param chi2max:
    :param n_runs:
    :param kwargs:
    :return:
    """
    # save initial parameter values
    pv = fit.model.parameter_values
    for i_run in range(n_runs):
        fn = os.path.splitext(filename)[0] + "_" + str(i_run) + '.er4'

        if method == 'mcmc':
            r = chisurf.fitting.sample.walk_mcmc(
                fit=fit,
                steps=steps,
                thin=thin,
                chi2max=chi2max,
                step_size=step_size,
                temp=temp
            )
        else: #'emcee'
            n_walkers = int(fit.n_free * 2)
            r = chisurf.fitting.sample.sample_emcee(
                fit,
                steps=steps,
                nwalkers=n_walkers,
                thin=thin,
                chi2max=chi2max,
                **kwargs
            )

        chi2 = r['chi2r']
        parameter_values = r['parameter_values']
        parameter_names = r['parameter_names']

        mask = np.where(np.isfinite(chi2))
        scan = np.vstack([chi2[mask], parameter_values[mask].T])
        header = "chi2r\t"
        header += "\t".join(parameter_names)
        chisurf.fio.ascii.Csv().save(
            scan,
            fn,
            delimiter='\t',
            file_type='txt',
            header=header
        )
    # restore initial parameter values
    fit.model.parameter_values = pv
    fit.model.update()


#@nb.jit#(nopython=True)
def approx_grad(
        xk: np.array,
        fit: chisurf.fitting.fit.Fit,
        epsilon: float,
        args=(),
        f0=None
) -> typing.Tuple[float, np.array]:
    """Approximate the derivative of a fit with respect to the parameters
    xk. The return value of the function is an array.

    :param xk: values around which gradient is estimated. These values should
    be an array of length of the free fitting parameters
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
        fit: chisurf.fitting.fit.Fit,
        epsilon: float = chisurf.settings.eps,
        **kwargs
) -> typing.Tuple[np.array, typing.List[int]]:
    """Calculate the covariance matrix

    :param fit:
    :param kwargs:
    :return: the covariance matrix and a list of of parameter indices which
    are "important", i.e., have a partial derivative deviating from zero.
    """
    model = fit.model
    xk = np.array(model.parameter_values)
    fi_v, partial_derivatives = approx_grad(
        xk,
        fit,
        epsilon
    )

    # find parameters which do not change the models
    # use only parameters which change the models
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
        cov_m = np.zeros_like(
            (n_important_parameters, n_important_parameters),
            dtype=float
        )
    return cov_m, important_parameters


def get_wres(
        parameter_values: typing.List[float],
        model: chisurf.models.Model
) -> np.array:
    """Returns the weighted residuals for a list of parameters of a models

    :param parameter_values: a list of the parameter values / or None. If None
    the models is not updated.

    :param model:
    :return:
    """
    if len(parameter_values) > 0:
        model.parameter_values = parameter_values
        model.update_model()
    return model.weighted_residuals


def get_chi2(
        parameter_values: typing.List[float],
        model: chisurf.models.model.ModelCurve,
        reduced: bool = True
) -> float:
    """Returns either the reduced chi2 or the sum of squares (chi2)

    :param parameter: a list of the parameter values or None. If None the models
    is not updated
    :param model:
    :param reduced: If True the returned value is divided by
    (n_points - n_free - 1.0) where n_points is the number of data points
    and n_free is the number of model parameters
    :return:
    """
    chi2 = (get_wres(parameter_values, model)**2.0).sum()
    chi2 = np.inf if np.isnan(chi2) else chi2
    chi2r = chi2 / float(model.n_points - model.n_free - 1.0)
    if reduced:
        return chi2r
    else:
        return chi2


def lnprior(
        parameter_values: typing.List[float],
        fit: chisurf.fitting.fit.Fit,
        bounds: typing.List[
            typing.Tuple[float, float]
        ] = None
) -> float:
    """The probability determined by the prior which is given by the bounds
    of the models parameters. If the models parameters leave the bounds, the
    ln of the probability is minus infinity otherwise it is zero.
    """
    if bounds is None:
        bounds = fit.model.parameter_bounds
    for (bound, value) in zip(bounds, parameter_values):
        lb, ub = bound
        if lb is not None:
            if value < lb:
                return -np.inf
        if ub is not None:
            if value > ub:
                return -np.inf
    return 0.0


def lnprob(
        parameter_values: typing.List[float],
        fit: Fit,
        chi2max: float = float("inf"),
        bounds: typing.List[
            typing.Tuple[float, float]
        ] = None
) -> float:
    """

    :param parameter_values:
    :param bounds:
    :param fit:
    :param chi2max:
    :return:
    """
    lp = lnprior(
        parameter_values,
        fit,
        bounds=bounds
    )
    if not np.isfinite(lp):
        return float("-inf")
    else:
        chi2 = get_chi2(
            parameter_values,
            model=fit.model,
            reduced=False
        )
        lnlike = -0.5 * chi2 if chi2 < chi2max else -np.inf
        return lnlike + lp


