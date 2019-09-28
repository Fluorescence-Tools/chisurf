"""

"""
from __future__ import annotations
from typing import List, Tuple, Dict, Type

import os
import numpy as np
import scipy.linalg
import scipy.stats

import mfm
import mfm.base
import mfm.io
import mfm.curve
import mfm.experiments
import mfm.experiments.data
import mfm.fitting.parameter
import mfm.fitting.sample
import mfm.fitting.support_plane
import mfm.models

from mfm.math.optimization.leastsqbound import leastsqbound


class Fit(
    mfm.base.Base
):
    """

    """

    def __init__(
            self,
            model_class: Type[mfm.models.model.Model] = type,
            data: mfm.experiments.data.DataCurve = None,
            xmin: int = 0,
            xmax: int = 0,
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
            data = mfm.experiments.data.DataCurve(
                x=np.arange(10),
                y=np.arange(10)
            )

        self._data = data
        self.plots = list()
        self._xmin, self._xmax = xmin, xmax
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
    ) -> mfm.experiments.data.DataCurve:
        return self._data

    @data.setter
    def data(
            self,
            v: mfm.experiments.data.DataCurve
    ):
        self._data = v

    @property
    def model(
            self
    ) -> mfm.models.model.ModelCurve:
        return self._model

    @model.setter
    def model(
            self,
            model_class: Type[
                mfm.models.model.ModelCurve
            ]
    ):
        if issubclass(model_class, mfm.models.model.Model):
            self._model = model_class(
                self,
                **self._model_kw
            )

    @property
    def weighted_residuals(
            self
    ) -> np.array:
        return self.model.weighted_residuals

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
    ) -> Tuple[int, int]:
        return self.xmin, self.xmax

    @fit_range.setter
    def fit_range(
            self,
            v: Tuple[int, int]
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
            mfm.eps
        )
        return grad

    @property
    def covariance_matrix(
            self
    ) -> Tuple[np.array, List[int]]:
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
            self
    ) -> Dict[str, mfm.curve.Curve]:
        """Returns a dictionary containing the current data and the
        model as mfm.curve.Curve objects.

        :return:
        """
        return {
            'model': self.model,
            'data': self.data
        }

    def get_chi2(
            self,
            parameter=None,
            model: mfm.models.model.Model = None,
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
        self.model.save(filename + '.json')
        if file_type == 'txt':
            csv = mfm.io.ascii.Csv()
            wr = self.weighted_residuals
            xmin, xmax = self.xmin, self.xmax
            x, m = self.model[xmin:xmax]
            csv.save(
                data=np.vstack([x, wr]),
                filename=filename+'_wr.txt'
            )
            csv.save(
                data=self.model[:],
                filename=filename+'_fit.txt'
            )
            if isinstance(
                    self.data,
                    mfm.curve.Curve
            ):
                self.data.save(
                    filename=filename + '_data.txt',
                    file_type='txt'
                )
                self.data.save(
                    filename=filename + '_data.json',
                    file_type='json'
                )
                with open(filename+'_info.txt', 'w') as fp:
                    fp.write(str(self))

    def run(
            self,
            *args,
            **kwargs
    ) -> None:
        fitting_options = mfm.settings.cs_settings['fitting']['leastsq']
        self.model.find_parameters(
            parameter_type=mfm.fitting.parameter.FittingParameter
        )
        self.results = leastsqbound(
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
            print("Problems calculating the covariances")

    def chi2_scan(
            self,
            parameter_name: str,
            rel_range: float = None,
            **kwargs
    ) -> Tuple[np.array, np.array]:
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
        kwargs['rel_range'] = (rel_range, rel_range)
        parameter.parameter_scan = mfm.fitting.support_plane.scan_parameter(
            self,
            parameter_name,
            **kwargs
        )
        return parameter.parameter_scan


class FitGroup(
    list,
    Fit
):

    @property
    def selected_fit(
            self
    ) -> Fit:
        return self[self.selected_fit_index]

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
    ) -> mfm.experiments.data.DataCurve:
        return self.selected_fit.data

    @data.setter
    def data(
            self,
            v: mfm.base.Data
    ):
        self.selected_fit.data = v

    @property
    def model(
            self
    ) -> mfm.models.model.Model:
        return self.selected_fit.model

    @model.setter
    def model(
            self,
            v: Type[mfm.models.model.Model]
    ):
        self.selected_fit.model = v

    @property
    def weighted_residuals(
            self
    ) -> np.ndarray:
        return self.selected_fit.weighted_residuals

    @property
    def chi2r(
            self
    ) -> float:
        return self.selected_fit.chi2r

    @property
    def fit_range(
            self
    ) -> Tuple[int, int]:
        return self.xmin, self.xmax

    @fit_range.setter
    def fit_range(
            self,
            v: Tuple[int, int]
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
        self._xmin = v

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
        self._xmax = v

    def update(
            self
    ) -> None:
        self.global_model.update()
        for p in self.plots:
            p.update_all()

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
        fit = self
        if local_first is None:
            local_first = mfm.settings.fitting['global']['fit_local_first']

        if local_first:
            for f in fit:
                f.run(
                    **kwargs
                )
        for f in fit:
            f.model.find_parameters()

        fit.global_model.find_parameters()
        fitting_options = mfm.settings.fitting['leastsq']
        bounds = [pi.bounds for pi in fit.global_model.parameters]

        results = leastsqbound(
            get_wres,
            fit.global_model.parameter_values,
            args=(
                fit.global_model,
            ),
            bounds=bounds,
            **fitting_options
        )
        self.results = results

        self.update()
        self.global_model.finalize()

    def __init__(
            self,
            data: mfm.experiments.data.DataGroup,
            model_class: Type[mfm.models.model.Model] = type,
            model_kw: Dict = None,
            **kwargs
    ):
        """

        :param data:
        :param model_class:
        :param kwargs:
        """
        self._selected_fit_index = 0
        self._fits = list()

        for d in data:
            if model_kw is None:
                model_kw = dict()
            fit = Fit(
                model_class=model_class,
                data=d,
                model_kw=model_kw
            )
            self._fits.append(fit)

        list.__init__(self, self._fits)
        Fit.__init__(
            self,
            data=data,
            **kwargs
        )

        self.global_model = mfm.models.global_model.GlobalFitModel(
            self
        )
        self.global_model.fits = self._fits

    def __str__(self):
        s = ""
        for f in self:
            s += str(f)
            s += "---\n"
        return s

    def next(self):
        if self._selected_fit_index > len(self._fits):
            raise StopIteration
        else:
            self._selected_fit_index += 1
            return self._fits[self._selected_fit_index - 1]


def sample_fit(
        fit: Fit,
        filename: str,
        method: str = 'emcee',
        steps: int = 1000,
        thin: int = 1,
        chi2max: float = float("inf"),
        n_runs: int = 10,
        **kwargs
):
    # save initial parameter values
    pv = fit.model.parameter_values
    for i_run in range(n_runs):
        fn = os.path.splitext(filename)[0] + "_" + str(i_run) + '.er4'

        if method != 'emcee':
            pass
        else:
            n_walkers = int(fit.n_free * 2)
            try:
                chi2, para = mfm.fitting.sample.sample_emcee(
                    fit,
                    steps=steps,
                    nwalkers=n_walkers,
                    thin=thin,
                    chi2max=chi2max,
                    **kwargs
                )
            except ValueError:
                fit.models.parameter_values = pv
                fit.models.update()

        mask = np.where(np.isfinite(chi2))
        scan = np.vstack([chi2[mask], para[mask].T])
        header = "chi2\t"
        header += "\t".join(fit.model.parameter_names)
        mfm.io.ascii.Csv().save(
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
        fit: mfm.fitting.fit.Fit,
        epsilon: float,
        args=(),
        f0=None
) -> Tuple[float, np.array]:
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
        fit: mfm.fitting.fit.Fit,
        epsilon: float = mfm.eps,
        **kwargs
) -> Tuple[np.array, List[int]]:
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
        parameter: List[float],
        model: mfm.models.model.Model
) -> np.array:
    """Returns the weighted residuals for a list of parameters of a models

    :param parameter: a list of the parameter values / or None. If None
    the models is not updated.

    :param model:
    :return:
    """
    if len(parameter) > 0:
        model.parameter_values = parameter
        model.update_model()
    return model.weighted_residuals


def get_chi2(
        parameter: List[float],
        model: mfm.models.model.ModelCurve,
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
    chi2 = (get_wres(parameter, model)**2.0).sum()
    chi2 = np.inf if np.isnan(chi2) else chi2
    chi2r = chi2 / float(model.n_points - model.n_free - 1.0)
    if reduced:
        return chi2r
    else:
        return chi2


def lnprior(
        parameter_values: List[float],
        fit: mfm.fitting.fit.Fit,
        bounds: List[Tuple[float, float]] = None
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
        parameter_values: List[float],
        fit: Fit,
        chi2max: float = float("inf"),
        **kwargs
) -> float:
    """

    :param parameter_values:
    :param fit:
    :param chi2max:
    :return:
    """
    lp = lnprior(parameter_values, fit, **kwargs)
    if not np.isfinite(lp):
        return float("-inf")
    else:
        chi2 = get_chi2(parameter_values, model=fit.model, reduced=False)
        lnlike = -0.5 * chi2 if chi2 < chi2max else -np.inf
        return lnlike + lp


