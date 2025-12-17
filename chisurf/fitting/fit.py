from __future__ import annotations

from chisurf import typing
from collections import deque

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
from chisurf.math.optimization.leastsqbound import OptimizationCancelled


def _raw_fit_name(f) -> str:
    """Compute the base (non-unique) name for a Fit/FitGroup instance."""
    try:
        model = getattr(f, "model", None)
        model_cls = getattr(model, "__class__", None)
        model_name = getattr(model_cls, "name", None)
        if model_name is None:
            model_name = getattr(model_cls, "__name__", "no model")
    except Exception:
        model_name = "no model"

    try:
        data = getattr(f, "_data", None)
        data_name = getattr(data, "name", None)
    except Exception:
        data_name = None

    if not data_name:
        data_name = "no data"

    return f"{model_name} - {data_name}"


class Fit(chisurf.base.Base):
    """Fit of a single data set with a single model.

    The :class:`Fit` object owns a :class:`chisurf.data.DataCurve` instance
    (accessible via :attr:`data`) and a :class:`chisurf.models.ModelCurve`
    instance (via :attr:`model`). It provides convenience properties for
    weighted residuals, chi² statistics, and running a local least-squares
    optimization.
    """

    @property
    def fit_idx(self) -> int:
        return chisurf.fitting.find_fit_idx(self)

    @property
    def xmin(self) -> int:
        return self._xmin

    @xmin.setter
    def xmin(self, v: int):
        self._xmin = max(0, v)

    @property
    def xmax(self) -> int:
        return self._xmax

    @xmax.setter
    def xmax(self, v: int):
        try:
            self._xmax = min(len(self.data.y) - 1, v)
        except AttributeError:
            self._xmax = v

    @property
    def data(self) -> chisurf.data.DataCurve:
        return self._data

    @data.setter
    def data(self, v: chisurf.data.DataCurve):
        self._data = v

    @property
    def model(self) -> chisurf.models.ModelCurve:
        return self._model

    @model.setter
    def model(
            self,
            model_class: typing.Type[
                chisurf.models.model.ModelCurve
            ]
    ):
        if issubclass(model_class, chisurf.models.Model):
            self._model = model_class(self, **self._model_kw)

    @property
    def weighted_residuals(self) -> chisurf.curve.Curve:
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
    def chi2(self) -> float:
        return get_chi2(
            self.model.parameter_values,
            model=self.model,
            reduced=False
        )

    @property
    def chi2r(self) -> float:
        return get_chi2(list(), model=self.model)

    @property
    def durbin_watson(self) -> float:
        return chisurf.math.statistics.durbin_watson(
            self.weighted_residuals.y
        )

    @property
    def name(self) -> str:
        """Return a human-readable, *globally unique* fit name.

        The base name is derived from the model class and attached data
        ("ModelName - DataName"). If multiple active fits share the same
        base name, a numeric suffix " (1)", " (2)", ... is appended based
        on the order of appearance in ``chisurf.fits``.
        """

        try:
            base_name = _raw_fit_name(self)
        except Exception:
            return "no name"

        # Try to enforce uniqueness across active fits tracked in chisurf.fits.
        # On any error we simply fall back to the base name.
        try:
            all_fits = []
            for fg in getattr(chisurf, "fits", []):
                if isinstance(fg, Fit):
                    all_fits.append(fg)
                    grouped = getattr(fg, "grouped_fits", None)
                    if isinstance(grouped, (list, tuple)):
                        for lf in grouped:
                            if isinstance(lf, Fit):
                                all_fits.append(lf)

            if not all_fits:
                return base_name

            # Collect all fits that share the same base name, using the
            # raw-name helper to avoid recursion via the name property.
            same_base = [f for f in all_fits if _raw_fit_name(f) == base_name]
            if not same_base:
                return base_name

            try:
                idx = same_base.index(self)
            except ValueError:
                # This fit is not registered in chisurf.fits; treat it as
                # a standalone instance without suffix.
                return base_name

            if idx == 0:
                # First fit with this base name keeps the plain name.
                return base_name

            # Subsequent fits with the same base name receive a numeric
            # suffix reflecting their order of appearance.
            return f"{base_name} ({idx})"
        except Exception:
            return base_name

    @property
    def fit_range(self) -> typing.Tuple[int, int]:
        return self.xmin, self.xmax

    @fit_range.setter
    def fit_range(self, v):
        vals = tuple(int(x) for x in v)
        if len(vals) == 2:
            # Backwards-compatible 1D range: do not touch any existing mask.
            xmin1, xmax1 = vals
            self.xmin, self.xmax = xmin1, xmax1

            # Rebuild a simple 1D mask matching the fit range so that APIs
            # like ``cs.current_fit.fit_range = i, j`` also update the mask
            # used by residual-based tools and the Data table.
            try:
                y = getattr(self.data, "y", None)
                n = int(len(y)) if y is not None else 0
            except Exception:
                n = 0
            if n <= 0:
                self.mask = None
                return

            lb1 = max(0, int(xmin1))
            ub1 = max(lb1, min(int(xmax1), n))
            mask = np.zeros(n, dtype=float)
            if ub1 > lb1:
                mask[lb1:ub1] = 1.0
            self.mask = mask
            return
        if len(vals) != 4:
            raise ValueError("fit_range must be a 2- or 4-tuple of integers")

        xmin1, xmax1, xmin2, xmax2 = vals
        # Primary 1D range still defined by the first interval
        self.xmin, self.xmax = xmin1, xmax1

        # Build a 1D mask as the union of [xmin1, xmax1) and [xmin2, xmax2)
        # over the current data length.
        try:
            y = getattr(self.data, "y", None)
            n = int(len(y)) if y is not None else 0
        except Exception:
            n = 0
        if n <= 0:
            # No data: clear mask but keep the primary range assignment.
            self.mask = None
            return

        lb1 = max(0, int(xmin1))
        ub1 = max(lb1, min(int(xmax1), n))
        lb2 = max(0, int(xmin2))
        ub2 = max(lb2, min(int(xmax2), n))
        mask = np.zeros(n, dtype=float)
        if ub1 > lb1:
            mask[lb1:ub1] = 1.0
        if ub2 > lb2:
            mask[lb2:ub2] = 1.0
        self.mask = mask

    @property
    def mask(self):
        """Optional 1D mask or weights applied to weighted residuals.

        If *mask* is boolean, False entries are excluded (residuals set to 0
        effectively via a multiplicative 0/1 weight). If numeric, values are
        treated as multiplicative weights on the residuals.
        """

        return getattr(self, "_mask", None)

    @mask.setter
    def mask(self, v):
        if v is None:
            self._mask = None
            return
        arr = np.asarray(v)
        if arr.ndim != 1:
            raise ValueError("fit.mask must be a 1D array or sequence")
        self._mask = arr

    @property
    def grad(self) -> np.array:
        """Approximate gradient of the residuals at current parameters.

        The gradient is computed numerically via :func:`approx_grad`.
        """
        _, grad = approx_grad(
            self.model.parameter_values,
            self,
            chisurf.settings.eps
        )
        return grad

    @property
    def covariance_matrix(self) -> typing.Tuple[np.array, typing.typing.List[int]]:
        """Approximate covariance matrix and indices of relevant parameters.

        The matrix is computed from numerical partial derivatives obtained
        via :func:`covariance_matrix`.
        """
        return covariance_matrix(self)

    @property
    def n_free(self) -> int:
        """Number of free (non-fixed, non-linked) parameters in the model."""
        return self.model.n_free


    def __init__(
            self,
            model_class: typing.Type[chisurf.models.Model] = type,
            data: chisurf.data.DataCurve = None,
            xmin: int = 0,
            xmax: int = 0,
            model_kw: typing.Dict = None,
            group: list = None,
            **kwargs
    ):
        """Create a :class:`Fit` with a given model class and data.

        Parameters
        ----------
        model_class : type
            Model class to instantiate, typically a subclass of
            :class:`chisurf.models.ModelCurve`.
        data : chisurf.data.DataCurve, optional
            Data to be fitted. If omitted, a dummy ramp is used.
        xmin, xmax : int, optional
            Initial fitting range indices.
        model_kw : dict, optional
            Keyword arguments forwarded to the model constructor.
        group : list, optional
            Optional list used to collect multiple :class:`Fit` instances
            into a :class:`FitGroup`.
        """
        super().__init__(**kwargs)
        self._model: chisurf.models.Model = None
        self._result_current = 0
        self.results = deque(maxlen=500)
        self._mask = None
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
        # Store the group reference before creating the model
        if isinstance(group, list):
            self.group = group
            self.group.append(self)
        self.model = model_class

    def __getstate__(self):
        d = super().__getstate__()

        # Try to pickle model; remove unpickleable attributes
        model_state = self.model.__getstate__()
        import pickle
        # Attempt to pickle each attribute
        for key in list(model_state.keys()):  # Use list to avoid modifying dict while iterating
            try:
                pickle.dumps(model_state[key])  # Try pickling the attribute
            except (pickle.PicklingError, TypeError):
                # Remove unpickleable attributes
                del model_state[key]

        d['data'] = self.data.__getstate__()
        d['model'] = model_state
        return d

    def __setstate__(self, state):
        m = state.pop('model')
        d = state.pop('data')
        self.model.__setstate__(m)
        self.data.__setstate__(d)
        super().__init__(**state)
        self.model.finalize()
        self.update()

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of this fit's model state.

        By default this delegates to :meth:`model.get_state` so that the
        underlying model (and any nested parameter groups) control how their
        state is serialized.
        """

        model = getattr(self, "model", None)
        get_state = getattr(model, "get_state", None)
        if callable(get_state):
            try:
                return get_state()
            except Exception:
                return {}
        return {}

    def set_state(self, state: dict) -> None:
        """Restore model state from :meth:`get_state` output.

        The provided ``state`` must be a dictionary produced by
        :meth:`get_state` above. This updates parameter values, bounds,
        fixed flags, links and any registered model-specific extras (e.g.
        TCSPC IRF/linearization) but does not change the data object.
        """

        if not isinstance(state, dict):
            return
        model = getattr(self, "model", None)
        set_state = getattr(model, "set_state", None)
        if callable(set_state):
            try:
                set_state(state)
            except Exception:
                return
        # After restoring the internal model/parameter state, trigger a
        # standard fit update so that derived quantities, plots and any
        # GUI widgets stay in sync.
        try:
            self.update()
        except Exception:
            pass
        # Ask the model to finalize its parameter controllers so that all
        # FittingParameter widgets refresh from the restored values.
        try:
            model = getattr(self, "model", None)
            finalize = getattr(model, "finalize", None)
            if callable(finalize):
                finalize()
        except Exception:
            pass

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

    def get_curves(self, copy_curves: bool = False) -> typing.OrderedDict[str, chisurf.curve.Curve]:
        """Return a mapping of named curves associated with this fit.

        The dictionary typically contains entries for ``"model"``,
        ``"data"``, ``"weighted residuals"`` and ``"autocorrelation"``.
        """
        d = self.model.get_curves(
            copy_curves=copy_curves
        )
        d['data'] = self.data
        d['weighted residuals'] = self.weighted_residuals
        d['autocorrelation'] = self.autocorrelation
        return d

    def get_score(self, score_type: str = 'chi2'):
        """Return a scalar goodness-of-fit score.

        Parameters
        ----------
        score_type : {"chi2", "chi2r"}
            Select unreduced or reduced chi².
        """
        if score_type == 'chi2':
            return self.chi2
        elif score_type == 'chi2r':
            return self.chi2r

    def get_chi2(
            self,
            parameter=None,
            model: chisurf.models.Model = None,
            reduced: bool = True
    ) -> float:
        """Convenience wrapper around :func:`get_chi2` using this fit."""
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
        """Return weighted residuals for a model attached to this fit."""
        if model is None:
            model = self.model
        if parameter is not None:
            model.parameter_values = parameter
            model.update_model()
        wres = model.get_wres(self, **kwargs)
        return _apply_fit_mask(model, wres)

    def save(
            self,
            filename: str,
            file_type: str = 'csv',
            save_curves: bool = False,
            verbose: bool = False,
            **kwargs
    ) -> None:
        """Save fit metadata and, optionally, all associated curves."""
        super().save(
            filename=filename,
            file_type=file_type,
            verbose=verbose
        )
        if save_curves:
            curve_dict = self.get_curves()
            with open(filename+'_info.txt', mode='w') as fp:
                fp.write(str(self))
            for curve_key in curve_dict:
                curve = curve_dict[curve_key]
                curve_file_root = filename + "_%s" % curve_key
                curve.save(
                    filename=curve_file_root + '.' + file_type,
                    file_type=file_type
                )

    def run(self, *args, **kwargs) -> None:
        """Run a local least-squares optimization on this fit."""
        fitting_options = chisurf.settings.cs_settings['optimization']['leastsq']
        self.model.find_parameters(
            parameter_type=chisurf.fitting.parameter.FittingParameter
        )
        progress_callback = kwargs.get("progress_callback")
        cancelled = False
        try:
            chisurf.math.optimization.leastsqbound(
                get_wres,
                self.model.parameter_values,
                args=(self.model,),
                bounds=self.model.parameter_bounds,
                progress_callback=progress_callback,
                **fitting_options
            )
        except OptimizationCancelled:
            cancelled = True
        self._last_run_cancelled = cancelled
        self.update()
        if not cancelled:
            self.update_error_estimates()
            self.results.append(self.model.__getstate__())
        self.model.finalize()
        if cancelled:
            raise OptimizationCancelled()

    def set_result_idx(self, idx: int):
        idx = np.clip(idx, 0, len(self.results) - 1)
        self._result_current = idx
        self.model.__setstate__(self.results[idx])
        self.update()
        self.model.finalize()

    def next_result(self):
        self.set_result_idx(self._result_current + 1)

    def previous_result(self):
        self.set_result_idx(self._result_current - 1)

    def update_error_estimates(self):
        # Estimate errors based on gradient
        fit = self
        cov_m, used_parameters = fit.covariance_matrix
        err = np.sqrt(np.diag(cov_m))
        for p, e in zip(used_parameters, err):
            fit.model.parameters[p].error_estimate = e

    def update(self) -> None:
        self.model.update()

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
        :return: a list containing arrays of the chi2 and the parameter-values
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


class FitGroup(Fit):
    """Group of :class:`Fit` objects that share a global model.

    A :class:`FitGroup` manages multiple single-curve fits while exposing
    an aggregate model for global optimization.
    """

    @property
    def selected_fit(self) -> Fit:
        return self.grouped_fits[self.selected_fit_index]

    @property
    def selected_fit_index(self) -> int:
        return self._selected_fit_index

    @selected_fit.setter
    def selected_fit(self, v: int):
        self._selected_fit_index = v

    @property
    def data(self) -> chisurf.data.DataCurve:
        return self.selected_fit.data

    @data.setter
    def data(self, v: chisurf.base.Data):
        self.selected_fit.data = v

    @property
    def model(self) -> chisurf.models.Model:
        return self.selected_fit.model

    @model.setter
    def model(self, v: typing.Type[chisurf.models.Model]):
        self.selected_fit.model = v

    @property
    def weighted_residuals(self) -> chisurf.curve.Curve:
        return self.selected_fit.weighted_residuals

    @property
    def chi2r(self) -> float:
        return self.selected_fit.chi2r

    @property
    def durbin_watson(self) -> float:
        return chisurf.math.statistics.durbin_watson(
            self.weighted_residuals.y
        )

    @property
    def mask(self):
        """Optional global mask shared across all grouped fits.

        This forwards to the currently selected fit's mask for reading and
        propagates any assignment to all member fits, so that both the global
        model and individual fits see the same residual weights.
        """

        return getattr(self.selected_fit, "mask", None)

    @mask.setter
    def mask(self, v):
        for f in self:
            setattr(f, "mask", v)

    @property
    def fit_range(self) -> typing.Tuple[int, int]:
        return self.xmin, self.xmax

    @fit_range.setter
    def fit_range(self, v):
        vals = tuple(int(x) for x in v)
        if len(vals) == 2:
            # Backwards-compatible 1D range: propagate to all member fits.
            xmin, xmax = vals
            for f in self:
                f.xmin, f.xmax = xmin, xmax
            self.xmin, self.xmax = xmin, xmax

            # Initialize or rebuild simple 1D masks per fit matching the
            # common [xmin, xmax) range, clipped to each dataset length.
            for f in self:
                try:
                    y = getattr(f.data, "y", None)
                    n = int(len(y)) if y is not None else 0
                except Exception:
                    n = 0
                if n <= 0:
                    try:
                        f.mask = None
                    except Exception:
                        pass
                    continue
                lb1 = max(0, int(xmin))
                ub1 = max(lb1, min(int(xmax), n))
                mask = np.zeros(n, dtype=float)
                if ub1 > lb1:
                    mask[lb1:ub1] = 1.0
                try:
                    f.mask = mask
                except Exception:
                    pass
            return
        if len(vals) != 4:
            raise ValueError("FitGroup.fit_range must be a 2- or 4-tuple of integers")

        xmin1, xmax1, xmin2, xmax2 = vals

        # Primary 1D range still defined by the first interval; propagate to
        # all member fits and to the group itself.
        for f in self:
            f.xmin, f.xmax = xmin1, xmax1
        self.xmin, self.xmax = xmin1, xmax1

        # Build per-fit masks as the union of the two index intervals on
        # each fit's data length so that all residuals see consistent
        # weighting regardless of individual data sizes.
        for f in self:
            try:
                y = getattr(f.data, "y", None)
                n = int(len(y)) if y is not None else 0
            except Exception:
                n = 0
            if n <= 0:
                try:
                    f.mask = None
                except Exception:
                    pass
                continue
            lb1 = max(0, int(xmin1))
            ub1 = max(lb1, min(int(xmax1), n))
            lb2 = max(0, int(xmin2))
            ub2 = max(lb2, min(int(xmax2), n))
            mask = np.zeros(n, dtype=float)
            if ub1 > lb1:
                mask[lb1:ub1] = 1.0
            if ub2 > lb2:
                mask[lb2:ub2] = 1.0
            try:
                f.mask = mask
            except Exception:
                pass

    @property
    def xmin(self) -> int:
        return self.selected_fit.xmin

    @xmin.setter
    def xmin(self, v: int):
        for f in self:
            f.xmin = v

    @property
    def xmax(self) -> int:
        return self.selected_fit.xmax

    @xmax.setter
    def xmax(self, v: int):
        for f in self:
            f.xmax = v

    def get_curves(self, copy_curves: bool = False, idx: int = None) -> typing.OrderedDict[str, chisurf.curve.Curve]:
        """Return curves for one or all grouped fits.

        If ``idx`` is ``None``, curves from :meth:`super().get_curves` are
        returned. Otherwise curves are collected from the selected or all
        grouped fits, with keys suffixed by ``"_%02d"``.
        """
        curves = {}
        if idx is None:
            curves = super().get_curves()
        else:
            if idx >= 0:
                fit = self.grouped_fits[idx]
                curves = fit.get_curves()
            else:
                for i, f in enumerate(self.grouped_fits):
                    fit_curves = f.get_curves(copy_curves)
                    for curve_key in fit_curves:
                        new_curve_key = curve_key + "_%02d" % i
                        curves[new_curve_key] = fit_curves[curve_key]
        return curves

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
        """Finalize the global model and all grouped fits."""
        self.update()
        self._model.finalize()

    def update(self) -> None:
        for f in self.grouped_fits:
            f.update()

    def run(self, local_first: bool = None, **kwargs):
        """Run local fits followed by a global least-squares optimization."""
        fit: FitGroup = self
        if local_first is None:
            local_first = chisurf.settings.optimization['global_optimize_local_first']
        cancelled = False
        try:
            if local_first:
                for f in fit:
                    f.run(**kwargs)
            for f in fit:
                f.model.find_parameters()
            fit._model.find_parameters()
            fitting_options = chisurf.settings.optimization['leastsq']
            bounds = [pi.bounds for pi in fit._model.parameters]
            progress_callback = kwargs.get("progress_callback")
            chisurf.math.optimization.leastsqbound(
                func=get_wres,
                x0=fit._model.parameter_values,
                args=(fit._model,),
                bounds=bounds,
                progress_callback=progress_callback,
                **fitting_options
            )
        except OptimizationCancelled:
            cancelled = True
        self._last_run_cancelled = cancelled
        self.update()
        if not cancelled:
            self.update_error_estimates()
            self.results.append(self.model.__getstate__())
        if cancelled:
            raise OptimizationCancelled()

    def __init__(
            self,
            data: chisurf.data.DataGroup,
            model_class: typing.Type[chisurf.models.Model] = type,
            model_kw: typing.Dict = None
    ):
        """Create a :class:`FitGroup` over a :class:`DataGroup`.

        One :class:`Fit` instance is created per entry in ``data`` and
        collected into ``grouped_fits``. A global model is then constructed
        from these.
        """
        self._selected_fit_index = 0
        self.grouped_fits = list()

        for d in data:
            if model_kw is None:
                model_kw = dict()
            fit = Fit(
                model_class=model_class,
                data=d,
                model_kw=model_kw,
                group=self.grouped_fits
            )

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

    def __len__(self):
        return len(self.grouped_fits)

    def __getitem__(self, key) -> typing.List[Fit]:
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
    """Sample free parameters of a fit and save the chain to disk.

    Parameters
    ----------
    fit : Fit
        Fit whose parameters should be sampled.
    filename : str
        Output file stem for the chain; run indices are appended.
    method : {"emcee", "mcmc"}, optional
        Sampling backend to use.
    steps, thin, chi2max, n_runs, step_size, temp : float or int, optional
        Sampling configuration passed through to
        :mod:`chisurf.fitting.sample`.
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
                progress_bar=chisurf.cs.progress_bar,
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
    """Approximate gradient of the weighted residuals with respect to ``xk``.

    Parameters
    ----------
    xk : array_like
        Parameter values around which the gradient is estimated.
    fit : Fit
        Fit providing :meth:`Fit.get_wres` and a model.
    epsilon : float
        Differential step size for the finite-difference approximation.
    args : tuple, optional
        Additional positional arguments forwarded to ``fit.get_wres``.
    f0 : array_like, optional
        Pre-computed weighted residuals at ``xk``.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Tuple ``(f0, grad)`` where ``grad`` has shape
        ``(len(xk), len(f0))``.
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
        epsilon: float = 1e-12, #chisurf.settings.eps,
        **kwargs
) -> typing.Tuple[np.array, typing.List[int]]:
    """Estimate the covariance matrix of the fit parameters.

    Parameters
    ----------
    fit : Fit
        The fit whose model and residuals are used.
    epsilon : float, optional
        Step size for the numerical gradient.

    Returns
    -------
    cov_m : numpy.ndarray
        Approximate covariance matrix of important parameters.
    important_parameters : list of int
        Indices of parameters whose partial derivatives are non-zero.
    """
    model = fit.model
    xk = np.array(model.parameter_values)
    fi_v, partial_derivatives = approx_grad(xk, fit, epsilon)

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
    except:
        cov_m = np.zeros_like(
            (n_important_parameters, n_important_parameters),
            dtype=float
        )
    return cov_m, important_parameters


def _apply_fit_mask(
        model: chisurf.models.Model,
        wres: np.array
) -> np.array:
    """Apply an optional Fit-level mask to a residual vector.

    The mask is taken from ``model.fit.mask`` if available. Boolean masks are
    interpreted as 0/1 inclusion weights; numeric masks are used as
    multiplicative weights. The mask is truncated to the residual length.
    """

    if wres is None:
        return wres

    fit = getattr(model, "fit", None)
    if fit is None:
        return wres

    mask = getattr(fit, "mask", None)
    if mask is None:
        return wres

    try:
        m = np.asarray(mask, dtype=float).ravel()
    except Exception:
        return wres
    if m.ndim != 1 or m.size == 0:
        return wres

    try:
        xmin = int(getattr(fit, "xmin", 0))
        xmax = int(getattr(fit, "xmax", xmin + int(len(wres))))
    except Exception:
        return wres

    if xmax < xmin:
        xmin, xmax = xmax, xmin

    xmin = max(0, xmin)
    xmax = min(m.size, max(xmin, xmax))
    window_len = max(0, xmax - xmin)
    if window_len == 0 or len(wres) == 0:
        return wres

    n = len(wres)
    if window_len != n:
        return wres

    wres = np.array(wres, copy=True)
    w_slice = m[xmin:xmax]
    wres *= w_slice
    return wres


def get_wres(
        parameter_values: typing.List[float],
        model: chisurf.models.Model
) -> np.array:
    """Return weighted residuals for a list of model parameters.

    Parameters
    ----------
    parameter_values : list of float
        Parameter values to assign before computing residuals. If the list
        is empty, the model is not updated.
    model : chisurf.models.Model
        Model providing :attr:`weighted_residuals`.
    """
    if len(parameter_values) > 0:
        model.parameter_values = parameter_values
        model.update_model()
    wres = model.weighted_residuals
    return _apply_fit_mask(model, wres)


def get_chi2(
        parameter_values: typing.List[float],
        model: chisurf.models.model.ModelCurve,
        reduced: bool = True
) -> float:
    """Return either the reduced chi² or the sum of squares (chi²).

    Parameters
    ----------
    parameter_values : list of float
        Parameter values to apply before computing residuals. If the list
        is empty, the model is not updated.
    model : chisurf.models.ModelCurve
        Model providing :attr:`weighted_residuals`, :attr:`n_points` and
        :attr:`n_free`.
    reduced : bool, optional
        If *True*, return the reduced chi², i.e. chi² divided by
        ``(n_points - n_free - 1)``.

    Examples
    --------
    Use a tiny dummy model with three residuals:

    >>> import numpy as np
    >>> class _DummyModel:
    ...     def __init__(self):
    ...         self._wres = np.array([1.0, -1.0, 0.0])
    ...         self.n_points = self._wres.size
    ...         self.n_free = 1
    ...     @property
    ...     def weighted_residuals(self):
    ...         return self._wres
    ...     @property
    ...     def parameter_values(self):
    ...         return []
    ...     @parameter_values.setter
    ...     def parameter_values(self, v):
    ...         pass
    ...     def update_model(self):
    ...         pass
    >>> m = _DummyModel()
    >>> round(get_chi2([], m, reduced=False), 1)
    2.0
    >>> round(get_chi2([], m, reduced=True), 1)
    2.0
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
    """Log-prior probability induced by parameter bounds.

    The prior is uniform inside the bounds and zero outside. This function
    returns ``0`` inside the allowed region and ``-inf`` if any parameter
    violates its bounds.

    Parameters
    ----------
    parameter_values : list of float
        Parameter values to be tested.
    fit : Fit
        Fit providing default bounds via ``fit.model.parameter_bounds`` if
        ``bounds`` is *None*.
    bounds : list of (float, float), optional
        Explicit bounds to use instead of those from ``fit``.

    Examples
    --------
    >>> bounds = [(0.0, 2.0), (None, 1.0)]
    >>> round(lnprior([1.0, 0.5], fit=None, bounds=bounds), 1)
    0.0
    >>> lnprior([3.0, 0.5], fit=None, bounds=bounds)
    -inf
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
    """Log-posterior probability for use in MCMC sampling.

    The posterior is given by ``lnprior + lnlikelihood`` where the
    likelihood is assumed to be Gaussian in the residuals, i.e.

    ``lnlikelihood = -0.5 * chi2``

    and ``chi2`` is obtained from :func:`get_chi2`.

    Parameters
    ----------
    parameter_values : list of float
        Parameter values at which to evaluate the posterior.
    fit : Fit
        Fit providing the model and default bounds.
    chi2max : float, optional
        Hard cutoff on chi²; values above this threshold return ``-inf``.
    bounds : list of (float, float), optional
        Explicit bounds to use for the prior.

    Examples
    --------
    >>> import numpy as np
    >>> class _DummyModel:
    ...     def __init__(self):
    ...         self._wres = np.array([1.0, -1.0, 0.0])
    ...         self.n_points = self._wres.size
    ...         self.n_free = 1
    ...     @property
    ...     def weighted_residuals(self):
    ...         return self._wres
    ...     @property
    ...     def parameter_values(self):
    ...         return []
    ...     @parameter_values.setter
    ...     def parameter_values(self, v):
    ...         pass
    ...     def update_model(self):
    ...         pass
    >>> class _DummyFit:
    ...     def __init__(self):
    ...         self.model = _DummyModel()
    >>> fit = _DummyFit()
    >>> bounds = [(0.0, 2.0)]
    >>> val = lnprob([1.0], fit, chi2max=10.0, bounds=bounds)
    >>> isinstance(val, float) and np.isfinite(val)
    True
    >>> lnprob([10.0], fit, chi2max=10.0, bounds=bounds)
    -inf
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


