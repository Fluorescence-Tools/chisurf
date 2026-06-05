from __future__ import annotations

import abc

from chisurf import typing

from collections import OrderedDict

import numpy as np

import chisurf.core.parameter
import chisurf.core.curve

from chisurf.core.fitting.parameter import FittingParameterGroup


class Model(FittingParameterGroup):
    """Abstract base class for all ChiSurf models.

    A model wraps the relationship between a set of fitting parameters and
    one or more model curves. Subclasses typically implement
    :meth:`update_model` to compute ``self.y`` (and optionally auxiliary
    state) from the current parameter values and experimental data
    referenced via ``self.fit``.
    """

    name = "Model name not available"

    @property
    def n_free(self) -> int:
        """Number of free (non-linked or fixed) fitting parameters.

        This is a thin wrapper around :attr:`parameters` provided by
        :class:`chisurf.core.fitting.parameter.FittingParameterGroup`.
        """
        return len(self.parameters)

    @property
    def weighted_residuals(self) -> np.ndarray:
        """Return weighted residuals for the current fit window.

        The residuals are computed via
        :func:`chisurf.core.fitting.calculate_weighted_residuals` using
        ``self.fit.data`` and ``self.fit.model`` between ``xmin`` and
        ``xmax`` as defined on the associated :class:`Fit` instance.
        """
        return self.get_wres(
            self.fit,
            xmin=self.fit.xmin,
            xmax=self.fit.xmax
        )

    @abc.abstractmethod
    def update_model(self, **kwargs):
        """Update the internal model state from the current parameters.

        Subclasses must implement this method and usually:

        - read experimental data from ``self.fit``;
        - compute the model curve ``self.y`` (and corresponding ``self.x``
          if necessary);
        - update any cached state used by the GUI or other components.
        """
        pass

    @abc.abstractmethod
    def update(self, **kwargs) -> None:
        """High-level update hook called by the fitting machinery.

        The default implementation refreshes all nested
        :class:`FittingParameterGroup` instances and then calls
        :meth:`update_model`. Subclasses may extend this method but should
        usually call ``super().update()``.
        """
        self.find_parameters()

        # Update ParameterGroups
        d = [v for v in self.__dict__.values() if v is not self]
        pgs = chisurf.core.base.find_objects(
            search_iterable=d,
            searched_object_type=chisurf.core.fitting.parameter.FittingParameterGroup
        )
        for pg in pgs:
            try:
                pg.update()
            except Exception as e:
                import logging
                logging.warning(f"Failed to update parameter group {pg}: {e}")
                continue

        self.update_model()

    def finalize(self) -> None:
        """Propagate finalization to all parameter groups.

        This ensures that UI controllers and other secondary outputs
        linked to model components are refreshed after updates.
        """
        import chisurf.core.base
        import chisurf.core.fitting.parameter
        
        # 1. Finalize ourselves (Model is a FittingParameterGroup)
        super().finalize()
        
        # 2. Finalize all sub-groups discovered via attributes
        d = [v for v in self.__dict__.values() if v is not self]
        pgs = chisurf.core.base.find_objects(
            search_iterable=d,
            searched_object_type=chisurf.core.fitting.parameter.FittingParameterGroup
        )
        for pg in set(pgs):
            if hasattr(pg, "finalize") and pg is not self:
                try:
                    pg.finalize()
                except Exception as e:
                    import chisurf.logging
                    chisurf.logging.error(f"Failed to finalize parameter group: {e}")

    def get_wres(
            self,
            fit: chisurf.core.fitting.fit.Fit,
            xmin: int = None,
            xmax: int = None
    ) -> np.ndarray:
        """Compute weighted residuals for a given :class:`Fit` instance.

        Parameters
        ----------
        fit : chisurf.core.fitting.fit.Fit
            Fit object providing data and the associated model.
        xmin, xmax : int, optional
            Index range over which to compute residuals. If omitted, the
            ``xmin`` / ``xmax`` attributes of ``fit`` are used.

        Returns
        -------
        numpy.ndarray
            Weighted residuals for the selected window.
        """
        if xmin is None:
            xmin = fit.xmin
        if xmax is None:
            xmax = fit.xmax
        return chisurf.core.fitting.calculate_weighted_residuals(
            fit.data,
            fit.model,
            xmin=xmin,
            xmax=xmax
        )

    def __init__(self, fit: chisurf.core.fitting.fit.Fit, model_number: int = 0, **kwargs):
        """Create a new model instance.

        Parameters
        ----------
        fit : chisurf.core.fitting.fit.Fit
            Fit object this model is attached to.
        model_number : int, optional
            Index of the model within the fit (used for multi-model fits).
        """
        # Set model to none otherwise will result in self reference
        super().__init__(model=None, **kwargs)
        self.fit = fit
        self.flatten_weighted_residuals = True
        self.model_number = model_number

    def __getstate__(self):
        """Return picklable state for the model.

        The implementation currently delegates to
        :class:`FittingParameterGroup` and may be extended in subclasses.
        """
        state = super().__getstate__()
        return state
    
    def __setstate__(self, state):
        """Restore model state from a pickled representation.

        Parameter states are mapped back onto the corresponding
        :class:`~chisurf.core.fitting.parameter.FittingParameter` instances
        contained in :attr:`parameters_all_dict`.
        """
        super().__setstate__(state)
        model = self
        for key in state:
            if key in model.parameters_all_dict.keys():
                target = model.parameters_all_dict.get(key)
                target.__setstate__(state[key])

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of this model's state.

        This is the default mechanism used by project save/load for all
        models (TCSPC, FCS, PDA, etc.). The heavy lifting is implemented in
        :mod:`chisurf.core.project.fit_state` so that the same logic can be used
        both from :class:`Fit` and directly from models. Subclasses may
        override this for custom behaviour but should generally extend it.
        """

        try:
            from chisurf.core.project import fit_state as _fit_state
        except Exception:
            return {}

        # Prefer a model-centric helper if available to avoid depending on
        # a fully constructed Fit instance.
        try:
            model_to_state = getattr(_fit_state, "_model_to_state", None)
            if callable(model_to_state):
                return model_to_state(self)
        except Exception:
            pass

        # Fallback: use the public fit-based API if we have a Fit attached.
        fit = getattr(self, "fit", None)
        if fit is None:
            return {}
        try:
            return _fit_state.fit_to_state(fit)
        except Exception:
            return {}

    def set_state(self, state: dict) -> None:
        """Restore model state from :meth:`get_state` output.

        The input must be a dictionary produced by :meth:`get_state` (or
        the corresponding :func:`chisurf.core.project.fit_state.fit_to_state`).
        Structural elements such as the number of lifetime or Gaussian
        components are applied before scalar parameters and links.
        """

        if not isinstance(state, dict):
            return

        try:
            from chisurf.core.project import fit_state as _fit_state
        except Exception:
            return

        # Prefer a model-centric helper if available.
        try:
            apply_model = getattr(_fit_state, "_apply_state_to_model", None)
            if callable(apply_model):
                apply_model(self, state)
                return
        except Exception:
            pass

        # Fallback: use the public fit-based API if we have a Fit.
        fit = getattr(self, "fit", None)
        if fit is None:
            return
        try:
            _fit_state.apply_state_to_fit(fit, state)
        except Exception:
            return

    def __str__(self):
        """Return a readable summary of model parameters with uncertainty and links."""
        s = f"Model: {self.name}\n\n"
        s += f"  {'Name':<12s}  {'Value':<11s}  {'Error':<13s}  {'Source'}  {'Link'}\n"
        pd = self.parameters_all_dict
        for k in sorted(pd.keys()):
            p = pd[k]
            if not isinstance(p, chisurf.core.fitting.parameter.FittingParameter):
                continue
            if getattr(p, 'is_output', False):
                continue
            val = f"{p.value:.5g}"
            if p.fixed:
                s += f"  {p.name:<12s}  {val:<11s}  fixed"
            else:
                try:
                    ee = p.error_estimate
                    if isinstance(ee, float):
                        rel = abs(ee / (p.value + 1e-12) * 100.0)
                        err = f"±{ee:.3g}({rel:.1f}%)"
                        src = "sp" if p.scan_result is not None else "cov"
                    else:
                        err = "±N/A"
                        src = ""
                except Exception:
                    err = "±N/A"
                    src = ""
                s += f"  {p.name:<12s}  {val:<11s}  {err:<13s}  {src:<6s}"
            if p.is_linked and p.link is not None:
                s += f"  →{p.link.name}"
            s += "\n"
        return s


class ModelCurve(Model, chisurf.core.curve.Curve):
    """Base class for models represented by a single 1D curve.

    This mixin combines :class:`Model` with :class:`chisurf.core.curve.Curve`
    and provides convenience accessors for the x/y data arrays along with
    helpers for extracting sub-curves.
    """

    @property
    def n_points(self) -> int:
        """Number of data points effectively contributing to chi².

        This counts the unmasked points inside the current fit window
        ``[xmin, xmax)`` based on the Fit-level 1D ``mask``. When no
        valid mask is present, it falls back to the plain window
        length ``xmax - xmin``.
        """
        fit = getattr(self, "fit", None)
        if fit is None:
            return 0

        try:
            xmin = int(getattr(fit, "xmin", 0))
            xmax = int(getattr(fit, "xmax", 0))
        except Exception:
            return 0

        if xmax < xmin:
            xmin, xmax = xmax, xmin

        mask = getattr(fit, "mask", None)
        if mask is not None:
            try:
                m = np.asarray(mask)
                if m.ndim == 1 and m.size > 0:
                    n = m.size
                    lb = max(0, xmin)
                    ub = max(lb, min(xmax, n))
                    if ub > lb:
                        sub = m[lb:ub]
                        count = int(np.count_nonzero(sub))
                        if count > 0:
                            return count
            except Exception:
                pass

        return max(int(xmax - xmin), 0)

    @property
    def x(self) -> np.ndarray:
        """Abscissa array of the model curve."""
        return self.__dict__['d'][0]

    @x.setter
    def x(self, v: np.ndarray):
        """Set the abscissa array of the model curve."""
        self.__dict__['d'][0] = v

    @property
    def y(self) -> np.array:
        """Ordinate array of the model curve."""
        return self.__dict__['d'][1]

    @y.setter
    def y(self, v: np.ndarray):
        """Set the ordinate array of the model curve."""
        self.__dict__['d'][1] = v

    def __init__(self, fit: chisurf.core.fitting.fit.Fit, *args, **kwargs):
        """Create a new curve-based model attached to ``fit``.

        The initial x-grid is taken from ``fit.data.x`` (if present) and
        the model values ``y`` are initialized to zeros.
        """
        super().__init__(fit, *args, **kwargs)
        if fit.data.x is None:
            x = np.array([], dtype=np.float64)
        else:
            x = fit.data.x
        chisurf.core.curve.Curve.__init__(
            self,
            x=x, y=np.zeros_like(x),
            *args,
            **kwargs
        )

    def get_curves(self, copy_curves: bool = False) -> typing.OrderedDict[str, chisurf.core.curve.Curve]:
        """Return a mapping of named curves produced by this model.

        Currently a single entry ``"model"`` is provided, containing the
        full model curve as a :class:`chisurf.core.curve.Curve` instance.
        """
        #xmin = self.fit.xmin
        #xmax = self.fit.xmax
        d = OrderedDict()
        #d['model'] = chisurf.core.curve.Curve(x=self.x[xmin:xmax], y=self.y[xmin:xmax], copy_array=copy_curves)
        d['model'] = chisurf.core.curve.Curve(x=self.x, y=self.y, copy_array=copy_curves)
        return d

    def __getitem__(self, key) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Return a slice of the model curve as ``(x, y)``.

        Parameters
        ----------
        key : slice
            Slice object specifying start/stop/step along the x-axis.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            Tuple of sliced ``(x, y)`` arrays.
        """
        start = key.start
        stop = key.stop
        step = 1 if key.step is None else key.step
        x, y = self.x[start:stop:step], self.y[start:stop:step]
        return x, y
