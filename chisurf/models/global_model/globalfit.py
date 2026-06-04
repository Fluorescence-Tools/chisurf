from __future__ import annotations
from chisurf import typing

import threading
import numpy as np
from typing import TYPE_CHECKING

import chisurf.decorators
import chisurf.parameter

from chisurf.curve import Curve
from chisurf.models import model

if TYPE_CHECKING:
    from chisurf.fitting.fit import Fit, FitGroup


class GlobalFitModel(model.Model, Curve):

    name = "Global fit"

    @property
    def weighted_residuals(self) -> np.ndarray:
        """Concatenated weighted residuals from all local fits.

        Returns
        -------
        np.ndarray
            1-D array of weighted residuals, or empty.
        """
        if len(self.fits) > 0:
            re = list()
            for f in self.fits:
                re.append(f.model.weighted_residuals.flatten())
            return np.concatenate(re)
        else:
            return np.array([], dtype=np.float64)

    @property
    def fit_names(self) -> typing.List[str]:
        """Names of all local fits in this global model."""
        return [f.name for f in self.fits]

    @property
    def links(self) -> typing.List[chisurf.fitting.parameter.FittingParameter]:
        """List of link definitions between local-fit parameters."""
        return self._links

    @links.setter
    def links(self, v: typing.List[chisurf.fitting.parameter.FittingParameter]):
        """Set the list of link definitions.

        Parameters
        ----------
        v : list
            List of link tuples ``(enabled, fit_idx, param_name, formula)``.
        """
        self._links = v if isinstance(v, list) else list()

    @property
    def n_points(self) -> int:
        """Total number of data points across all local fits."""
        nbr_points = 0
        for f in self.fits:
            nbr_points += f.model.n_points
        return nbr_points

    @property
    def global_parameters_all(self) -> typing.List[chisurf.fitting.parameter.FittingParameter]:
        """All global parameters (fixed and variable)."""
        return list(self._global_parameters.values())

    @property
    def global_parameters_all_names(self) -> typing.List[str]:
        """Names of all global parameters."""
        return [p.name for p in self.global_parameters_all]

    @property
    def global_parameters(self) -> typing.List[chisurf.fitting.parameter.FittingParameter]:
        """Non-fixed (variable) global parameters."""
        return [p for p in self.global_parameters_all if not p.fixed]

    @property
    def global_parameters_names(self) -> typing.List[str]:
        """Names of variable global parameters."""
        return [p.name for p in self.global_parameters]

    @property
    def global_parameters_bound_all(self) -> typing.List[typing.Tuple[float, float]]:
        """Bounds for all global parameters."""
        return [pi.bounds for pi in self.global_parameters_all]

    @property
    def global_parameter_linked_all(self) -> typing.List[bool]:
        """Whether each global parameter is linked."""
        return [p.is_linked for p in self.global_parameters_all]

    @property
    def parameters(self) -> typing.List[chisurf.fitting.parameter.FittingParameter]:
        """All fitting parameters (local variable + global variable)."""
        p = list()
        for f in self.fits:
            p += f.model.parameters
        p += self.global_parameters
        return p

    @property
    def parameter_names(self) -> typing.List[str]:
        """Formatted names of variable parameters across all local fits.

        Each local parameter is prefixed with its fit index, e.g. ``1:N``.
        """
        try:
            re = list()
            for i, f in enumerate(self.fits):
                if f.model is not None:
                    re += ["%i:%s" % (i + 1, p.name) for p in f.model.parameters]
            re += self.global_parameters_names
            return re
        except AttributeError:
            return list()

    @property
    def parameters_all(self) -> typing.List[chisurf.fitting.parameter.FittingParameter]:
        """All parameters (local + global), including fixed ones."""
        try:
            re = list()
            for f in self.fits:
                if f.model is not None:
                    re += [p for p in f.model.parameters_all]
            re += self.global_parameters_all
            return re
        except AttributeError:
            return []

    @property
    def global_parameters_values_all(self) -> typing.List[float]:
        """Current values of all global parameters."""
        return [g.value for g in self.global_parameters_all]

    @property
    def global_parameters_fixed_all(self) -> typing.List[bool]:
        """Fixed-state of all global parameters."""
        return [p.fixed for p in self.global_parameters_all]

    @property
    def parameter_names_all(self) -> typing.List[str]:
        """Formatted names of all parameters (local + global), including fixed."""
        try:
            re = list()
            for i, f in enumerate(self.fits):
                if f.model is not None:
                    re += ["%i:%s" % (i + 1, p.name) for p in f.model._parameters]
            re += self.global_parameters_all_names
            return re
        except AttributeError:
            return []

    @property
    def parameter_dict(self) -> typing.Dict[str, chisurf.fitting.parameter.FittingParameter]:
        """Dictionary mapping formatted parameter names to parameters."""
        re = dict()
        for i, f in enumerate(self.fits):
            d = f.model.parameter_dict
            k = [str(i+1)+":"+dk for dk in d.keys()]
            for j, di in enumerate(d.keys()):
                re[k[j]] = d[di]
        return re

    @property
    def data(self) -> typing.Tuple[np.array, np.array, np.array]:
        """Concatenated (x, y, weight) data from all local fits.

        Returns
        -------
        tuple of np.array
            ``(x, y, weights)`` where x is a running index.
        """
        d = list()
        w = list()
        for f in self.fits:
            x, di, wi = f.data[0:-1]
            d.append(di)
            w.append(wi)
        dn = np.hstack(d)
        wn = np.hstack(w)
        xn = np.arange(0, dn.shape[0], 1)
        return xn, dn, wn

    def __init__(
            self,
            fit: Fit,
            fits: typing.List[Fit] = None,
            *args,
            **kwargs
    ):
        """Initialize the global fit model.

        Parameters
        ----------
        fit : Fit
            The parent fit (FitGroup or similar).
        fits : list of Fit, optional
            Initial list of local fits.
        *args
            Positional arguments forwarded to the base class.
        **kwargs
            Keyword arguments forwarded to the base class.
        """
        if fits is None:
            fits = list()
        self.fits = fits
        self.fit = fit
        self._global_parameters = dict()
        self.parameters_calculated = list()
        self._links = list()
        super().__init__(fit, *args, **kwargs)


    def get_wres(
            self,
            fit: Fit,
            xmin: int = None,
            xmax: int = None
    ) -> np.array:
        """Compute weighted residuals for a given fit within a range.

        Parameters
        ----------
        fit : Fit
            The local fit to evaluate.
        xmin, xmax : int, optional
            Index range for the residuals.

        Returns
        -------
        np.ndarray
            Weighted residuals array.
        """
        try:
            f = fit
            if xmin is None:
                xmin = f.xmin
            if xmax is None:
                xmax = f.xmax
            x, m = f.model[xmin:xmax]
            x, d, w = f.model.data[xmin:xmax]
            ml = min([len(m), len(d)])
            wr = np.array((d[:ml] - m[:ml]) * w[:ml], dtype=np.float64)
        except Exception as e:
            import logging
            logging.warning(f"Failed to calculate weighted residuals: {e}")
            wr = np.array([1.0])
        return wr

    def append_fit(self, fit: Fit) -> None:
        """Add a local fit to the global model.

        Parameters
        ----------
        fit : Fit
            The fit instance to append.
        """
        try:
            import chisurf
            chisurf.logging.info(
                f"GlobalFitModel.append_fit: receiver={type(self).__name__}, incoming fit type={type(fit).__name__}, name={getattr(fit, 'name', None)}; already_present={fit in getattr(self, 'fits', [])}"
            )
        except Exception:
            pass
        if fit not in self.fits:
            self.fits.append(fit)
            try:
                import chisurf
                chisurf.logging.info(
                    f"GlobalFitModel.append_fit: appended successfully; total_fits={len(self.fits)}; names={getattr(self, 'fit_names', [])}"
                )
            except Exception:
                pass
            # Notify any subscribers that a fit was appended (non-Qt callbacks)
            try:
                callbacks = getattr(self, "_on_fit_appended", None)
                if isinstance(callbacks, list):
                    for cb in list(callbacks):
                        try:
                            cb(fit)
                        except Exception:
                            # Keep notifications best-effort; ignore callback errors
                            pass
            except Exception:
                pass

    # --- Lightweight non-Qt subscription API for append notifications ---
    def on_fit_appended(self, fn) -> None:
        """Register a callback called with (fit) whenever a new fit is appended.

        This keeps the model free of Qt dependencies while allowing UI layers
        to react to changes triggered via macros/actions as well as the GUI.

        Parameters
        ----------
        fn : callable
            Callback accepting one argument (fit).
        """
        lst = getattr(self, "_on_fit_appended", None)
        if lst is None:
            lst = []
            setattr(self, "_on_fit_appended", lst)
        if callable(fn) and fn not in lst:
            lst.append(fn)

    def off_fit_appended(self, fn) -> None:
        """Unregister a callback previously registered with :meth:`on_fit_appended`.

        Parameters
        ----------
        fn : callable
            The callback to remove.
        """
        lst = getattr(self, "_on_fit_appended", None)
        if isinstance(lst, list) and fn in lst:
            lst.remove(fn)

    def append_global_parameter(self, parameter: chisurf.parameter.Parameter) -> None:
        """Add a global parameter to the model.

        Parameters
        ----------
        parameter : chisurf.parameter.Parameter
            The parameter instance to add.
        """
        variable_name = parameter.name
        if variable_name not in list(self._global_parameters.keys()):
            self._global_parameters[parameter.name] = parameter

    def setLinks(self):
        """Evaluate link formulas and set up parameter links."""
        self.parameters_calculated = list()
        if self.clear_on_update:
            self.clear_all_links()
        f = [fit.model.parameters_all_dict for fit in self.fits]
        g = self._global_parameters
        for link in self.links:
            en, origin_fit, origin_name, formula = link
            if not en:
                continue
            try:
                origin_parameter = f[origin_fit][origin_name]
                target = eval(str(formula), {"__builtins__": {}}, {"f": f, "g": g})
                if not isinstance(target, chisurf.parameter.Parameter):
                    chisurf.logging.warning("Global link formula did not resolve to a Parameter: %r" % (formula,))
                    continue
                origin_parameter.link = target
                print("f[%s][%s] linked to %s" % (origin_fit, origin_parameter.name, target.name))
            except IndexError:
                print("not enough fits index out of range")

    def autofitrange(self, fit: FitGroup):
        """Reset auto-fit range to cover all data.

        Parameters
        ----------
        fit : FitGroup
            Ignored (kept for API compatibility).

        Returns
        -------
        tuple of None
            ``(None, None)``.
        """
        self.xmin, self.xmax = None, None
        return self.xmin, self.xmax

    def clear_local_fits(self) -> None:
        """Remove all local fits from the global model."""
        self.fits = list()

    def remove_local_fit(self, fit_index: int):
        """Remove a local fit by index.

        Parameters
        ----------
        fit_index : int
            Index of the fit to remove.
        """
        del self.fits[fit_index]

    def clear_all_links(self) -> None:
        """Unlink all parameters in all local fits."""
        for fit in self.fits:
            for p in fit.model.parameters_all:
                p.link = None

    def clear_listed_links(self):
        """Clear the link definition list."""
        self.links = list()

    def __str__(self):
        """Return a string summary of the global model."""
        s = "\n"
        s += "Model: Global-fit\n"
        s += "Global-parameters:"
        p0 = list(zip(self.global_parameters_all_names, self.global_parameters_values_all,
                 self.global_parameters_bound_all, self.global_parameters_fixed_all,
                 self.global_parameter_linked_all))
        s += "Parameter \t Value \t Bounds \t Fixed \t Linked\n"
        for p in p0:
            s += "%s \t %.4f \t %s \t %s \t %s \n" % p
        for fit in self.fits:
            s += "\n"
            s += fit.name + "\n"
            s += str(fit.model) + "\n"
        s += "\n"
        return s

    @property
    def x(self) -> np.array:
        """x-data from all local fits, one array per fit."""
        x = list()
        for f in self.fits:
            x.append(f.model.x)
        return np.array(x)

    @x.setter
    def x(self, v):
        """Set x-data (no-op, data come from local fits)."""
        pass

    @property
    def y(self) -> np.array:
        """y-data from all local fits, one array per fit."""
        y = list()
        for f in self.fits:
            y.append(f.model.y)
        return np.array(y)

    @y.setter
    def y(self, v):
        """Set y-data (no-op, data come from local fits)."""
        pass

    def __getitem__(self, key):
        """Slice data from all local fits.

        Parameters
        ----------
        key : slice
            Slice object with start/stop/step.

        Returns
        -------
        tuple
            ``(x_slice, y_slice)``.
        """
        start = key.start
        stop = key.stop
        step = 1 if key.step is None else key.step
        return self.x[start:stop:step], self.y[start:stop:step]

    def finalize(self):
        """Finalize all local-fit models."""
        for f in self.fits:
            f.model.finalize()

    def update(self) -> None:
        """Update all local-fit models."""
        super().update()
        for f in self.fits:
            f.model.update()

    def update_model(self, **kwargs) -> None:
        """Recompute all local-fit models, optionally in parallel threads.

        Parameters
        ----------
        **kwargs
            Forwarded to each local model's ``update_model``.
        """
        if chisurf.settings.cs_settings['optimization']['global_threaded_model_update']:
            threads = [threading.Thread(target=f.model.update_model) for f in self.fits]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for f in self.fits:
                f.model.update_model()
