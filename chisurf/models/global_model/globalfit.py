from __future__ import annotations
from chisurf import typing

import threading
import numpy as np

import chisurf.decorators
import chisurf.parameter
import chisurf.fitting.fit

from chisurf.curve import Curve
from chisurf.models import model
from chisurf.fitting.parameter import GlobalFittingParameter


class GlobalFitModel(model.Model, Curve):

    name = "Global fit"

    @property
    def weighted_residuals(self) -> np.ndarray:
        if len(self.fits) > 0:
            re = list()
            for f in self.fits:
                re.append(f.model.weighted_residuals.flatten())
            return np.concatenate(re)
        else:
            return np.array([], dtype=np.float64)

    @property
    def fit_names(self) -> typing.List[str]:
        return [f.name for f in self.fits]

    @property
    def links(self) -> typing.List[chisurf.fitting.parameter.FittingParameter]:
        return self._links

    @links.setter
    def links(self, v: typing.List[chisurf.fitting.parameter.FittingParameter]):
        self._links = v if isinstance(v, list) else list()

    @property
    def n_points(self) -> int:
        nbr_points = 0
        for f in self.fits:
            nbr_points += f.model.n_points
        return nbr_points

    @property
    def global_parameters_all(self) -> typing.List[chisurf.fitting.parameter.FittingParameter]:
        return list(self._global_parameters.values())

    @property
    def global_parameters_all_names(self) -> typing.List[str]:
        return [p.name for p in self.global_parameters_all]

    @property
    def global_parameters(self) -> typing.List[chisurf.fitting.parameter.FittingParameter]:
        return [p for p in self.global_parameters_all if not p.fixed]

    @property
    def global_parameters_names(self) -> typing.List[str]:
        return [p.name for p in self.global_parameters]

    @property
    def global_parameters_bound_all(self) -> typing.List[typing.Tuple[float, float]]:
        return [pi.bounds for pi in self.global_parameters_all]

    @property
    def global_parameter_linked_all(self) -> typing.List[bool]:
        return [p.is_linked for p in self.global_parameters_all]

    @property
    def parameters(self) -> typing.List[chisurf.fitting.parameter.FittingParameter]:
        p = list()
        for f in self.fits:
            p += f.model.parameters
        p += self.global_parameters
        return p

    @property
    def parameter_names(self) -> typing.List[str]:
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
        return [g.value for g in self.global_parameters_all]

    @property
    def global_parameters_fixed_all(self) -> typing.List[bool]:
        return [p.fixed for p in self.global_parameters_all]

    @property
    def parameter_names_all(self) -> typing.List[str]:
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
        re = dict()
        for i, f in enumerate(self.fits):
            d = f.model.parameter_dict
            k = [str(i+1)+":"+dk for dk in d.keys()]
            for j, di in enumerate(d.keys()):
                re[k[j]] = d[di]
        return re

    @property
    def data(self) -> typing.Tuple[np.array, np.array, np.array]:
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
            fit: chisurf.fitting.fit.Fit,
            fits: typing.List[chisurf.fitting.fit.Fit] = None,
            *args,
            **kwargs
    ):
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
            fit: chisurf.fitting.fit.Fit,
            xmin: int = None,
            xmax: int = None
    ) -> np.array:
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
        except:
            wr = np.array([1.0])
        return wr

    def append_fit(self, fit: chisurf.fitting.fit.Fit) -> None:
        if fit not in self.fits:
            self.fits.append(fit)

    def append_global_parameter(self, parameter: chisurf.parameter.Parameter) -> None:
        variable_name = parameter.name
        if variable_name not in list(self._global_parameters.keys()):
            self._global_parameters[parameter.name] = parameter

    def setLinks(self):
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
                target_parameter = GlobalFittingParameter(f, g, formula)

                origin_parameter.link = target_parameter
                print("f[%s][%s] linked to %s" % (origin_fit, origin_parameter.name, target_parameter.name))
            except IndexError:
                print("not enough fits index out of range")

    def autofitrange(self, fit: chisurf.fitting.fit.FitGroup):
        self.xmin, self.xmax = None, None
        return self.xmin, self.xmax

    def clear_local_fits(self) -> None:
        self.fits = list()

    def remove_local_fit(self, fit_index: int):
        del self.fits[fit_index]

    def clear_all_links(self) -> None:
        for fit in self.fits:
            for p in fit.model.parameters_all:
                p.link = None

    def clear_listed_links(self):
        self.links = list()

    def __str__(self):
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
        x = list()
        for f in self.fits:
            x.append(f.model.x)
        return np.array(x)

    @x.setter
    def x(self, v):
        pass

    @property
    def y(self) -> np.array:
        y = list()
        for f in self.fits:
            y.append(f.model.y)
        return np.array(y)

    @y.setter
    def y(self, v):
        pass

    def __getitem__(self, key):
        start = key.start
        stop = key.stop
        step = 1 if key.step is None else key.step
        return self.x[start:stop:step], self.y[start:stop:step]

    def finalize(self):
        for f in self.fits:
            f.model.finalize()

    def update(self) -> None:
        super().update()
        for f in self.fits:
            f.model.update()

    def update_model(self, **kwargs) -> None:
        if chisurf.settings.cs_settings['optimization']['global_threaded_model_update']:
            threads = [threading.Thread(target=f.model.update_model) for f in self.fits]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for f in self.fits:
                f.model.update_model()
