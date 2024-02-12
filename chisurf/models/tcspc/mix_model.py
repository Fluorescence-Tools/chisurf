"""

"""
from __future__ import annotations

import numpy as np

from chisurf.fluorescence.general import stack_lifetime_spectra
from chisurf.models.tcspc.lifetime import LifetimeModel


class LifetimeMixModel(LifetimeModel):

    name = "Lifetime mix"

    @property
    def current_model_idx(self) -> int:
        return self._current_model_idx

    @current_model_idx.setter
    def current_model_idx(
            self,
            v: int
    ):
        self._current_model_idx = v

    @property
    def fractions(self) -> np.array:
        x = np.abs([f.value for f in self._fractions])
        x /= sum(x)
        return x

    @fractions.setter
    def fractions(
            self,
            x: np.array
    ):
        x = np.abs(x)
        x /= sum(x)
        for i, va in enumerate(x):
            self._fractions[i].value = va

    @property
    def lifetime_spectrum(self) -> np.array:
        if len(self) > 0:
            fractions = self.fractions
            lifetime_spectra = [m.lifetime_spectrum for m in self.models]
            return stack_lifetime_spectra(lifetime_spectra, fractions)
        else:
            return np.array([1.0, 1.0], dtype=np.float64)

    def clear_models(self):
        self._fractions = list()
        self.models = list()

    def append(self, model, fraction):
        self._fractions.append(fraction)
        model.find_parameters()
        self.models.append(model)

    def pop(self):
        return self.models.pop()

    @property
    def current_model(self):
        if len(self.models) > 0:
            return self.models[self.current_model_idx]
        else:
            return None

    @property
    def parameters_all(self):
        if self.current_model is not None:
            return self._parameters + self.current_model.parameters_all
        else:
            return self._parameters

    def update(self):
        self.find_parameters()
        for m in self.models:
            m.update()
        self.update_model()

    def finalize(self):
        LifetimeModel.finalize(self)
        for m in self.models:
            m.finalize()

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError as initial:
            try:
                return object.__getattribute__(self.current_model, attr)
            except AttributeError:
                raise initial

    def __len__(self):
        return len(self.models)

    def __init__(self, fit, **kwargs):
        LifetimeModel.__init__(self, fit, **kwargs)
        self.__dict__.pop('lifetimes')
        self._current_model_idx = 0
        self.models = list()
        self._fractions = list()

    def __str__(self):
        s = "Mix-models\n"
        s += "========\n\n"
        for m, x in zip(self.models, self.fractions):
            s += "\nFraction: %.3f\n\n" % x
            s += str(m)
        return s


