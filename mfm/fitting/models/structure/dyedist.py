from ..globalfit import GlobalFitModel
from mfm.parameter import AggregatedParameters, Parameter
import numpy as np


class DyePosition(AggregatedParameters):

    @property
    def coord(self):
        return np.array([self.x, self.y, self.z])

    @property
    def x(self):
        return self._x.value

    @x.setter
    def x(self, v):
        self._x.value = v

    @property
    def y(self):
        return self._x.value

    @y.setter
    def y(self, v):
        self._y.value = v

    @property
    def z(self):
        return self._z.value

    @z.setter
    def z(self, v):
        self._z.value = v

    def __init__(self, x, y, z, **kwargs):
        AggregatedParameters.__init__(self, **kwargs)
        self._x = Parameter(x)
        self._y = Parameter(y)
        self._z = Parameter(z)

        sigma = kwargs.get('sigma', 6.0)
        self._sigma = Parameter(sigma)


class DyeNetwork(GlobalFitModel):

    def __init__(self, fit):
        GlobalFitModel.__init__(self, fit)