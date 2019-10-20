"""

"""
from __future__ import annotations

import numpy as np
from qtpy import QtWidgets

import chisurf.settings as mfm
from mfm import plots
from chisurf.models.tcspc.lifetime import _membrane, Lifetime
from chisurf.models.tcspc.fret import FRETModel
from chisurf.models.tcspc.widgets import GenericWidget, CorrectionsWidget, ConvolveWidget, LifetimeWidget
from fitting.widgets import FittingControllerWidget, FittingParameterWidget
from fitting.parameter import FittingParameter


class GridModel(FRETModel):

    @property
    def ncols(self):
        return int(self._ncols.value)

    @ncols.setter
    def ncols(self, v):
        self._ncols.value = int(v)

    @property
    def nrows(self):
        return int(self._nrows.value)

    @nrows.setter
    def nrows(self, v):
        self._nrows.value = int(v)

    @property
    def r1(self):
        return self._r1.value

    @r1.setter
    def r1(self, v):
        self._r1.value = float(v)

    @property
    def r2(self):
        return self._r2.value

    @r2.setter
    def r2(self, v):
        self._r2.value = float(v)

    @property
    def d(self):
        return self._d.value

    @d.setter
    def d(self, v):
        self._d.value = float(v)

    @property
    def alpha(self):
        return self._alpha.value

    @alpha.setter
    def alpha(self, v):
        self._alpha.value = float(v)

    @property
    def vec1(self):
        return np.array([1., 0, 0], dtype=np.float64) * self.r1

    @property
    def vec2(self):
        a = self.alpha / 360. * np.pi * 2.0
        return np.array([np.cos(a), np.sin(a), 0], dtype=np.float64) * self.r2

    @property
    def pa(self):
        return self._pa.value / (self._pu.value + self._pd.value + self._pa.value)

    @pa.setter
    def pa(self, v):
        self._pa = float(v)

    @property
    def pd(self):
        return self._pd.value / (self._pa.value + self._pu.value + self._pd.value)

    @pd.setter
    def pd(self, v):
        self._pd.value = float(v)

    @property
    def pu(self):
        return self._pu.value / (self._pa.value + self._pd.value + self._pu.value)

    @pu.setter
    def pu(self, v):
        self._pu.value = float(v)

    @property
    def fret_rate_spectrum(self):
        self.calculate_rates()
        rates = self.calculate_rates()
        amplitudes = np.zeros_like(rates)
        return np.hstack([amplitudes, rates]).ravel(-1)

    def __init__(self, fit, **kwargs):
        """
        :param ncols: number of columns
        :param nrows: number of rows
        :param r1: distance parameter
        :param r2: distance parameter
        :param d: distance between the av-points on each grid-point
        :param alpha: the angle of the unit-cell in degrees
        :param pa: probability of acceptor
        :param pd: probability of donor
        :param pu: probability of unlabeled
        :return:

        >>> from chisurf.models.tcspc.membrane import GridModel
        >>> forster_radius = 50
        >>> tau0 = 4.0
        >>> n_cols, n_rows = 100, 100
        >>> r1, r2 = 50, 70
        >>> d = 10
        >>> alpha = 80
        >>> pa, pd, pu = 0.2, 0.1, 0.7
        >>> g = GridModel(n_cols, n_rows, r1, r2, d, alpha, pa, pd, pu, forster_radius, tau0)
        >>> g.update_grid()
        >>> rates = g.calculate_rates()

        Look at dependence of angle

        """
        FRETModel.__init__(self, fit, **kwargs)
        self._av_primitive = np.array(
            [
                [0.,  0., 0.],
                [1.,  0., 0.],
                [-1.,  0., 0.],
                [0., 1., 0.],
                [0., -1., 0.],
                [0., 0., 1.0]
            ],
            dtype=np.float64
        )

        # Model parameter
        self._ncols = FittingParameter(kwargs.get('ncols', 30), name='ncols', fixed=True)
        self._nrows = FittingParameter(kwargs.get('nrows', 30), name='nrows', fixed=True)

        # Shape paramter
        self._r1 = FittingParameter(kwargs.get('r1', 40), name='r1')
        self._r2 = FittingParameter(kwargs.get('r1', 60), name='r2')
        self._alpha = FittingParameter(kwargs.get('alpha', 60), name='alpha')
        self._d = FittingParameter(kwargs.get('d', 60), name='d', fixed=True)

        # Concentration parameter
        self._pa = FittingParameter(kwargs.get('pa', 0.8), name='pa', fixed=True)
        self._pd = FittingParameter(kwargs.get('pd', 0.1), name='pd', fixed=True)
        self._pu = FittingParameter(kwargs.get('pu', 0.0), name='pu', fixed=True)

        # FRET-rate parameter
        self._forster_radius = FittingParameter(kwargs.get('forster_radius',
                                                           chisurf.settings.cs_settings['fret']['forster_radius']),
                                                name='R0', fixed=True)
        self._tau0 = FittingParameter(kwargs.get('tau0',
                                                 chisurf.settings.cs_settings['fret']['tau0']),
                                      name='tau0', fixed=True)
        self._donors = kwargs.get('donors', Lifetime('D'))

        # Hidden internal parameters
        self._grid = None
        self._donor_list = list()
        self._acceptor_list = list()

        # FRETModel initialization
        FRETModel.__init__(self, fit, **kwargs)

    def update_grid(self):
        n_cols = self.ncols
        n_rows = self.nrows
        n_av = self._av_primitive.shape[0]
        av = self._av_primitive * self.d
        vec_1 = self.vec1
        vec_2 = self.vec2

        grid = np.zeros((n_cols, n_rows, n_av, 3), dtype=np.float64)
        random_av_points = np.random.randint(0, n_av, (n_cols, n_rows))

        donor_list = list()
        acceptor_list = list()
        for i in range(self.nrows):
            ri = vec_1 * i
            for j in range(self.ncols):
                rj = vec_2 * j

                rij = ri + rj
                grid[i, j] = av + rij
                # assign dye types to grid-points
                if np.random.ranf() > self.pu:
                    if np.random.ranf() < self.pd / (self.pd + self.pa):
                        donor_list.append([i, j, random_av_points[i, j]])
                    else:
                        acceptor_list.append([i, j, random_av_points[i, j]])

        self._donor_list = donor_list
        self._acceptor_list = acceptor_list
        self._grid = grid

    def calculate_rates(self):
        self.update_grid()
        r0 = self.forster_radius
        tau0 = self.tauD0
        grid = self._grid
        donors = np.array(self._donor_list, dtype=np.uint32)
        acceptors = np.array(self._acceptor_list, dtype=np.uint32)
        try:
            rates = _membrane.calculate_rates(r0, tau0, donors, acceptors, grid)
            rates = rates[np.isfinite(rates)]
        except:
            return np.array([0.0], dtype=np.float64)
        return rates


class GridModelWidget(GridModel, QtWidgets.QWidget):

    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                      'd_scaley': 'log',
                                      'r_scalex': 'lin',
                                      'r_scaley': 'lin',
    }),
                    (plots.SurfacePlot, {})
    ]

    name = "FD(A): Grid-fit"

    def __init__(self, fit, **kwargs):
        QtWidgets.QWidget.__init__(self)

        ## TCSPC-specific
        convolve = ConvolveWidget(fit=fit, model=self, hide_curve_convolution=True, **kwargs)
        donors = LifetimeWidget(parent=self, model=self, title='Donor(0)', short='D')
        generic = GenericWidget(fit=fit, model=self, parent=self, **kwargs)
        fitting = FittingControllerWidget(fit=fit, **kwargs)
        corrections = CorrectionsWidget(fit, model=self, **kwargs)
        GridModel.__init__(self, fit=fit, donors=donors,
                      generic=generic, corrections=corrections, convolve=convolve)

        ### Model parameter
        self._ncols = FittingParameterWidget(value=kwargs.get('ncols', 30), name='ncols', fixed=True)
        self._nrows = FittingParameterWidget(value=kwargs.get('nrows', 30), name='nrows', fixed=True)
        # Shape paramter
        self._r1 = FittingParameterWidget(value=kwargs.get('r1', 40), name='r1')
        self._r2 = FittingParameterWidget(value=kwargs.get('r1', 60), name='r2')
        self._alpha = FittingParameterWidget(value=kwargs.get('alpha', 60), name='alpha')
        self._d = FittingParameterWidget(value=kwargs.get('d', 60), name='d', fixed=True)
        # Concentration parameter
        self._pa = FittingParameterWidget(value=kwargs.get('pa', 0.8), name='pa', fixed=True)
        self._pd = FittingParameterWidget(value=kwargs.get('pd', 0.1), name='pd', fixed=True)
        self._pu = FittingParameterWidget(value=kwargs.get('pu', 0.0), name='pu', fixed=True)
        # FRET-rate parameter
        self._forster_radius = FittingParameterWidget(value=kwargs.get('forster_radius',
                                                                       chisurf.settings.cs_settings['fret']['forster_radius']),
                                                      name='R0', fixed=True)
        self._tau0 = FittingParameterWidget(value=kwargs.get('tau0',
                                                             chisurf.settings.cs_settings['fret']['tau0']),
                                            name='tau0', fixed=True)

        # Setup the layout
        self.layout = QtWidgets.QGridLayout(self)

        v = QtWidgets.QVBoxLayout()
        self.layout.addLayout(v, 0, 0, 1, 2)
        v.addWidget(fitting)
        v.addWidget(convolve)
        v.addWidget(generic)
        v.addWidget(donors)

        self.layout.addWidget(self._ncols, 1, 0)
        self.layout.addWidget(self._nrows, 1, 1)

        self.layout.addWidget(self._r1, 2, 0)
        self.layout.addWidget(self._r2, 2, 1)

        self.layout.addWidget(self._alpha, 3, 0)
        self.layout.addWidget(self._d, 3, 1)

        self.layout.addWidget(self._pa, 4, 0)
        self.layout.addWidget(self._pd, 4, 1)
        self.layout.addWidget(self._pu, 5, 0)

        self.layout.addWidget(self._forster_radius, 6, 0)
        self.layout.addWidget(self._tau0, 6, 1)

        self.layout.addWidget(corrections, 7, 0, 1, 2)
