from __future__ import annotations
from typing import Tuple

import numpy as np

import mfm
import mfm.experiments
import mfm.base
import mfm.fluorescence
import mfm.experiments.data
import mfm.widgets
from mfm.io.ascii import Csv

from .. import reader


class TCSPCReader(
    reader.ExperimentReader
):

    def __init__(
            self,
            dt: float = None,
            rep_rate: float = None,
            is_jordi: bool = False,
            mode: str = 'vm',
            g_factor: float = None,
            rebin: Tuple[int, int] = (1, 1),
            matrix_columns: Tuple[int, int] = (0, 1),
            skiprows: int = 7,
            polarization: str = 'vm',
            use_header: bool = True,
            fit_area: float = None,
            fit_count_threshold: float = None,
            *args,
            **kwargs
    ):
        """

        :param dt:
        :param rep_rate:
        :param is_jordi:
        :param mode:
        :param g_factor:
        :param rebin:
        :param matrix_columns:
        :param skiprows:
        :param polarization:
        :param use_header:
        :param fit_area:
        :param fit_count_threshold:
        :param args:
        :param kwargs:
        """
        super().__init__(
            *args,
            **kwargs
        )
        if dt is None:
            dt = mfm.settings.tcspc['dt']
        if g_factor is None:
            g_factor = mfm.settings.tcspc['g_factor']
        if rep_rate is None:
            rep_rate = mfm.settings.tcspc['rep_rate']
        if fit_area is None:
            fit_area = mfm.settings.tcspc['fit_area']
        if fit_count_threshold is None:
            fit_count_threshold = mfm.settings.tcspc['fit_count_threshold']

        self.dt = dt
        self.excitation_repetition_rate = rep_rate
        self.is_jordi = is_jordi
        self.polarization = mode
        self.g_factor = g_factor
        self.rep_rate = rep_rate
        self.rebin = rebin
        self.matrix_columns = matrix_columns
        self.skiprows = skiprows
        self.polarization = polarization
        self.use_header = use_header
        self.matrix_columns = matrix_columns
        self.fit_area = fit_area
        self.fit_count_threshold = fit_count_threshold

    def autofitrange(
            self,
            data,
            **kwargs
    ) -> Tuple[int, int]:
        return mfm.fluorescence.tcspc.fitrange(
            data.y,
            self.fit_count_threshold,
            self.fit_area
        )

    def read(
            self,
            filename: str = None,
            *args,
            **kwargs
    ) -> mfm.experiments.data.DataCurveGroup:
        return read_tcspc_csv(
            filename=filename,
            skiprows=self.skiprows,
            rebin=self.rebin,
            dt=self.dt,
            matrix_columns=self.matrix_columns,
            use_header=self.use_header,
            is_jordi=self.is_jordi,
            polarization=self.polarization,
            g_factor=self.g_factor,
            setup=self,
        )


def read_tcspc_csv(
        filename: str = None,
        skiprows: int = None,
        rebin: Tuple[int, int] = (1, 1),
        dt: float = 1.0,
        matrix_columns: Tuple[int, int] = (0, 1),
        use_header: bool = False,
        is_jordi: bool = False,
        polarization: str = "vm",
        g_factor: float = 1.0,
        setup: reader.ExperimentReader = None,
        *args,
        **kwargs
) -> mfm.experiments.data.DataCurveGroup:
    """

    :param filename:
    :param skiprows:
    :param rebin:
    :param dt:
    :param matrix_columns:
    :param use_header:
    :param is_jordi:
    :param polarization:
    :param g_factor:
    :param setup:
    :param args:
    :param kwargs:
    :return:
    """

    # Load data
    rebin_x, rebin_y = rebin
    mc = matrix_columns

    csvSetup = Csv(
        *args,
        **kwargs
    )
    csvSetup.load(
        filename,
        skiprows=skiprows,
        use_header=use_header,
        usecols=mc
    )
    data = csvSetup.data

    if is_jordi:
        if data.ndim == 1:
            data = data.reshape(1, len(data))
        n_data_sets, n_vv_vh = data.shape
        n_data_points = n_vv_vh / 2
        c1, c2 = data[:, :n_data_points], data[:, n_data_points:]

        new_channels = int(n_data_points / rebin_y)
        c1 = c1.reshape([n_data_sets, new_channels, rebin_y]).sum(axis=2)
        c2 = c2.reshape([n_data_sets, new_channels, rebin_y]).sum(axis=2)
        n_data_points = c1.shape[1]

        if polarization == 'vv':
            y = c1
            ey = mfm.fluorescence.tcspc.weights(c1)
        elif polarization == 'vh':
            y = c2
            ey = mfm.fluorescence.tcspc.weights(c2)
        elif polarization == 'vv/vh':
            e1 = mfm.fluorescence.tcspc.weights(c1)
            e2 = mfm.fluorescence.tcspc.weights(c2)
            y = np.vstack([c1, c2])
            ey = np.vstack([e1, e2])
        else:
            f2 = 2.0 * g_factor
            y = c1 + f2 * c2
            ey = mfm.fluorescence.tcspc.weights_ps(
                c1, c2, f2
            )
        x = np.arange(n_data_points, dtype=np.float64) * dt
    else:
        x = data[0] * dt
        y = data[1:]

        n_datasets, n_data_points = y.shape
        n_data_points = int(n_data_points / rebin_y)
        y = y.reshape([n_datasets, n_data_points, rebin_y]).sum(axis=2)
        ey = mfm.fluorescence.tcspc.weights(y)
        x = np.average(
            x.reshape([n_data_points, rebin_y]), axis=1
        ) / rebin_y

    # TODO: in future adaptive binning of time axis
    #from scipy.stats import binned_statistic
    #dt = xn[1]-xn[0]
    #xb = np.logspace(np.log10(dt), np.log10(np.max(xn)), 512)
    #tmp = binned_statistic(xn, yn, statistic='sum', bins=xb)
    #xn = xb[:-1]
    #print xn
    #yn = tmp[0]
    #print tmp[1].shape
    x = x[n_data_points % rebin_y:]
    y = y[n_data_points % rebin_y:]

    # rebin along x-axis
    y_rebin = np.zeros_like(y)
    ib = 0
    for ix in range(0, y.shape[0], rebin_x):
        y_rebin[ib] += y[ix:ix+rebin_x, :].sum(axis=0)
        ib += 1
    y_rebin = y_rebin[:ib, :]
    ex = np.zeros(x.shape)
    data_curves = list()
    n_data_sets = y_rebin.shape[0]
    fn = csvSetup.filename
    for i, yi in enumerate(y_rebin):
        eyi = ey[i]
        if n_data_sets > 1:
            name = '{} {:d}_{:d}'.format(fn, i, n_data_sets)
        else:
            name = filename
        data = mfm.experiments.data.DataCurve(
            x=x,
            y=yi,
            ex=ex,
            ey=1. / eyi,
            setup=setup,
            name=name,
            **kwargs
        )
        data.filename = filename
        data_curves.append(data)
    data_group = mfm.experiments.data.DataCurveGroup(
        data_curves,
        filename,
    )
    return data_group
