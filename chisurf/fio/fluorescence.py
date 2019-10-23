from __future__ import annotations

from typing import Tuple

import numpy as np

import chisurf.fluorescence
import chisurf.fluorescence.fcs
import chisurf.fluorescence.tcspc
import chisurf.experiments
from chisurf.experiments import reader
from chisurf.fio.ascii import Csv


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
) -> chisurf.experiments.data.DataCurveGroup:
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

    if is_jordi:
        infer_delimiter = False
        mc = None
    else:
        mc = matrix_columns
        infer_delimiter = True

    csvSetup = Csv(
        *args,
        **kwargs
    )
    csvSetup.load(
        filename,
        skiprows=skiprows,
        use_header=use_header,
        usecols=mc,
        infer_delimiter=infer_delimiter
    )
    data = csvSetup.data

    if is_jordi:
        if data.ndim == 1:
            data = data.reshape(1, len(data))
        n_data_sets, n_vv_vh = data.shape
        n_data_points = n_vv_vh // 2
        c1, c2 = data[:, :n_data_points], data[:, n_data_points:]

        new_channels = int(n_data_points / rebin_y)
        c1 = c1.reshape([n_data_sets, new_channels, rebin_y]).sum(axis=2)
        c2 = c2.reshape([n_data_sets, new_channels, rebin_y]).sum(axis=2)
        n_data_points = c1.shape[1]

        if polarization == 'vv':
            y = c1
            ey = chisurf.fluorescence.tcspc.weights(c1)
        elif polarization == 'vh':
            y = c2
            ey = chisurf.fluorescence.tcspc.weights(c2)
        elif polarization == 'vv/vh':
            e1 = chisurf.fluorescence.tcspc.weights(c1)
            e2 = chisurf.fluorescence.tcspc.weights(c2)
            y = np.vstack([c1, c2])
            ey = np.vstack([e1, e2])
        else:
            f2 = 2.0 * g_factor
            y = c1 + f2 * c2
            ey = chisurf.fluorescence.tcspc.weights_ps(
                c1, c2, f2
            )
        x = np.arange(
            n_data_points,
            dtype=np.float64
        ) * dt
    else:
        x = data[0] * dt
        y = data[1:]

        n_datasets, n_data_points = y.shape
        n_data_points = int(n_data_points / rebin_y)
        try:
            y = y.reshape(
                [n_datasets, n_data_points, rebin_y]
            ).sum(axis=2)
            ey = chisurf.fluorescence.tcspc.weights(y)
            x = np.average(
                x.reshape([n_data_points, rebin_y]), axis=1
            ) / rebin_y
        except ValueError:
            print("Cannot reshape array")

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
        data = chisurf.experiments.data.DataCurve(
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
    data_group = chisurf.experiments.data.DataCurveGroup(
        data_curves,
        filename,
    )
    return data_group


def read_fcs(
        filename: str,
        setup: reader.ExperimentReader = None,
        *args,
        **kwargs
) -> chisurf.experiments.data.DataCurve:
    csv = chisurf.fio.ascii.Csv()
    csv.load(
        filename=filename,
        verbose=chisurf.verbose,
        *args,
        **kwargs
    )
    x, y = csv.data[0], csv.data[1]
    w = csv.data[2]
    d = chisurf.experiments.data.DataCurve(
        setup=setup,
        x=x,
        y=y,
        ey=w,
        ex=np.ones_like(x)
    )
    return d


def save_fcs_kristine(
        filename: str,
        correlation_amplitude: np.ndarray,
        correlation_time: np.ndarray,
        mean_countrate: float,
        acquisition_time: float,
        correlation_amplitude_uncertainty: np.ndarray = None,
        verbose: bool = True
) -> None:
    """

    :param filename: the filename
    :param correlation_amplitude: an array containing the amplitude of the
    correlation function
    :param correlation_amplitude_uncertainty: an estimate for the
    uncertainty of the correlation amplitude
    :param correlation_time: an array containing the correlation times
    :param mean_countrate: the mean countrate of the experiment in kHz
    :param acquisition_time: the acquisition of the FCS experiment in
    seconds
    :return:
    """
    if verbose:
        print("Saving correlation: %s" % filename)
    col_1 = np.array(correlation_time)
    col_2 = np.array(correlation_amplitude)
    col_3 = np.zeros_like(correlation_amplitude)
    col_3[0] = mean_countrate
    col_3[1] = acquisition_time
    if isinstance(
            correlation_amplitude_uncertainty,
            np.ndarray
    ):
        data = np.vstack(
            [
                col_1,
                col_2,
                col_3,
                correlation_amplitude_uncertainty
            ]
        ).T
    else:
        data = np.vstack(
            [
                col_1,
                col_2,
                col_3
            ]
        ).T
    np.savetxt(
        filename,
        data,
    )


def read_fcs_kristine(
        filename: str,
        experiment_reader: chisurf.experiments.reader.ExperimentReader = None,
        verbose=chisurf.verbose
) -> chisurf.experiments.data.DataCurve:
    """Uses either the error provided by the correlator (4. column)
    or calculates the error based on the correlation curve,
    the aquisition time and the count-rate.

    :param experiment_reader:
    :param filename:
    :param verbose:

    :return:
    """
    csv = chisurf.fio.ascii.Csv(
        skiprows=0,
        use_header=False,
        filename=filename,
        verbose=verbose
    )
    data = csv.data
    d = chisurf.experiments.data.DataCurve(
        setup=experiment_reader
    )

    # In Kristine file-type
    # First try to use experimental errors
    x, y = data[0], data[1]
    i = np.where(x > 0.0)
    x = x[i]
    y = y[i]
    try:
        w = 1. / data[3][i]
    except IndexError:
        # In case this doesnt work
        # use calculated errors using count-rate and duration
        try:
            dur, cr = data[2, 0], data[2, 1]
            w = chisurf.fluorescence.fcs.weights(x, y, dur, cr)
        except IndexError:
            # In case everything fails
            # Use no errors at all but uniform weighting
            w = np.ones_like(y)
    d.set_data(
        x=x,
        y=y,
        ex=np.ones_like(x),
        ey=w
    )
    return d
