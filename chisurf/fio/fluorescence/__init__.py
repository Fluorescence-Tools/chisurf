from __future__ import annotations

# the following imports are only for the type annotations
# and currently (20.03.13) cause conflicts
# chisurf.experiments.Experiment
# import chisurf.data

import os

import numpy as np

import chisurf.data
import chisurf.fio
import chisurf.fio.fluorescence.fcs
import chisurf.fio.fluorescence.photons
import chisurf.fio.fluorescence.sdtfile

from chisurf import typing


def read_fcs(
        filename: str,
        data_reader: chisurf.experiments.reader.ExperimentReader = None,
        reader_name: str = 'csv',
        verbose: bool = False,
        experiment: chisurf.experiments.Experiment = None,
        **kwargs
) -> chisurf.data.ExperimentDataCurveGroup:
    """

    Option Kristine:

    Uses either the error provided by the correlator (4. column)
    or calculates the error based on the correlation curve,
    the aquisition time and the count-rate.

    :param filename:
    :param data_reader:
    :param data_reader:
    :param verbose:
    :param args:
    :param kwargs:
    :return:
    """
    data_sets = list()
    root, ext = os.path.splitext(
        os.path.basename(filename)
    )

    # Files with single curve
    if reader_name in ['csv', 'kristine', 'pycorrfit']:
        if reader_name == 'csv':
            csv = chisurf.fio.ascii.Csv()
            csv.load(
                filename=filename,
                verbose=chisurf.verbose,
                **kwargs
            )
            x, y = csv.data[0], csv.data[1]
            ey = csv.data[2]
        elif reader_name == 'kristine':
            r = chisurf.fio.fluorescence.fcs.kristine.read_kristine(
                filename=filename,
                verbose=verbose
            )
            x = np.array(r[0]['correlation_time'])
            y = np.array(r[0]['correlation_amplitude'])
            ey = 1. / np.array(r[0]['weights'])
            ex = np.ones_like(x)
        elif reader_name == 'pycorrfit':
            r = chisurf.fio.fluorescence.fcs.pycorrfit.read_pycorrfit(
                filename=filename,
                verbose=verbose
            )
            x = np.array(r[0]['correlation_time'])
            y = np.array(r[0]['correlation_amplitude'])
            ey = 1. / np.array(r[0]['weights'])
            ex = np.ones_like(x)
        name = root
        data_sets.append(
            chisurf.data.DataCurve(
                data_reader=data_reader,
                name=name,
                x=x, y=y, ey=ey, ex=ex,
                experiment=experiment
            )
        )
    # Files with multiple curves per file
    elif reader_name in ['china-mat', 'confocor3', 'alv']:
        if reader_name == 'confocor3':
            ds = chisurf.fio.fcs.fcs_confocor3.read_zeiss_fcs(
                filename=filename,
                verbose=verbose
            )
        elif reader_name == 'china-mat':
            ds = chisurf.fio.fcs.fcs_china.read_china_mat(
                filename=filename,
                verbose=verbose
            )
        elif reader_name == 'alv':
            ds = chisurf.fio.fcs.fcs_asc_alv.read_asc(
                filename
            )
        for r in ds:
            name = root + "_" + r['measurement_id']
            x = np.array(r['correlation_time'])
            y = np.array(r['correlation_amplitude'])
            ey = 1. / np.array(r['weights'])
            ex = np.ones_like(x)
            data_sets.append(
                chisurf.data.DataCurve(
                    name=name,
                    data_reader=data_reader,
                    x=x, y=y, ey=ey, ex=ex
                )
            )
    return chisurf.data.ExperimentDataCurveGroup(data_sets)


def read_tcspc_csv(
        filename: str = None,
        skiprows: int = None,
        rebin: typing.Tuple[int, int] = (1, 1),
        dt: float = 1.0,
        matrix_columns: typing.Tuple[int, int] = (0, 1),
        use_header: bool = False,
        is_jordi: bool = False,
        polarization: str = "vm",
        g_factor: float = 1.0,
        experiment: chisurf.experiments.Experiment = None,
        *args,
        **kwargs
) -> chisurf.data.DataCurveGroup:
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

    csvSetup =  chisurf.fio.ascii.Csv(
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
            ey = chisurf.fluorescence.tcspc.counting_noise(
                decay=c1
            )
        elif polarization == 'vh':
            y = c2
            ey = chisurf.fluorescence.tcspc.counting_noise(
                decay=c2
            )
        elif polarization == 'vv/vh':
            e1 = chisurf.fluorescence.tcspc.counting_noise(
                decay=c1
            )
            e2 = chisurf.fluorescence.tcspc.counting_noise(
                decay=c2
            )
            y = np.vstack([c1, c2])
            ey = np.vstack([e1, e2])
        else:
            f2 = 2.0 * g_factor
            y = c1 + f2 * c2
            ey = chisurf.fluorescence.tcspc.counting_noise_combined_parallel_perpendicular(
                parallel=c1,
                perpendicular=c2,
                g_factor=g_factor
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
            ey = chisurf.fluorescence.tcspc.counting_noise(y)
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
        data = chisurf.data.DataCurve(
            x=x,
            y=yi,
            ex=ex,
            ey=eyi,
            experiment=experiment,
            name=name,
            **kwargs
        )
        data.filename = filename
        data_curves.append(data)
    data_group = chisurf.data.DataCurveGroup(
        data_curves,
        filename,
    )
    return data_group