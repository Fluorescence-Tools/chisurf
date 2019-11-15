from __future__ import annotations
import typing

import pathlib
import numpy as np
import scipy.io
import os

from . read_CSV_PyCorrFit import openCSV
from . asc_alv import openASC
from . fcs_confocor3 import openFCS
from .weights import weights
import chisurf


def write_kristine(
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
        print("Writing Kristine .cor to file: ", filename)
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


def write_dict_to_kristine(
        filename: str,
        ds: typing.List[typing.Dict],
        verbose: bool = True
) -> None:
    for i, d in enumerate(ds):
        root, ext = os.path.splitext(
            filename
        )
        fn = root + ("_%02d_" % i) + ext
        write_kristine(
            filename=fn,
            verbose=verbose,
            correlation_time=d['correlation_time'],
            correlation_amplitude=d['correlation_amplitude'],
            correlation_amplitude_uncertainty=1. / np.array(d['weights']),
            acquisition_time=d['acquisition_time'],
            mean_countrate=d['mean_count_rate']
        )


def read_kristine(
        filename: str,
        verbose: bool = False
) -> typing.List[typing.Dict]:
    """

    :param filename:
    :param verbose:
    :return:
    """
    if verbose:
        print("Reading Kristine .cor from file: ", filename)

    data = np.loadtxt(
        filename
    ).T

    # In Kristine file-type
    x, y = data[0], data[1]
    i = np.where(x > 0.0)
    x = x[i]
    y = y[i]
    dur, cr = data[2, 0], data[2, 1]

    # First try to use experimental errors
    try:
        w = 1. / data[3][i]
    except IndexError:
        # In case everything fails
        # Use no errors at all but uniform weighting
        w = weights(x, y, dur, cr)
    return [
        {
            'filename': filename,
            'correlation_time': x.tolist(),
            'correlation_amplitude': y.tolist(),
            'weights': w.tolist(),
            'acquisition_time': float(dur),
            'mean_count_rate': float(cr),
            'intensity_trace': None
        }
    ]


def write_china_mat(
        filename: str,
        d: typing.List[typing.Dict],
        verbose: bool = False
) -> None:
    if verbose:
        print("Writing to file: %s" % filename)
    mdict = dict()

    scipy.io.savemat(
        file_name=filename,
        mdict=mdict
    )


def read_china_mat(
        filename: str,
        skip_points: int = 4,
        verbose: bool = False
) -> typing.List[typing.Dict]:
    m = scipy.io.loadmat(filename)
    n_measurements = m['AA'].shape[1]
    # save intensity traces
    if verbose:
        print("Reading from file: %s" % filename)

    correlation_keys = {
        'Auto_AA': {
            'Intensity': 'IntA',
            'Correlation': 'AA'
        },
        'Auto_BB': {
            'Intensity': 'IntB',
            'Correlation': 'BB'
        },
        'Cross_Axb': {
            'Intensity': ['IntA', 'IntB'],
            'Correlation': 'AAxB'
        }
    }
    correlations = list()

    for correlation_key in correlation_keys:
        intensity_key = correlation_keys[correlation_key]['Intensity']
        correlation_key = correlation_keys[correlation_key]['Correlation']
        correlation_time = m[correlation_key][:, 0]

        for measurement_number in range(
                1, n_measurements
        ):
            correlation_amplitude = m[correlation_key][:, measurement_number] + 1.0
            if isinstance(intensity_key, list):
                k = intensity_key[0]
                intensity = np.zeros_like(m[k][:, measurement_number])
                intensity_time = m[k][:, 0]
                for k in intensity_key:
                    intensity += m[k][:, measurement_number]
            else:
                intensity_time = m[intensity_key][:, 0]
                intensity = m[intensity_key][:, measurement_number]
            aquisition_time = intensity_time[-1]
            mean_count_rate = intensity.sum() / (aquisition_time * 1000.0)
            w = weights(
                correlation_time,
                correlation_amplitude,
                aquisition_time,
                mean_count_rate=mean_count_rate,
                skip_points=skip_points
            )
            correlations.append(
                {
                    'filename': filename,
                    'measurement_id': "%s_%s" % (correlation_key, measurement_number),
                    'correlation_time': correlation_time.tolist(),
                    'correlation_amplitude': correlation_amplitude.tolist(),
                    'weights': w.tolist(),
                    'acquisition_time': float(aquisition_time),
                    'mean_count_rate': float(mean_count_rate),
                    'intensity_trace_time_ch1': intensity_time.tolist(),
                    'intensity_trace_ch1': intensity.tolist(),
                }
            )
    return correlations


def read_zeiss_fcs(
        filename: str,
        verbose: bool = False
) -> typing.List[typing.Dict]:
    if verbose:
        print("Reading ALV .asc from file: ", filename)
    d = openFCS(filename)

    correlations = list()
    for i, correlation in enumerate(d['Correlation']):
        correlation_time = correlation[:, 0]
        correlation_amplitude = correlation[:, 1] + 1.0

        trace = d['Trace'][i]
        if len(trace) == 2:
            # Cross correlation and two channels
            # Intensity in channel 1
            intensity_time_ch1 = trace[0][:, 0]
            intensity_ch1 = trace[0][:, 1]
            aquisition_time_ch1 = intensity_time_ch1[-1]
            mean_count_rate_ch1 = np.sum(intensity_ch1) / aquisition_time_ch1

            # Intensity in channel 2
            intensity_time_ch2 = trace[1][:, 0]
            intensity_ch2 = trace[1][:, 1]
            aquisition_time_ch2 = intensity_time_ch2[-1]
            mean_count_rate_ch2 = np.sum(intensity_ch2) / aquisition_time_ch2

            # Mean intensity
            mean_count_rate = 0.5 * (mean_count_rate_ch1 + mean_count_rate_ch2)

            # Mean aquisition time
            aquisition_time = 0.5 * (aquisition_time_ch1 + aquisition_time_ch2) / 1000.0

            w = weights(
                correlation_time,
                correlation_amplitude,
                aquisition_time,
                mean_count_rate=mean_count_rate
            )

            correlations.append(
                {
                    'filename': filename,
                    'measurement_id': "%s_%s" % (d['Filename'][i], i),
                    'correlation_time': correlation_time.tolist(),
                    'correlation_amplitude': correlation_amplitude.tolist(),
                    'weights': w.tolist(),
                    'acquisition_time': aquisition_time,
                    'mean_count_rate': mean_count_rate,
                    'intensity_trace_time_ch1': intensity_time_ch1.tolist(),
                    'intensity_trace_ch1': intensity_ch1.tolist(),
                    'intensity_trace_time_ch2': intensity_time_ch2.tolist(),
                    'intensity_trace_ch2': intensity_ch2.tolist(),
                }
            )
        else:
            # Auto correlation only one channel
            # Intensity in channel 1
            intensity_time = trace[:, 0]
            intensity = trace[:, 1]
            aquisition_time = intensity_time[-1] / 1000.0
            mean_count_rate = float(np.mean(intensity))

            w = weights(
                correlation_time,
                correlation_amplitude,
                aquisition_time,
                mean_count_rate=mean_count_rate
            )

            correlations.append(
                {
                    'filename': filename,
                    'measurement_id': "%s_%s" % (d['Filename'][i], i),
                    'correlation_time': correlation_time.tolist(),
                    'correlation_amplitude': correlation_amplitude.tolist(),
                    'weights': w.tolist(),
                    'acquisition_time': aquisition_time,
                    'mean_count_rate': mean_count_rate,
                    'intensity_trace_time_ch1': intensity_time.tolist(),
                    'intensity_trace_ch1': intensity.tolist(),
                }
            )

    return correlations


avl_to_yaml = {
    'Temperature [K] :': {
        'name': 'temperature',
        'type': float
    },
    'Viscosity [cp]  :': {
        'name': 'viscosity',
        'type': float
    },
    'Duration [s]    :': {
        'name': 'acquisition time',
        'type': float
    },
    'MeanCR0 [kHz]   :': {
        'name': 'mean count rate',
        'type': float
    },
    'MeanCR1 [kHz]   :': {
        'name': 'mean count rate',
        'type': float
    },
    'MeanCR2 [kHz]   :': {
        'name': 'mean count rate',
        'type': float
    },
    'MeanCR3 [kHz]   :': {
        'name': 'mean count rate',
        'type': float
    }
}


def read_asc_header(
    filename: str
):
    path = pathlib.Path(filename)
    d = dict()
    with path.open(
        mode='r',
        encoding="iso8859_1"
    ) as fp:
        for line in fp.readlines():
            lv = line.split("\t")
            try:
                print(lv[0])
                d[
                    avl_to_yaml[lv[0]]['name']
                ] = avl_to_yaml[lv[0]]['type'].__call__(lv[1])
            except:
                pass
    return d


def read_asc(
        filename: str,
        verbose: bool = False
) -> typing.List[typing.Dict]:
    if verbose:
        print("Reading ALV .asc from file: ", filename)
    d = openASC(filename)
    correlations = list()

    for i, correlation in enumerate(d['Correlation']):
        correlation_time = correlation[:, 0]
        correlation_amplitude = correlation[:, 1] + 1.0
        intensity_time = d['Trace'][i][:, 0]
        intensity = d['Trace'][i][:, 1]
        aquisition_time = intensity_time[-1]
        mean_count_rate = np.sum(intensity) / aquisition_time

        w = weights(
            correlation_time,
            correlation_amplitude,
            aquisition_time,
            mean_count_rate=mean_count_rate
        )

        correlations.append(
            {
                'filename': filename,
                'measurement_id': "%s_%s" % (d['Filename'], i),
                'correlation_time': correlation_time.tolist(),
                'correlation_amplitude': correlation_amplitude.tolist(),
                'weights': w.tolist(),
                'acquisition_time': aquisition_time,
                'mean_count_rate': mean_count_rate,
                'intensity_trace_time': intensity_time.tolist(),
                'intensity_trace': intensity.tolist(),
            }
        )

    return correlations


def read_pycorrfit(
        filename: str,
        verbose: bool = False
) -> typing.List[typing.Dict]:
    if verbose:
        print("Reading PyCorrFit from file: ", filename)
    d = openCSV(filename)
    correlations = list()

    for i, correlation in enumerate(d['Correlation']):
        r = dict()
        correlation_time = correlation[:, 0]
        correlation_amplitude = correlation[:, 1]
        intenstiy_trace_time = (d['Trace'][i][0][:, 0] / 1000.0).tolist()
        intensity_trace_ch1 = (d['Trace'][i][0][:, 1]).tolist()
        aquisition_time = float(intenstiy_trace_time[-1])
        mean_count_rate = float(d['Trace'][i][0][:, 0].sum()) / (aquisition_time * 1000.0)
        r.update(
            {
                'filename': filename,
                'measurement_id': "%s_%s" % (d['Filename'], i),
                'correlation_time': correlation_time,
                'correlation_amplitude': correlation_amplitude,
                'intensity_trace_time_ch1': intenstiy_trace_time,
                'intensity_trace_ch1': intensity_trace_ch1,
                'acquisition_time': aquisition_time,
                'mean_count_rate': mean_count_rate,
            }
        )
        r.update(
            {
                'weights': weights(
                    times=correlation_time,
                    correlation=correlation_amplitude,
                    measurement_duration=r['acquisition_time'],
                    mean_count_rate=r['mean_count_rate'],
                ).tolist()
            }
        )
        try:
            intenstiy_trace_time = (d['Trace'][i][1][:, 0]).tolist()
            intensity_trace_ch2 = (d['Trace'][i][1][:, 1]).tolist(),
            r.update(
                {
                    'intensity_trace_time_ch2': intenstiy_trace_time,
                    'intensity_trace_ch2': intensity_trace_ch2,
                }
            )
        except KeyError:
            chisurf.logging.warning("PyCorrFit loading no second intensity trace")

        correlations.append(r)

    return correlations
