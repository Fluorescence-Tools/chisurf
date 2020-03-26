import numpy as np
import scipy.io

import chisurf
import chisurf.fluorescence.fcs

from chisurf import typing
from chisurf.fio.fluorescence.fcs.definitions import FCSDataset


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
) -> typing.List[FCSDataset]:
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
            w = 1. / chisurf.fluorescence.fcs.noise(
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
                    'correlation_amplitude_weights': w.tolist(),
                    'acquisition_time': float(aquisition_time),
                    'mean_count_rate': float(mean_count_rate),
                    'intensity_trace_time_ch1': intensity_time.tolist(),
                    'intensity_trace_ch1': intensity.tolist(),
                }
            )
    return correlations