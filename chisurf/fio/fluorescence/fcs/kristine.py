import os

import numpy as np

import chisurf
import chisurf.fluorescence

from chisurf import typing
from chisurf.fio.fluorescence.fcs.definitions import FCSDataset


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
        print("Writing kristine .cor to file: ", filename)
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


def read_kristine(
        filename: str,
        verbose: bool = False
) -> typing.List[FCSDataset]:
    """

    :param filename:
    :param verbose:
    :return:
    """
    if verbose:
        print("Reading kristine .cor from file: ", filename)

    data = np.loadtxt(
        filename
    ).T

    # In kristine file-type
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
        w = 1. / chisurf.fluorescence.fcs.noise(x, y, dur, cr)
    measurement_id, _ = os.path.splitext(
        os.path.basename(
            filename
        )
    )
    return [
        {
            'filename': filename,
            'measurement_id': measurement_id,
            'acquisition_time': float(dur),
            'mean_count_rate': float(cr),
            'correlation_times': x.tolist(),
            'correlation_amplitudes': y.tolist(),
            'correlation_amplitude_weights': w.tolist(),
            'intensity_trace': None
        }
    ]


def write_dict_to_kristine(
        filename: str,
        ds: typing.List[FCSDataset],
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
            correlation_time=d['correlation_times'],
            correlation_amplitude=d['correlation_amplitudes'],
            correlation_amplitude_uncertainty=1. / np.array(d['correlation_amplitude_weights']),
            acquisition_time=d['acquisition_time'],
            mean_countrate=d['mean_count_rate']
        )
