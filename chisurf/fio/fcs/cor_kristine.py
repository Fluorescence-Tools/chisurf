from __future__ import annotations
from typing import Dict, List
import numpy as np
import os

from . import weights


def fcs_write_kristine(
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


def fcs_write_dict_to_kristine(
        filename: str,
        ds: List[Dict],
        verbose: bool = True
) -> None:
    for i, d in enumerate(ds):
        root, ext = os.path.splitext(
            filename
        )
        fn = root + ("_%02d_" % i) + ext
        fcs_write_kristine(
            filename=fn,
            verbose=verbose,
            correlation_time=d['correlation_time'],
            correlation_amplitude=d['correlation_amplitude'],
            correlation_amplitude_uncertainty=1. / np.array(d['weights']),
            acquisition_time=d['acquisition_time'],
            mean_countrate=d['mean_count_rate']
        )


def fcs_read_kristine(
        filename: str,
        verbose: bool = False
) -> List[Dict]:
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
        w = weights.weights(x, y, dur, cr)
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
