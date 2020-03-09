#!/usr/bin/env python
from __future__ import annotations

import os
import argparse
import numpy as np
import scipy.io


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


def save_intensity_trace(
        filename: str,
        time_axis: np.ndarray,
        intensity: np.ndarray,
        verbose: bool = True
):
    """

    :param filename: the output filename
    :param time_axis: the time axis
    :param intensity: the intensity corresponding to the time axis
    :return:
    """
    if verbose:
        print("Saving intensity trace: %s" % filename)
    data = np.vstack(
        [
            time_axis,
            intensity
        ]
    ).T
    np.savetxt(
        filename,
        data,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Convert Mat-FCS to Kristine files.'
    )

    parser.add_argument(
        'filename',
        metavar='file',
        type=str,
        help='The filename used to generate the histogram'
    )
    args = parser.parse_args()
    print("Convert a Mat FCS file to Kristine")
    print("==================================")
    print("\tFilename: %s" % args.filename)
    print("")

    file_prefix, _ = os.path.splitext(
        args.filename
    )

    m = scipy.io.loadmat(args.filename)

    n_measurements = m['AA'].shape[1]
    # save intensity traces

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
            'Intensity': 'MInt',
            'Correlation': 'AAxB'
        }
    }
    for correlation_key in correlation_keys:
        intensity_key = correlation_keys[correlation_key]['Intensity']
        correlation_key = correlation_keys[correlation_key]['Correlation']

        correlations = list()
        correlation_time = m[correlation_key][:, 0]
        correlations = list()
        for measurement_number in range(
                1, n_measurements
        ):
            correlation_amplitude = m[correlation_key][:, measurement_number]
            intensity_time = m[intensity_key][:, 0]
            intensity = m[intensity_key][:, measurement_number]
            save_fcs_kristine(
                filename=file_prefix + '_%s_%s' % (correlation_key, measurement_number) + '.cor',
                correlation_amplitude=correlation_amplitude,
                correlation_time=correlation_time,
                mean_countrate=intensity.mean() / 1000.0,
                acquisition_time=intensity_time[-1]
            )
            save_intensity_trace(
                filename=file_prefix + '_%s_%s' % (correlation_key, measurement_number) + '.int',
                intensity=intensity,
                time_axis=intensity_time
            )
            correlations.append(correlation_amplitude)
        correlations = np.array(correlations)
        save_fcs_kristine(
            filename=file_prefix + '_%s_mean' % (
                correlation_key
            ) + '.cor',
            correlation_amplitude=correlations.mean(axis=0),
            correlation_time=correlation_time,
            mean_countrate=intensity.mean() / 1000.0,
            acquisition_time=intensity_time[-1],
            correlation_amplitude_uncertainty=correlations.std(axis=0)
        )



