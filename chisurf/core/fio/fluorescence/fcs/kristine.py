import os

import numpy as np

import chisurf as cs
import chisurf.core.fluorescence

from chisurf import typing
from chisurf.core.fio.fluorescence.fcs.definitions import FCSDataset


def write_kristine(
        filename: str,
        correlation_amplitude: np.ndarray,
        correlation_time: np.ndarray,
        mean_countrate: float,
        acquisition_time: float,
        correlation_amplitude_uncertainty: np.ndarray = None,
        mask: np.ndarray = None,
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
    col_3[0] = acquisition_time
    col_3[1] = mean_countrate
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
        )
    if isinstance(mask, np.ndarray):
        data = np.vstack([data, mask])
    data = data.T
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

    data = np.loadtxt(filename, encoding='utf-8')

    # In kristine file-type
    # data is (n_points, n_columns), so we take all rows for each column
    x, y = data[:, 0], data[:, 1]
    i = np.where(x > 0.0)
    x = x[i]
    y = y[i]

    # Metadata (duration, count rate) is stored in the 3rd column (index 2)
    try:
        dur, cr = data[0, 2], data[1, 2]
    except IndexError:
        dur, cr = 1.0, 1.0

    # First try to use experimental errors in the 4th column (index 3)
    try:
        w = 1. / data[:, 3][i]
    except (IndexError, ValueError):
        # In case everything fails
        # Use no errors at all but uniform weighting
        w = 1. / cs.core.fluorescence.fcs.noise(x, y, dur, cr, weight_type='suren')

    # Try to load mask from the 5th column (index 4)
    try:
        mask = data[:, 4][i]
    except (IndexError, ValueError):
        mask = np.ones_like(x)

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
            'mask': mask.tolist(),
            'intensity_trace': None
        }
    ]


def write_dict_to_kristine(
        filename: str,
        ds: typing.List[FCSDataset],
        verbose: bool = True
) -> None:
    """Write multiple FCS datasets to individual Kristine .cor files.

    Parameters
    ----------
    filename : str
        Base filename; individual curves are enumerated.
    ds : list of FCSDataset
        FCS datasets to write.
    verbose : bool
        If True, print progress.
    """
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
