from __future__ import annotations

import yaml

import chisurf.base
import scikit_fluorescence.io.zipped

from chisurf import typing
from chisurf.fio.fluorescence.fcs.definitions import FCSDataset


def write_yaml(
        filename: str,
        data: typing.List[FCSDataset],
        verbose: bool = False
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
        print("Writing yaml .yaml to file: ", filename)
    txt = yaml.dump(
        data=chisurf.base.to_elementary(
            obj=data
        )
    )
    with scikit_fluorescence.io.zipped.open_maybe_zipped(
            filename=filename,
            mode='w'
    ) as fp:
        fp.write(txt)


def read_yaml(
        filename: str,
        verbose: bool = False
) -> typing.List[FCSDataset]:
    pass
