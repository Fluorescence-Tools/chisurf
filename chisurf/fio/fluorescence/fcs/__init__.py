from __future__ import annotations

import os

import numpy as np
import yaml

import chisurf.base
import chisurf.data
import chisurf.fio
import chisurf.fio.ascii
import chisurf.fio.zipped
import chisurf.fio.fluorescence.fcs.definitions
import chisurf.fio.fluorescence.fcs.asc_alv
import chisurf.fio.fluorescence.fcs.china
import chisurf.fio.fluorescence.fcs.confocor3
import chisurf.fio.fluorescence.fcs.kristine
import chisurf.fio.fluorescence.fcs.pycorrfit
import chisurf.fio.fluorescence.fcs.pq_dat
import chisurf.fio.fluorescence.fcs.fcs_yaml

from chisurf import typing
from chisurf.fio.fluorescence.fcs.definitions import FCSDataset


def read_fcs(
        filename: str,
        data_reader: chisurf.experiments.reader.ExperimentReader = None,
        reader_name: str = 'csv',
        verbose: bool = False,
        experiment: chisurf.experiments.Experiment = None,
        **kwargs
) -> chisurf.data.ExperimentDataCurveGroup:
    """

    Option kristine:

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
    root, _ = os.path.splitext(
        os.path.basename(filename)
    )

    # Files with single curve
    meta_data: typing.Dict = dict()
    ds: typing.List = list()
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
            x = np.array(r[0].pop('correlation_time'))
            y = np.array(r[0].pop('correlation_amplitude'))
            ey = 1. / np.array(r[0].pop('weights'))
            ex = np.ones_like(x)
            meta_data = r[0]
        elif reader_name == 'pycorrfit':
            r = chisurf.fio.fluorescence.fcs.pycorrfit.read_pycorrfit(
                filename=filename,
                verbose=verbose
            )
            x = np.array(r[0].pop('correlation_time'))
            y = np.array(r[0].pop('correlation_amplitude'))
            ey = 1. / np.array(r[0].pop('weights'))
            ex = np.ones_like(x)
            meta_data = r[0]
        name = root
        data_sets.append(
            chisurf.data.DataCurve(
                data_reader=data_reader,
                name=name,
                x=x, y=y, ey=ey, ex=ex,
                experiment=experiment,
                meta_data=meta_data
            )
        )
    # Files with multiple curves per file
    elif reader_name in [
        'china-mat',
        'confocor3',
        'alv',
        'pq.dat',
        'yaml'
    ]:
        if reader_name == 'confocor3':
            ds = chisurf.fio.fluorescence.fcs.confocor3.read_zeiss_fcs(
                filename=filename,
                verbose=verbose
            )
        elif reader_name == 'china-mat':
            ds = chisurf.fio.fluorescence.fcs.china.read_china_mat(
                filename=filename,
                verbose=verbose
            )
        elif reader_name == 'alv':
            ds = chisurf.fio.fluorescence.fcs.asc_alv.read_asc(
                filename=filename,
                verbose=verbose
            )
        elif reader_name == 'pq.dat':
            ds = chisurf.fio.fluorescence.fcs.pq_dat.read_dat(
                filename=filename,
                verbose=verbose
            )
        elif reader_name == 'yaml':
            ds = chisurf.fio.fluorescence.fcs.fcs_yaml.read_yaml(
                filename=filename,
                verbose=verbose
            )
        for r in ds:
            name = root + "_" + r.get('measurement_id')
            x = np.array(r.pop('correlation_time'))
            y = np.array(r.pop('correlation_amplitude'))
            ey = 1. / np.array(r.pop('correlation_amplitude_weights'))
            ex = np.ones_like(x)
            meta_data = dict()
            meta_data.update(r)
            data_sets.append(
                chisurf.data.DataCurve(
                    name=name,
                    data_reader=data_reader,
                    x=x, y=y, ey=ey, ex=ex,
                    meta_data=meta_data
                )
            )
    return chisurf.data.ExperimentDataCurveGroup(data_sets)


def write_single_fcs(
        data_set: chisurf.data.DataCurve,
        fn: str,
        file_type: str
):
    correlation_amplitude = data_set.y
    correlation_time = data_set.x
    correlation_amplitude_uncertainty = data_set.ey
    if file_type == "kristine":
        try:
            aquisition_time = data_set.meta_data["acquisition_time"]
        except KeyError:
            aquisition_time = 1.0
            print("WARNING: Aquisition time not in dataset")
        try:
            mean_countrate = data_set.meta_data["mean_count_rate"]
        except KeyError:
            mean_countrate = 1.0
            print("WARNING: Mean count rate not in dataset")
        chisurf.fio.fluorescence.fcs.kristine.write_kristine(
            filename=fn,
            correlation_amplitude=correlation_amplitude,
            correlation_time=correlation_time,
            correlation_amplitude_uncertainty=correlation_amplitude_uncertainty,
            acquisition_time=aquisition_time,
            mean_countrate=mean_countrate,
        )


def write_fcs(
        data: chisurf.data.ExperimentDataCurveGroup,
        filename: str,
        file_type: str,
        verbose: bool = True
):
    single_fcs_datatypes = ["kristine"]
    multi_fcs_datatypes = ["yaml"]
    # some data types can only hold a single FCS curve
    # in these data types the FCS curves are enumerated and
    # saved separately
    if file_type in single_fcs_datatypes:
        if verbose:
            print("Writing file with single FCS curve per file")
        if len(data) > 1:
            basename, ext = os.path.splitext(filename)
            for i, d in enumerate(data):
                write_single_fcs(
                    data_set=d,
                    fn=basename + "_%02d" % i + ext,
                    file_type=file_type
                )
        else:
            write_single_fcs(
                data_set=data[0],
                fn=filename,
                file_type=file_type
            )
    elif file_type in multi_fcs_datatypes:
        if verbose:
            print("Writing file with multiple FCS curves per file")
        if file_type == "yaml":
            txt = data.to_yaml()
            with chisurf.fio.zipped.open_maybe_zipped(
                filename=filename,
                mode="w"
            ) as fp:
                fp.write(
                    txt
                )


def write_intensity_trace(
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

