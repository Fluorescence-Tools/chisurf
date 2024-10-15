from __future__ import annotations

import os

import numpy as np

import chisurf.base
import chisurf.fio
import chisurf.fio.ascii
import chisurf.fio as io
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


def make_curve_kwargs(
        r: FCSDataset
):
    """This function takes a *FCSDataset* and prepares a dictionary
    that can be used to initialize a *DataCurve*

    Parameters
    ----------
    r : FCSDataset

    Returns
    -------
    dict
        A dictionary that can be used to initialize a *DataCurve*.

    """
    # a set of main keys are used to define FCS curves
    # with associated weights
    main_keys = [
        'measurement_id',
        'correlation_times',
        'correlation_amplitudes',
        'correlation_amplitude_weights'
    ]
    filename = r.get('filename')
    x = np.array(r.get('correlation_times'))
    # all other keys are stored in the meta data dict
    meta_data = r.pop('meta_data', dict())
    for key in r.keys():
        if key not in main_keys:
            meta_data[key] = r.get(key, None)
    meta_data["experiment_name"] = "FCS"
    meta_data["filename"] = filename
    curve_kwargs = {
        "name": r.get('measurement_id'),
        "x": x,
        "y": np.array(r.get('correlation_amplitudes')),
        "ex": np.ones_like(x),
        "ey": 1. / np.array(r.get('correlation_amplitude_weights')),
        "filename": filename,
        "meta_data": meta_data,
        "load_filename_on_init": False
    }
    return curve_kwargs


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

    name_reader = {
        'confocor3': chisurf.fio.fluorescence.fcs.confocor3.read_zeiss_fcs,
        'china-mat': chisurf.fio.fluorescence.fcs.china.read_china_mat,
        'alv': chisurf.fio.fluorescence.fcs.asc_alv.read_asc,
        'pq.dat': chisurf.fio.fluorescence.fcs.pq_dat.read_dat,
        'yaml': chisurf.fio.fluorescence.fcs.fcs_yaml.read_yaml,
        'kristine': chisurf.fio.fluorescence.fcs.kristine.read_kristine,
        'pycorrfit': chisurf.fio.fluorescence.fcs.pycorrfit.read_pycorrfit
    }

    data_sets: typing.List[chisurf.data.DataCurve] = list()
    ds: typing.List[FCSDataset] = list()
    # Files that contain single FCS curves
    if reader_name == 'csv':
        csv = chisurf.fio.ascii.Csv()
        csv.load(
            filename=filename,
            verbose=chisurf.verbose,
            **kwargs
        )
        x, y = csv.data[0], csv.data[1]
        ey = csv.data[2]
    # Files with multiple curves per file
    elif reader_name in [
        'china-mat',
        'confocor3',
        'alv',
        'pq.dat',
        'yaml',
        'kristine',
        'pycorrfit'
    ]:
        reader = name_reader[reader_name]
        ds: typing.List[FCSDataset] = reader(
            filename=filename,
            verbose=verbose
        )
    for r in ds:
        data_sets.append(
            chisurf.data.DataCurve(
                **make_curve_kwargs(r)
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
        verbose: bool = True,
        mode: str = 'w'
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
            txt = data.to_yaml(
                remove_protected=True,
                convert_values_to_elementary=True
            )
            with io.zipped.open_maybe_zipped(
                filename=filename,
                mode=mode
            ) as fp:
                fp.write(
                    txt
                )


