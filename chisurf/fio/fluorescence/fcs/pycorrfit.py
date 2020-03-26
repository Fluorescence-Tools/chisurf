
"""CSV files"""
import csv
import pathlib
import warnings
import os

import numpy as np

import chisurf.fluorescence.fcs

from chisurf import typing
from chisurf.fio.fluorescence.fcs.definitions import FCSDataset


def openCSV(path, filename=None):
    """
    Read relevant data from a file looking like this:
        [...]
        # Comment
        # Data type: Autocorrelation
        [...]
        1.000000e-006   3.052373e-001
        1.020961e-006   3.052288e-001
        1.042361e-006   3.052201e-001
        1.064209e-006   3.052113e-001
        1.086516e-006   3.052023e-001
        1.109290e-006   3.051931e-001
        [...]
        # BEGIN TRACE
        [...]
        10.852761   31.41818
        12.058624   31.1271
        13.264486   31.27305
        14.470348   31.33442
        15.676211   31.15861
        16.882074   31.08564
        18.087936   31.21335
        [...]

    The correlation part could also look like this:
        # Channel (tau [s])    Experimental correlation    Fitted correlation    Residuals      Weights [model function]
        2.0000000000e-07    1.5649271000e-01    1.5380094370e-01    2.6917663029e-03    7.3158300646e-03
        4.0000000000e-07    1.4751239000e-01    1.5257959602e-01    -5.0672060199e-03    5.8123579098e-03
        6.0000000000e-07    1.5145113000e-01    1.5137624642e-01    7.4883584881e-05    8.5622019656e-03
        8.0000000000e-07    1.5661088000e-01    1.5019053433e-01    6.4203456659e-03    6.8098486549e-03
        1.0000000000e-06    1.5456273000e-01    1.4902210818e-01    5.5406218229e-03    7.2476381023e-03
        1.2000000000e-06    1.3293905000e-01    1.4787062503e-01    -1.4931575028e-02    6.9861494246e-03
        1.4000000000e-06    1.4715790000e-01    1.4673575040e-01    4.2214960494e-04    6.9810206017e-03
        1.6000000000e-06    1.5247520000e-01    1.4561715797e-01    6.8580420325e-03    6.6680066656e-03
        1.8000000000e-06    1.4703974000e-01    1.4451452937e-01    2.5252106284e-03    6.3299717550e-03
    In that case we are also importing the weights.

    Data type:
    If Data type is "Cross-correlation", we will try to import
    two traces after "# BEGIN SECOND TRACE"

    1st section:
     First column denotes tau in seconds and the second row the
     correlation signal.
    2nd section:
     First column denotes tau in seconds and the second row the
     intensity trace in kHz.


    Returns:
    1. A list with tuples containing two elements:
       1st: tau in ms
       2nd: corresponding correlation signal
    2. None - usually is the trace, but the trace is not saved in
              the PyCorrFit .csv format.
    3. A list with one element, indicating, that we are opening only
       one correlation curve.
    """
    path = pathlib.Path(path)
    if filename is not None:
        warnings.warn("Using `filename` is deprecated.", DeprecationWarning)
        path = path / filename
    filename = path.name
    # Check if the file is correlation data
    with path.open("r", encoding='utf-8') as fd:
        firstline = fd.readline()
        if firstline.lower().count("this is not correlation data") > 0:
            return None

    # Define what will happen to the file
    timefactor = 1000  # because we want ms instead of s
    csvfile = path.open('r', encoding='utf-8')
    readdata = csv.reader(csvfile, delimiter=',')
    data = list()
    weights = list()
    weightname = "external"
    trace = None
    traceA = None
    DataType = "AC"  # May be changed
    numtraces = 0
    duration = 1.0
    count_rates = list()
    prev_row = None
    for row in readdata:
        if len(row) == 0 or len(str(row[0]).strip()) == 0:
            # Do nothing with empty/whitespace lines
            pass
            # Beware that the len(row) statement has to be called first
            # (before the len(str(row[0]).strip()) ). Otherwise some
            # error would be raised.
        elif str(row[0])[:12].lower() == "# Type AC/CC".lower():
            corrtype = str(row[0])[12:].strip().strip(":").strip()
            if corrtype[:17].lower() == "cross-correlation":
                # We will later try to import a second trace
                DataType = "CC"
                DataType += corrtype[17:].strip()
            elif corrtype[0:15].lower() == "autocorrelation":
                DataType = "AC"
                DataType += corrtype[15:].strip()
        elif str(row[0])[0:13].upper() == '# BEGIN TRACE':
            # Correlation is over. We have a trace
            corr = np.array(data)
            data = list()
            numtraces = 1
        elif str(row[0])[0:20].upper() == '# BEGIN SECOND TRACE':
            # First trace is over. We have a second trace
            traceA = np.array(data)
            data = list()
            numtraces = 2
        elif "avg. signal" in row[0]:
            count_rates.append(
                float(row[0].split("\t")[1].strip())
            )
        # Exclude commentaries
        elif "#   duration [s]	" in row[0]:
            duration = float(row[0].split("\t")[1].strip())
        elif str(row[0])[0:1] != '#':
            # Read the 1st section
            # On Windows we had problems importing nan values that
            # had some white-spaces around them. Therefore: strip()
            # As of version 0.7.8 we are supporting white space
            # separated values as well
            if len(row) == 1:
                row = row[0].split()
            data.append((np.float(row[0].strip())*timefactor,
                         np.float(row[1].strip())))
            if len(row) == 5:
                # this has to be correlation with weights
                weights.append(np.float(row[4].strip()))
                if weightname == "external":
                    try:
                        weightname = "ext. " + \
                            prev_row[0].split("Weights")[1].split(
                                "[")[1].split("]")[0]
                    except:
                        pass
        prev_row = row
    # Collect the rest of the trace, if there is any:
    rest = np.array(data)
    if numtraces == 0:
        corr = rest
    elif numtraces >= 1:
        trace = rest
    del data
    # Remove any NaN numbers from thearray
    # Explanation:
    # np.isnan(data)
    #  finds the position of NaNs in the array (True positions); 2D array, bool
    # any(1)
    #  finds the rows that have True in them; 1D array, bool
    # ~
    #  negates them and is given as an argument (array type bool) to
    #  select which items we want.
    corr = corr[~np.isnan(corr).any(1)]
    # Also check for infinities.
    corr = corr[~np.isinf(corr).any(1)]
    csvfile.close()
    Traces = list()
    # Set correct trace data for import
    if numtraces == 1 and DataType[:2] == "AC":
        Traces.append(trace)
    elif numtraces == 2 and DataType[:2] == "CC":
        Traces.append([traceA, trace])
    elif numtraces == 1 and DataType[:2] == "CC":
        # Should not happen, but for convenience:
        Traces.append([trace, trace])
    else:
        Traces.append(None)
    dictionary = dict()
    dictionary["Correlation"] = [corr]
    dictionary["Trace"] = Traces
    dictionary["Type"] = [DataType]
    dictionary["Filename"] = filename
    dictionary["Duration"] = duration
    dictionary["Count rates"] = count_rates
    if len(weights) != 0:
        dictionary["Weight"] = [np.array(weights)]
        dictionary["Weight Name"] = [weightname]
    return dictionary


def read_pycorrfit(
        filename: str,
        verbose: bool = False
) -> typing.List[FCSDataset]:
    if verbose:
        print("Reading PyCorrFit from file: ", filename)
    d = openCSV(filename)
    correlations = list()
    for i, correlation in enumerate(d['Correlation']):
        r = dict()
        correlation_time = correlation[:, 0]
        correlation_amplitude = correlation[:, 1]
        if d['Trace'][i] is not None:
            intenstiy_trace_time = (d['Trace'][i][0][:, 0] / 1000.0).tolist()
            intensity_trace_ch1 = (d['Trace'][i][0][:, 1]).tolist()
            r.update(
                {
                    'intensity_trace_time_ch1': intenstiy_trace_time,
                    'intensity_trace_ch1': intensity_trace_ch1
                }
            )
        aquisition_time = d["Duration"]
        r.update(
            {
                'filename': filename,
                'measurement_id': "%s_%s" % (
                    os.path.splitext(os.path.basename(d['Filename']))[0], i
                ),
                'correlation_time': correlation_time,
                'correlation_amplitude': correlation_amplitude,
                'acquisition_time': aquisition_time,
                'mean_count_rate': np.mean(d["Count rates"][i]),
            }
        )
        w = 1. / chisurf.fluorescence.fcs.noise(
                    times=correlation_time,
                    correlation=correlation_amplitude,
                    measurement_duration=r['acquisition_time'],
                    mean_count_rate=r['mean_count_rate'],
                )
        r.update(
            {
                'weights': w.tolist()
            }
        )
        correlations.append(r)
    return correlations