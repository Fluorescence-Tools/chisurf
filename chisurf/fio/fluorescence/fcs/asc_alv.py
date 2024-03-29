"""ALV .ASC files"""
from __future__ import annotations

import pathlib
import warnings
import numpy as np
import csv

import chisurf
import chisurf.fluorescence.fcs

from chisurf import typing
from chisurf.fio.fluorescence.fcs.definitions import FCSDataset



class LoadALVError(BaseException):
    pass


def openASC(path, filename=None):
    """
    Read data from a ALV .ASC files.
    """
    path = pathlib.Path(path)
    if filename is not None:
        warnings.warn("Using `filename` is deprecated.", DeprecationWarning)
        path = path / filename

    with path.open("r", encoding="iso8859_15") as openfile:
        first = openfile.readline()

    # Open special format?
    filetype = first.strip()
    if filetype.count("ALV-7004"):
        return openASC_ALV_7004(path)
    else:
        # last resort
        return openASC_old(path)


def openASC_old(path):
    """ Read data from a .ASC file, created by
        some ALV-6000 correlator.

            ALV-6000/E-WIN Data
            Date :    "2/20/2012"
            ...
            "Correlation"
              1.25000E-004      3.00195E-001
              2.50000E-004      1.13065E-001
              3.75000E-004      7.60367E-002
              5.00000E-004      6.29926E-002
              6.25000E-004      5.34678E-002
              7.50000E-004      4.11506E-002
              8.75000E-004      4.36752E-002
              1.00000E-003      4.63146E-002
              1.12500E-003      3.78226E-002
            ...
              3.35544E+004     -2.05799E-006
              3.77487E+004      4.09032E-006
              4.19430E+004      4.26295E-006
              4.61373E+004      1.40265E-005
              5.03316E+004      1.61766E-005
              5.45259E+004      2.19541E-005
              5.87202E+004      3.26527E-005
              6.29145E+004      2.72920E-005

            "Count Rate"
               1.17188          26.77194
               2.34375          26.85045
               3.51563          27.06382
               4.68750          26.97932
               5.85938          26.73694
               7.03125          27.11332
               8.20313          26.81376
               9.37500          26.82741
              10.54688          26.88801
              11.71875          27.09710
              12.89063          27.13209
              14.06250          27.02200
              15.23438          26.95287
              16.40625          26.75657
              17.57813          26.43056
            ...
             294.14063          27.22597
             295.31250          26.40581
             296.48438          26.33497
             297.65625          25.96457
             298.82813          26.71902

        1. We are interested in the "Correlation" section,
        where the first column denotes tau in ms and the second row the
        correlation signal. Values are separated by a tabulator "\t" (some " ").

        2. We are also interested in the "Count Rate" section. Here the times
        are saved as seconds and not ms like above.

        3. There is some kind of mode where the ALV exports five runs at a
        time and averages them. The sole correlation data is stored in the
        file, but the trace is only stored as average or something.
        So I would not recommend this. However, I added support for this.
        PyCorrFit then only imports the average data.
         ~ Paul, 2012-02-20
        Correlation data starts at "Correlation (Multi, Averaged)".

        Returns:
        [0]:
         An array with tuples containing two elements:
         1st: tau in ms
         2nd: corresponding correlation signal
        [1]:
         Intensity trace:
         1st: time in ms
         2nd: Trace in kHz
        [2]:
         An array with N elements, indicating, how many curves we are opening
         from the file. Elements can be names and must be convertible to
         strings.
    """
    filename = path.name
    with path.open("r", encoding="iso8859_15") as openfile:
        Alldata = openfile.readlines()
    # End of trace
    EndT = Alldata.__len__()
    # Correlation function
    # Find out where the correlation function is
    for i in np.arange(len(Alldata)):
        if Alldata[i].startswith('Mode'):
            mode = Alldata[i][5:].strip(' ":').strip().strip('"')
            single_strings = ["a-ch0", "a-ch1", "auto ch0", "auto ch1",
                              "fast auto ch0", "fast auto ch1",
                              ]
            if (mode.lower().count('single') or
                    mode.lower().strip() in single_strings):
                single = True
                channel = mode.split(" ")[-1]
            else:
                # dual
                single = False
            # accc ?
            if mode.lower().count("cross") == 1:
                accc = "CC"
            else:
                accc = "AC"
        if Alldata[i].startswith('"Correlation'):
            # This tells us if there is only one curve or if there are
            # multiple curves with an average.
            if (Alldata[i].strip().lower() ==
                    '"correlation (multi, averaged)"'):
                multidata = True
            else:
                multidata = False
        if Alldata[i].startswith('"Correlation"'):
            # Start of correlation function
            StartC = i+1
        if Alldata[i].startswith('"Correlation (Multi, Averaged)"'):
            # Start of AVERAGED correlation function !!!
            # There are several curves now.
            StartC = i+2
        if Alldata[i].replace(" ", "").lower().strip() == '"countrate"':
            # takes care of "Count Rate" and "Countrate"
            # End of correlation function
            EndC = i-1
            # Start of trace (goes until end of file)
            StartT = i+1
        if Alldata[i].startswith('Monitor Diode'):
            EndT = i-1
    # Get the header
    Namedata = Alldata[StartC-1: StartC]
    # Define *curvelist*
    curvelist = csv.reader(Namedata, delimiter='\t').__next__()
    if len(curvelist) <= 2:
        # Then we have just one single correlation curve
        curvelist = [""]
    else:
        # We have a number of correlation curves. We need to specify
        # names for them. We take these names from the headings.
        # Lag times not in the list
        curvelist.remove(curvelist[0])
        # Last column is empty
        curvelist.remove(curvelist[-1])
    # Correlation function
    Truedata = Alldata[StartC: EndC]
    readdata = csv.reader(Truedata, delimiter='\t')
    # Add lists to *data* according to the length of *curvelist*
    data = [[]]*len(curvelist)
    # Work through the rows in the read data
    for row in readdata:
        for i in np.arange(len(curvelist)):
            if len(row) > 0:
                data[i].append((np.float(row[0]), np.float(row[i+1])))
    # Trace
    # Trace is stored in two columns
    # 1st column: time [s]
    # 2nd column: trace [kHz]
    # Get the trace
    Tracedata = Alldata[StartT: EndT]
    timefactor = 1000  # because we want ms instead of s
    readtrace = csv.reader(Tracedata, delimiter='\t')
    trace = list()
    trace2 = list()
    # Work through the rows
    for row in readtrace:
        # time in ms, countrate
        trace.append(list())
        trace[0].append((np.float(row[0])*timefactor,
                         np.float(row[1])))
        # Only trace[0] contains the trace!
        for i in np.arange(len(curvelist)-1):
            trace.append(list())
            trace[i+1].append((np.float(row[0])*timefactor, 0))
        if not single:
            k = len(curvelist)/2
            if int(k) != k:
                print("Problem with ALV data. Single mode not recognized.")
            # presumably dual mode. There is a second trace
            # time in ms, countrate
            trace2.append(list())
            trace2[0].append((np.float(row[0])*timefactor,
                              np.float(row[2])))
            # Only trace2[0] contains the trace!
            for i in np.arange(len(curvelist)-1):
                trace2.append(list())
                trace2[i+1].append((np.float(row[0])*timefactor, 0))

    # group the resulting curves
    corrlist = list()
    tracelist = list()
    typelist = list()

    if single:
        # We only have several runs and one average
        # split the trace into len(curvelist)-1 equal parts
        if multidata:
            nav = 1
        else:
            nav = 0
        splittrace = mysplit(trace[0], len(curvelist)-nav)
        i = 0
        for t in range(len(curvelist)):
            typ = curvelist[t]
            if typ.lower()[:7] == "average":
                typelist.append("{} average".format(channel))
                corrlist.append(np.array(data[t]))
                tracelist.append(np.array(trace[0]))
            else:
                typelist.append("{} {}".format(accc, channel))
                corrlist.append(np.array(data[t]))
                tracelist.append(splittrace[i])
                i += 1
    elif accc == "AC":
        # Dual mode, autocorrelation
        # We now have two averages and two different channels.
        # We now have two traces.
        # The data is assembled in blocks. That means the first block
        # contains an average and the data of channel 0 and the second
        # block contains data and average of channel 1. We can thus
        # handle the data from 0 to len(curvelist)/2 and from
        # len(curvelist)/2 to len(curvelist) as two separate data sets.
        # CHANNEL 0
        if multidata:
            nav = 1
        else:
            nav = 0
        channel = "CH0"
        splittrace = mysplit(trace[0], len(curvelist)/2-nav)
        i = 0
        for t in range(int(len(curvelist)/2)):
            typ = curvelist[t]
            if typ.lower()[:7] == "average":
                typelist.append("{} average".format(channel))
                corrlist.append(np.array(data[t]))
                tracelist.append(np.array(trace[0]))
            else:
                typelist.append("{} {}".format(accc, channel))
                corrlist.append(np.array(data[t]))
                tracelist.append(splittrace[i])
                i += 1
        # CHANNEL 1
        channel = "CH1"
        splittrace2 = mysplit(trace2[0], len(curvelist)/2-nav)
        i = 0
        for t in range(int(len(curvelist)/2), int(len(curvelist))):
            typ = curvelist[t]
            if typ.lower()[:7] == "average":
                typelist.append("{} average".format(channel))
                corrlist.append(np.array(data[t]))
                tracelist.append(np.array(trace2[0]))
            else:
                typelist.append("{} {}".format(accc, channel))
                corrlist.append(np.array(data[t]))
                tracelist.append(splittrace2[i])
                i += 1
    elif accc == "CC":
        if multidata:
            nav = 1
        else:
            nav = 0
        # Dual mode, cross-correlation
        channel = "CC01"
        splittrace = mysplit(trace[0], len(curvelist)/2-nav)
        splittrace2 = mysplit(trace2[0], len(curvelist)/2-nav)
        i = 0
        for t in range(int(len(curvelist)/2)):
            typ = curvelist[t]
            if typ.lower()[:7] == "average":
                typelist.append("{} average".format(channel))
                corrlist.append(np.array(data[t]))
                tracelist.append([np.array(trace[0]),
                                  np.array(trace2[0])])
            else:
                typelist.append("{} {}".format(accc, channel))
                corrlist.append(np.array(data[t]))
                tracelist.append([splittrace[i], splittrace2[i]])
                i += 1
        # CHANNEL 1
        channel = "CC10"
        i = 0
        for t in range(int(len(curvelist)/2), int(len(curvelist))):
            typ = curvelist[t]
            if typ.lower()[:7] == "average":
                typelist.append("{} average".format(channel))
                corrlist.append(np.array(data[t]))
                # order must be the same as above
                tracelist.append([np.array(trace[0]),
                                  np.array(trace2[0])])
            else:
                typelist.append("{} {}".format(accc, channel))
                corrlist.append(np.array(data[t]))
                # order must be the same as above
                tracelist.append([splittrace[i], splittrace2[i]])
                i += 1
    else:
        print("Could not detect data file format for: {}".format(filename))
        corrlist = np.array(data)
        tracelist = np.array(trace)
        typelist = curvelist

    dictionary = dict()
    dictionary["Correlation"] = corrlist
    dictionary["Trace"] = tracelist
    dictionary["Type"] = typelist
    filelist = list()
    for i in curvelist:
        filelist.append(filename)
    dictionary["Filename"] = filelist

    return dictionary


def openASC_ALV_7004(
        path: pathlib.Path
) -> typing.Dict:
    """
    Opens ALV file format with header information "ALV-7004/USB"

    This is a single-run file format.
    - data is identified by 4*"\t"
    - count rate is identified by string (also "countrate")
    - allzero-correlations are removed

    "Correlation"
      2.50000E-005     -9.45478E-001     -1.00000E+000      5.22761E-002      3.05477E-002
      5.00000E-005      6.73734E-001     -2.59938E-001      3.17894E-002      4.24466E-002
      7.50000E-005      5.30716E-001      3.21605E-001      5.91051E-002      2.93061E-002
      1.00000E-004      3.33292E-001      1.97860E-001      3.24102E-002      3.32379E-002
      1.25000E-004      2.42538E-001      1.19988E-001      4.37917E-002      3.05477E-002
      1.50000E-004      1.86396E-001      1.23318E-001      5.66218E-002      2.25806E-002
      1.75000E-004      1.73836E-001      8.53991E-002      4.64819E-002      3.46865E-002
      2.00000E-004      1.48080E-001      9.35377E-002      4.37917E-002      4.17223E-002
    [...]
      1.00663E+004      2.80967E-005     -2.23975E-005     -7.08272E-005      5.70470E-005
      1.09052E+004      9.40185E-005      2.76261E-004      1.29745E-004      2.39958E-004
      1.17441E+004     -2.82103E-004     -1.97386E-004     -2.88753E-004     -2.60987E-004
      1.25829E+004      1.42069E-004      3.82018E-004      6.03932E-005      5.40363E-004

    "Count Rate"
           0.11719         141.83165          81.54211         141.83165          81.54211
           0.23438         133.70215          77.90344         133.70215          77.90344
           0.35156         129.67148          74.58858         129.67148          74.58858
           0.46875         134.57133          79.53957         134.57133          79.53957
    [...]
          29.29688         143.78307          79.06236         143.78307          79.06236
          29.41406         154.80135          82.87147         154.80135          82.87147
          29.53125         187.43013          89.61197         187.43013          89.61197
          29.64844         137.82655          77.71597         137.82655          77.71597
    [...]


    """
    filename = path.name
    with path.open("r", encoding="iso8859_15") as openfile:
        Alldata = openfile.readlines()

    # Find the different arrays
    # correlation array: "  "
    # trace array: "       "
    allcorr = []
    alltrac = []
    count_rates = []
    i = 0
    intrace = False
    mode = False
    duration_sec = None

    for item in Alldata:
        if item.lower().strip().strip('"') == "count rate":
            intrace = True
            continue
        elif item.count("Mode"):
            mode = item.split(":")[1].strip().strip('" ').lower()
        elif "Duration" in item:
            front, back = item.split(":")
            if "[s]" in front:
                mul = 1.0
            elif "[ms]" in front:
                mul = 1000.0
            else:
                mul = 1.0
            duration_sec = mul * float(back.strip())
        elif "MeanCR" in item:
            count_rates.append(
                float(item.split(":")[1].strip())
            )
        i += 1
        if item.count("\t") == 4:
            if intrace:
                it = item.split("\t")
                it = [float(t.strip()) for t in it]
                alltrac.append(it)
            else:
                ic = item.split("\t")
                ic = [float(c.strip()) for c in ic]
                allcorr.append(ic)
    allcorr = np.array(allcorr)
    alltrac = np.array(alltrac)

    tau = allcorr[:, 0]
    time = alltrac[:, 0] * 1000
    lenc = allcorr.shape[0]
    lent = alltrac.shape[0]

    # Traces
    trace1 = np.zeros((lent, 2), dtype=np.float_)
    trace1[:, 0] = time
    trace1[:, 1] = alltrac[:, 1]
    trace2 = trace1.copy()
    trace2[:, 1] = alltrac[:, 2]
    trace3 = trace1.copy()
    trace3[:, 1] = alltrac[:, 3]
    trace4 = trace1.copy()
    trace4[:, 1] = alltrac[:, 4]

    # Correlations
    corr1 = np.zeros((lenc, 2), dtype=np.float_)
    corr1[:, 0] = tau
    corr1[:, 1] = allcorr[:, 1]
    corr2 = corr1.copy()
    corr2[:, 1] = allcorr[:, 2]
    corr3 = corr1.copy()
    corr3[:, 1] = allcorr[:, 3]
    corr4 = corr1.copy()
    corr4[:, 1] = allcorr[:, 4]

    typelist = []
    corrlist = []
    tracelist = []
    filelist = []
    if mode == False:
        raise LoadALVError("Undetermined ALV file mode: {}".format(path))
    # Go through all modes
    if mode == "a-ch0+1  c-ch0/1+1/0":
        # For some reason, the traces columns show the values
        # of channel 1 and 2 in channels 3 and 4.
        if not (np.allclose(trace1, trace3, rtol=.01) and
                np.allclose(trace2, trace4, rtol=.01)):
            raise LoadALVError("Unexpected data format: {}".format(path))
        if not np.allclose(corr1[:, 1], 0):
            corrlist.append(corr1)
            filelist.append(filename)
            tracelist.append(trace1)
            typelist.append("AC1")
        if not np.allclose(corr2[:, 1], 0):
            corrlist.append(corr2)
            filelist.append(filename)
            tracelist.append(trace2)
            typelist.append("AC2")
        if not np.allclose(corr3[:, 1], 0):
            corrlist.append(corr3)
            filelist.append(filename)
            tracelist.append([trace1, trace2])
            typelist.append("CC12")
        if not np.allclose(corr4[:, 1], 0):
            corrlist.append(corr4)
            filelist.append(filename)
            tracelist.append([trace1, trace2])
            typelist.append("CC21")
    elif mode in ["a-ch0", "a-ch0 a-"]:
        if not (np.allclose(trace2[:, 1], 0) and
                np.allclose(trace3[:, 1], 0) and
                np.allclose(trace4[:, 1], 0) and
                np.allclose(corr2[:, 1], 0) and
                np.allclose(corr3[:, 1], 0) and
                np.allclose(corr4[:, 1], 0)):
            raise LoadALVError("Unexpected data format: {}".format(path))
        corrlist.append(corr1)
        filelist.append(filename)
        tracelist.append(trace1)
        typelist.append("AC")
    elif mode in ["a-ch1", "a-ch1 a-"]:
        if not (np.allclose(trace1[:, 1], 0) and
                np.allclose(trace3[:, 1], 0) and
                np.allclose(trace4[:, 1], 0) and
                np.allclose(corr1[:, 1], 0) and
                np.allclose(corr3[:, 1], 0) and
                np.allclose(corr4[:, 1], 0)):
            raise LoadALVError("Unexpected data format: {}".format(path))
        corrlist.append(corr2)
        filelist.append(filename)
        tracelist.append(trace2)
        typelist.append("AC")
    elif mode in ["a-ch2", "a- a-ch2"]:
        if not (np.allclose(trace1[:, 1], 0) and
                np.allclose(trace2[:, 1], 0) and
                np.allclose(trace4[:, 1], 0) and
                np.allclose(corr1[:, 1], 0) and
                np.allclose(corr2[:, 1], 0) and
                np.allclose(corr4[:, 1], 0)):
            raise LoadALVError("Unexpected data format: {}".format(path))
        corrlist.append(corr3)
        filelist.append(filename)
        tracelist.append(trace3)
        typelist.append("AC")
    elif mode in ["a-ch3", "a- a-ch3"]:
        if not (np.allclose(trace1[:, 1], 0) and
                np.allclose(trace2[:, 1], 0) and
                np.allclose(trace3[:, 1], 0) and
                np.allclose(corr1[:, 1], 0) and
                np.allclose(corr2[:, 1], 0) and
                np.allclose(corr3[:, 1], 0)):
            raise LoadALVError("Unexpected data format: {}".format(path))
        corrlist.append(corr4)
        filelist.append(filename)
        tracelist.append(trace4)
        typelist.append("AC")
    else:
        msg = "ALV mode '{}' not implemented yet.".format(mode)
        raise NotImplementedError(msg)

    dictionary = dict()
    dictionary["Correlation"] = np.array(corrlist)
    dictionary["Trace"] = np.array(tracelist)
    dictionary["Type"] = typelist
    dictionary["Filename"] = filelist
    dictionary["Duration"] = duration_sec
    dictionary["Count rates"] = count_rates
    return dictionary


def mysplit(a, n):
    """
       Split a trace into n equal parts by interpolation.
       The signal average is preserved, but the signal variance will
       decrease.
    """
    if n <= 1:
        return [np.array(a)]
    a = np.array(a)
    N = len(a)
    lensplit = np.int(np.ceil(N/n))

    # xp is actually rounded -> recalculate
    xp = np.linspace(a[:, 0][0], a[:, 0][-1], N,  endpoint=True)

    # let xp start at zero
    xp -= a[:, 0][0]
    yp = a[:, 1]

    # time frame for each new curve
    #dx = xp[-1]/n

    # perform interpolation of new trace
    x, newstep = np.linspace(0, xp[-1], lensplit*n,
                             endpoint=True, retstep=True)
    # interpolating reduces the variance and possibly changes the avg
    y = np.interp(x, xp, yp)

    data = np.zeros((lensplit*n, 2))
    data[:, 0] = x + newstep
    # make sure that the average stays the same:
    data[:, 1] = y - np.average(y) + np.average(yp)
    return np.split(data, n)


def read_asc_header(
        filename: str
) -> str:
    """This returns the ASC header, i.e., the asc file content
    till the first correlation curve


    Parameters
    ----------
    filename : str
        The filename of the asc file

    Returns
    -------
    str
        The header of the asc file

    """
    with open(filename, "r", encoding='iso-8859-1') as fp:
        lines = fp.readlines()
        header_end = 0
        for i, line in enumerate(lines):
            if "Correlation" in line:
                header_end = i
                break
        return "".join(lines[:header_end])


def read_asc(
        filename: str,
        verbose: bool = False
) -> typing.List[FCSDataset]:
    if verbose:
        print("Reading ALV .asc from file: ", filename)
    d = openASC(filename)
    correlations = list()

    for i, correlation in enumerate(d['Correlation']):
        correlation_time = correlation[:, 0]
        correlation_amplitude = correlation[:, 1] + 1.0
        if isinstance(d['Trace'][i], list):
            intensity_time = np.vstack(
                [
                    d['Trace'][i][0][:, 0],
                    d['Trace'][i][1][:, 0]
                ]
            ).T
            intensity = np.vstack(
                [
                    d['Trace'][i][0][:, 1],
                    d['Trace'][i][1][:, 1]
                ]
            ).T
        else:
            intensity_time = d['Trace'][i][:, 0]
            intensity = d['Trace'][i][:, 1]

        # We want the intensity trace in seconds
        intensity_time /= 1000.0
        try:
            aquisition_time = d["Duration"]
        except KeyError:
            aquisition_time = intensity_time[-1]
        try:
            mean_count_rate = d["Count rates"][i]
        except (KeyError, IndexError):
            mean_count_rate = np.sum(intensity) / aquisition_time

        # in cross correlations the count rate of the correlation
        # channels are set to zero. In this case use the mean of the
        # two intensity traces
        if mean_count_rate == 0.0:
            aquisition_time = np.mean(intensity_time[-1])
            mean_count_rate = np.mean(intensity[-1])
        w = 1. / chisurf.fluorescence.fcs.noise(
            correlation_time,
            correlation_amplitude,
            aquisition_time,
            mean_count_rate=mean_count_rate
        )
        corr: FCSDataset = {
                'filename': filename,
                'measurement_id': "%s_%s" % (filename, i),
                'acquisition_time': aquisition_time,
                'mean_count_rate': mean_count_rate,
                'correlation_times': correlation_time.tolist(),
                'correlation_amplitudes': correlation_amplitude.tolist(),
                'correlation_amplitude_weights': w.tolist(),
                'intensity_trace_times': intensity_time.tolist(),
                'intensity_trace': intensity.tolist(),
                'meta_data': {
                    'header': read_asc_header(
                        filename=filename
                    )
                }
            }
        correlations.append(
            corr
        )
    return correlations


def write_asc(
        filename: str,
        correlation_amplitudes: typing.Tuple[np.ndarray],
        correlation_times: typing.Tuple[np.ndarray],
        time_traces: typing.Tuple[np.ndarray],
        mean_countrates: typing.Tuple[float],
        meta_data: typing.Dict,
        acquisition_time: float,
        verbose: bool = True
) -> None:
    pass
