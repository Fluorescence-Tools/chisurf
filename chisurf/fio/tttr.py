from __future__ import annotations
from typing import Tuple, List, Dict

import fnmatch
import tempfile
from collections import OrderedDict
import struct

import numba as nb
import numpy as np
import tables

import chisurf


class Photon(tables.IsDescription):
    ROUT = tables.UInt8Col()
    TAC = tables.UInt32Col()
    MT = tables.UInt64Col()
    FileID = tables.UInt16Col()


class Header(tables.IsDescription):
    DINV = tables.UInt16Col() # DataInvalid,
    NROUT = tables.UInt16Col() # Number of routing channels
    MTCLK = tables.Float32Col() # Macro Time clock
    nTAC = tables.UInt16Col()
    FileID = tables.UInt16Col()
    Filename = tables.StringCol(120)
    routine = tables.StringCol(10)


@nb.jit(nopython=True)
def pq_photons(
        b: np.array,
        invert_tac: bool = True
) -> Tuple[
    np.array,
    np.array,
    np.array,
    np.array
]:
    length = (b.shape[0] - 4) // 4
    event = np.zeros(length, dtype=np.uint64)
    mt = np.zeros(length, dtype=np.uint64)
    tac = np.zeros(length, dtype=np.uint16)
    can = np.zeros(length, dtype=np.uint8)

    ov, g, inv = 0, 0, 0
    for i in range(length):
        b3 = b[4*i+3]
        inv = (b3 & 240) >> 4

        b0 = b[4*i]
        b1 = b[4*i+1]
        b2 = b[4*i+2]

        if (inv == 15) & (((b3 & 15) << 8 | b2) == 0):
            ov += 1
        else:
            event[g] = g
            tac[g] = ((b3 & 15) << 8 | b2)
            ovfl = ov * 65536
            mt[g] = ((b1 << 8 |b0) + ovfl)
            if (inv == 15) & (((b3 & 15) << 8 | b2) > 0):
                can[g] = ((b3 & 15) << 8 | b2)+64
            else:
                can[g] = ((b3 & 240) >> 4)
            g += 1

    return g, mt, tac, can


@nb.jit(nopython=True)
def bh132_photons(
        b: np.array
) -> Tuple[
    np.array,
    np.array,
    np.array,
    np.array
]:
    """Get the macros-time, micro-time and the routing channel number of a
    BH132-file contained in a binary numpy-array of 8-bit chars.

    :param b: numpy-array
        a numpy array of chars containing the binary information of a
        BH132-file
    :return: list
        a list containing the number of photons, numpy-array of macros-time
        (64-bit unsigned integers), numpy-array of TAC-values (32-bit unsigned
        integers), numpy-array of channel numbers (8-bit unsigned integers)
    """
    length = (b.shape[0] - 4) // 4
    event = np.zeros(length, dtype=np.uint64)
    mt = np.zeros(length, dtype=np.uint64)
    tac = np.zeros(length, dtype=np.uint32)
    can = np.zeros(length, dtype=np.uint8)

    ov, g, mtov, inv = 0, 0, 0, 0
    for i in range(1, length):
        b3 = b[4*i+3]
        inv = (b3 & 128) >> 7
        mtov = (b3 & 64) >> 6
        b0, b1, b2 = b[4 * i], b[4 * i + 1], b[4 * i + 2]
        if inv == 0 and mtov == 0:
            event[g] = g
            tac[g] = (b3 & 0x0F) << 8 | b2
            ovfl = ov * 4096
            mt[g] = ((b1 & 15) << 8 | b0) + ovfl
            can[g] = (b1 & 0xF0) >> 4
            g += 1
        else:
            if inv == 0 and mtov == 1:
                ov += 1
                event[g] = g
                tac[g] = 4095 - ((b3 & 0x0F) << 8 | b2)
                ovfl = ov * 4096
                mt[g] = ((b1 & 15) << 8 | b0) + ovfl
                can[g] = (b1 & 0xF0) >> 4
                g += 1
            else:
                if inv == 1 and mtov == 1:
                    ov += ((b3 & 15) << 24) | ((b2 << 16) | ((b1 << 8) | b0))
    return g, mt, tac, can


@nb.jit()
def ht3_photons(
        b: np.array
) -> Tuple[
    np.array,
    np.array,
    np.array,
    np.array
]:
    """Processes the time-tagged entries of HT3 files.

    :param b: A binary numpy array of the entries
    :return:
    """
    length = (b.shape[0]) // 4
    event = np.zeros(length, dtype=np.uint64)
    mt = np.zeros(length, dtype=np.uint64)
    tac = np.zeros(length, dtype=np.uint32)
    can = np.zeros(length, dtype=np.uint8)

    ov, g, inv = 0, 0, 0
    for i in range(length):
        b0 = b[4*i]
        b1 = b[4*i+1]
        b2 = b[4*i+2]
        b3 = b[4*i+3]

        inv = (b3 & 254) >> 1
        if inv == 127:
            ov += 1
        else:
            event[g] = g
            tac[g] = ((b3 & 1) << 14 | b2 << 6 | (b1 & 252) >> 2)
            ovfl = ov * 1024
            mt[g] = (((b1 & 3) << 8 | b0) + ovfl)
            can[g] = ((b3 & 254) >> 1)
            if can[g] > 64:
                can[g] -= 64
            g += 1

    return g, mt, tac, can


@nb.jit()
def ht3_sf(
        b: np.array,
        stage: int = 0
) -> Tuple[
    np.array,
    np.array,
    np.array,
    np.array
]:
    """Processes the time-tagged entries of HT3 files that was compressed
    by Suren Felekyans HT3 conversion

    :param b:
    :param stage:
    :return:
    """
    length = (b.shape[0]) // 4
    event = np.zeros(length, dtype=np.uint64)
    mt = np.zeros(length, dtype=np.uint64)
    tac = np.zeros(length, dtype=np.uint32)
    can = np.zeros(length, dtype=np.uint8)

    ov, nph, inv = 0, 0, 0
    for i in range(length):
        b0 = b[4*i]
        b1 = b[4*i+1]
        b2 = b[4*i+2]
        b3 = b[4*i+3]

        inv = (b3 & 254) >> 1
        if inv == 127:
            ov += 1+(b2 << 16 | b1 << 8 | b0)
        else:
            event[nph] = nph
            tac[nph] = ((b3 & 1) << 14 | b2 << 6 | (b1 & 252) >> 2)
            ovfl = ov * 1024
            mt[nph] = (((b1 & 3) << 8 | b0) + ovfl)
            can[nph] = ((b3 & 254) >> 1)
            if can[nph] > 64:
                if stage == 0:
                    can[nph] -= 61
                else:
                    can[nph] -= 64
            nph += 1
    if stage == 1:
        can[0] = 7
        can[nph - 3] = 5
        can[nph - 2] = 4
        can[nph - 1] = 7
    return nph, mt, tac, can


@nb.jit()
def iss_16(
        b: np.array,
        can: np.ndarray,
        tac: np.ndarray,
        mt: np.ndarray,
        length: int,
        step: int,
        phMode: bool,
        offset: int
) -> np.array:
    """
    Reading of ISS-photon format (fcs-measurements)

    :param b:
    :return:
    """
    if step == 1:
        k = 1 if phMode else 0
        for i in range(offset, b.shape[0]):
            ch1 = b[i]
            if phMode:
                mt[k] = mt[k-1] + ch1
                can[k] = 0
                k += 1
            else:
                for j in range(ch1):
                    mt[k] = i
                    can[k] = 0
                    k += 1
    elif step == 2:
        k = 0
        for i in range(offset, b.shape[0], 2):
            ch1 = b[i]
            ch2 = b[i + 1]
            if phMode:
                mt[k] = mt[k-1] + ch1
                can[k] = 0
                k += 1
                mt[k] = mt[k-1] + ch2
                can[k] = 1
                k += 1
            else:
                for j in range(ch1):
                    mt[k] = i
                    can[k] = 0
                    k += 1
                for j in range(ch2):
                    mt[k] = i
                    can[k] = 1
                    k += 1
    return k


@nb.jit()
def iss_32(
        b,
        can,
        tac,
        mt,
        length,
        step,
        phMode: bool,
        offset
):
    """
    Reading of ISS-photon format (fcs-measurements)

    :param data:
    :return:
    """
    #Data is saved as 0: 16-bit or 1: 32-bit
    if step == 1:
        k = 1 if phMode else 0
        for i in range(offset, length):
            ch1 = b[i]
            if phMode:
                mt[k] = mt[k-1] + ch1
                can[k] = 0
                k += 1
            else:
                for j in range(ch1):
                    mt[k] = i
                    can[k] = 0
                    k += 1
    elif step == 2:
        k = 0
        for i in range(offset, length, 2):
            ch1 = b[i]
            ch2 = b[i + 1]
            if phMode:
                mt[k] = mt[k-1] + ch1
                can[k] = 0
                k += 1
                mt[k] = mt[k-1] + ch2
                can[k] = 1
                k += 1
            else:
                for j in range(ch1):
                    mt[k] = i
                    can[k] = 0
                    k += 1
                for j in range(ch2):
                    mt[k] = i
                    can[k] = 1
                    k += 1
    return k


def iss_photons(
        data,
        verbose: bool = chisurf.verbose
) -> Tuple[
    np.array,
    np.array,
    np.array,
    np.array
]:
    """

    # CHANNEL PHOTON MODE (first 2 bytes)
    # in brackets int values
    # H (72)one channel time reading_routine, h (104) one channel photon reading_routine
    # X (88) two channel time reading_routine, x (120) two channel photon reading_routine

    :param data:
    :param kwargs:
    :return:
    """
    step = 1 if (data[1] == 72) or (data[1] == 104) else 2

    #  X (88) two channel time reading_routine, x (120) two channel photon reading_routine
    phMode = False if (data[1] == 72) or (data[1] == 88) else True

    #  Data is saved as 0: 16-bit or 1: 32-bit
    data_32 = data[10]

    if data_32:
        b = data.view(dtype=np.uint32)
        offset = 256 // 2
    else:
        b = data.view(dtype=np.uint16)
        offset = 256 // 4

    if verbose:
        print("Ph-Mode (0/1):\t%s" % phMode)
        print("Nbr. Ch.:\t%s" % step)
        if data_32:
            print("Datasize: 32bit")
        else:
            print("Datasize: 16bit")

    length = (b.shape[0])
    mt = np.zeros(length, dtype=np.uint64)
    tac = np.zeros(length, dtype=np.uint32)
    can = np.zeros(length, dtype=np.uint8)

    if data_32:
        k = iss_32(b, can, tac, mt, length, step, phMode, offset)
    else:
        k = iss_16(b, can, tac, mt, length, step, phMode, offset)

    return k, mt[:k], tac[:k], can[:k]


def bh123_header(
        b: np.array
) -> Tuple[float, bool]:
    bHeader = np.unpackbits(b[0:4])
    conv8le = np.array([128, 64, 32, 16, 8, 4, 2, 1])
    conv24be = np.array([1, 256, 65536])
    bMTclock = bHeader[0:24]
    b0 = np.dot(bMTclock[0:8], conv8le)
    b1 = np.dot(bMTclock[8:16], conv8le)
    b2 = np.dot(bMTclock[16:24], conv8le)
    MTclock = np.dot(np.array([b0, b1, b2]), conv24be) / 10.
    DataInvalid = bool(bHeader[31])
    return MTclock, DataInvalid


def iss_header(
        b
) -> Tuple[float, bool]:
    # acquisition frequency in Hz
    frequency = b[2:6].view(dtype=np.uint32)[0]
    MTclock = 1. / float(frequency) * 1.e9
    return MTclock, False


def ht3_header(
        b
) -> Tuple[float, bool]:
    # TODO doesnt read header properly!!!!!
    frequency = b[2:6].view(dtype=np.uint32)[0]
    MTclock = 1. / float(frequency) * 1.e9
    DataInvalid = 0
    return MTclock, DataInvalid


def make_hdf(
        title: str = None,
        filename: str = None,
        verbose: bool = chisurf.verbose,
        complib: str = chisurf.settings.cs_settings['photons']['complib'],
        **kwargs
):
    """
    Creates a new h5-file/h5-handle for photons

    :param title: The title of the h5-group in which new h5-photon tables are created in
    :param kwargs:
    :return: hdf-file handle (pytables)
    """
    if title is None:
        title = str(chisurf.settings.cs_settings['photons']['title'])
    if filename is None:
        #filename = tempfile.NamedTemporaryFile(
        #    suffix=".photons.h5"
        #)
        _, filename = tempfile.mkstemp(
            suffix=".photons.h5"
        )
    complevel = kwargs.get('', int(chisurf.settings.cs_settings['photons']['complevel']))
    if verbose:
        print("-------------------------------------------")
        print("Make Photon HDF-File")
        print(" Filename: %s" % filename)
        print("-------------------------------------------")
    h5 = tables.open_file(
        filename,
        mode="a",
        title=title
    )
    filters = tables.Filters(
        complib=complib,
        complevel=complevel
    )
    h5.create_group("/", title, 'Name of measurement: %s' % title)
    h5.create_table('/' + title, 'header', description=Header, filters=filters)
    h5.create_table('/' + title, 'photons', description=Photon, filters=filters)

    return h5


def spc2hdf(
        spc_files: List[str],
        routine_name: str = "bh132",
        title: str = "spc",
        verbose: bool = chisurf.verbose,
        filename: str = None,
        **kwargs
):
    """
    Converts BH-SPC files into hdf file format

    :param spc_files: list
        A list of spc-files
    :param routine_name:
        Name of the used reading routine by default "bh132" alternatively
        "bh630_x48"
    :param verbose: bool
        By default False
    : param filename: str
        If no filename is provided a temporary file will be created.
    :param kwargs:
        If the parameter 'filename' is not provided only a temporary hdf (.h5)
        file is created. If the parameter 'title' is provided the data is
        stored in the hdf-group provided by the parameter 'title.
        Otherwise the default 'title' spc is used to store the data within
        the HDF-File.
    :return: tables.file.File

    Examples
    --------
    If the HDF-File doesn't exist it will be created
    To an existing HDF-File simply a new group with the title will be created
    After finished work with the HDF-File it should be closed.

    >>> import chisurf.fio.tttr
    >>> import chisurf.fio.photons
    >>> import glob
    >>> spc_files = glob.glob('./test/data/tttr/BH/132/*.spc')
    >>> h5 = chisurf.fio.tttr.spc2hdf(spc_files, filename='test.h5', title='hGBP1_18D')
    >>> h5 = chisurf.fio.photons.spc2hdf(spc_files, filename='test.h5', title='hGBP1_18D_2')
    >>> h5.close()

    """
    if isinstance(spc_files, str):
        spc_files = [spc_files]
    read = filetypes[routine_name]['read']
    name = filetypes[routine_name]['name']
    nTAC = filetypes[routine_name]['nTAC']
    nROUT = filetypes[routine_name]['nROUT']
    if verbose:
        print("===========================================")
        print(" Reading routine - %s" % name)
    spcs = list()

    fn_ending = filetypes[routine_name]['ending']
    for i, spc_file in enumerate(
            fnmatch.filter(spc_files, "*" + fn_ending)
    ):
        with chisurf.fio.zipped.open_maybe_zipped(
                filename=spc_file, mode='r'
        ) as fp:
            b = np.fromfile(fp, dtype=np.uint8)
            header = read_header(b, routine_name)
            nPh, aMT, aTAC, aROUT = read(b)
            spc = {
                'filename': spc_file,
                'header': header,
                'photon': {
                    'ROUT': aROUT[:nPh],
                    'MT': aMT[:nPh],
                    'TAC': aTAC[:nPh]
                }
            }
            spc['photon']['MT'] += max(spcs[-1]['photon']['MT']) if i > 0 else 0
            spcs.append(spc)
            if verbose:
                print("%s: reading photons..." % spc_file)
                print("-------------------------------------------")
                print(" Macro time clock        : %s" % (header['MTCLK']))
                print(" Number of events        : %i" % ((header['nEvents']) / 4))

    if verbose:
        print("-------------------------------------------")
        print(" Total number of files: %i " % (len(spc_files)))
        print("===========================================")

    h5 = make_hdf(
        title=title,
        filename=filename,
        verbose=verbose,
        **kwargs
    )
    headertable = h5.get_node('/'+title+'/header')
    header = headertable.row
    photontable = h5.get_node('/'+title+'/photons')
    for fileID, spc in enumerate(spcs):
        # Add Header
        header['DINV'] = spc['header']['DINV']
        header['NROUT'] = nROUT
        header['Filename'] = spc['filename']
        header['MTCLK'] = spc['header']['MTCLK']
        header['FileID'] = fileID
        header['routine'] = routine_name
        header['nTAC'] = nTAC
        header.append()
        # Add Photons
        fileID = np.zeros(
            spc['photon']['MT'].shape, np.uint16
        ) + fileID
        photonA = np.rec.array(
            (
                fileID,
                spc['photon']['MT'],
                spc['photon']['ROUT'],
                spc['photon']['TAC']
            )
        )
        photontable.append(photonA)

    photontable.cols.ROUT.create_index()
    h5.flush()
    if verbose:
        print("Reading done!")
    return h5


def read_header(
        binary: np.array,
        routine_name: str
):
    """
    Reads the header-information of binary TTTR-files. The TTTR-files have to be
    passed as numpy array of type numpy.uint8

    :param binary:
        A numpy array containing the data of the SPC or HT3 files
    :param routine_name:
        either bh132 or ht3
    :return:
        An dictionary containing the most important header data.

    Examples
    --------

    Reading Seidel-BID files

    >>> import glob, mfm
    >>> directory = "./test/data/tttr/BH/hGBP1_18D"
    >>> spc_files = glob.glob(directory+'/*.spc')
    >>> b = np.fromfile(spc_files[0], dtype=np.uint8)
    >>> header = chisurf.fio.photons.read_header(b, 'bh132')
    >>> print(header)
    {'MTCLK': 13.6, 'DINV': 0, 'nEvents': 1200000}

    """
    b = binary
    if routine_name == 'bh132':
        MTclock, DataInvalid = chisurf.fio.tttr.bh123_header(b)
    elif routine_name == 'iss':
        MTclock, DataInvalid = chisurf.fio.tttr.iss_header(b)
    elif routine_name == 'ht3':
        MTclock, DataInvalid = chisurf.fio.tttr.ht3_header(b)
    else:
        MTclock, DataInvalid = 1.0, 0
    dHeader = {'DINV': DataInvalid, 'MTCLK': MTclock, 'nEvents': b.shape[0]}
    return dHeader


filetypes = OrderedDict([
    ("hdf", {
        'name': "High density file",
        'ending': '.h5'
    }),
    ("bh132", {
        'name': "Becker-Hickl-132",
        'ending': '.spc',
        'nTAC': 4095,
        'nROUT': 255,
        'read': bh132_photons
    }),
    ("ht3", {
        'name': "PicoQuant-ht3",
        'ending': '.ht3',
        'nTAC': 65535,
        'nROUT': 255,
        'read': ht3_photons
    }),
    ("ht3c", {
        'name': "Suren ht3 compressed",
        'ending': '.ht3',
        'nTAC': 65535,
        'nROUT': 255,
        'read': ht3_sf
    }),
    ("ptu", {
        'name': "PicoQuant-PTU",
        'ending': '.ptu',
        'nTAC': 65535,
        'nROUT': 255,
        'read': pq_photons
    }),
    ("iss", {
        'name': "ISS-fcs",
        'ending': '.fcs',
        'nTAC': 1,
        'nROUT': 2,
        'read': iss_photons
    })
]
)


@nb.jit(nopython=True)
def read_hht3(
        data: np.array,
        n_rec: int,
        version: int = 1
):

    sb = np.zeros(n_rec, dtype=np.uint8)
    mt = np.zeros(n_rec, dtype=np.uint64)
    mi = np.zeros(n_rec, dtype=np.uint32)
    cn = np.zeros(n_rec, dtype=np.uint8)

    overflow_correction = 0
    nph = 0

    t3_wrap = 1024
    for di in data:
        ns = (di & 0b00000000000000000000001111111111) >> 0
        dt = (di & 0b00000001111111111111110000000000) >> 10
        ch = (di & 0b01111110000000000000000000000000) >> 25
        sp = (di & 0b10000000000000000000000000000000) >> 31

        if sp == 1:
            if ch == 0b111111:  # overflow of nsync occured
                if version == 1:
                    overflow_correction += t3_wrap
                else:
                    overflow_correction += t3_wrap * max(1, ns)
            elif (ch >= 0) and (ch <= 15):  # these are markers
                true_n_sync = overflow_correction + ns

                mt[nph] = true_n_sync
                cn[nph] = ch + 1
                sb[nph] = sp
                nph += 1
        else:  # this means a regular input channel
            true_n_sync = overflow_correction + ns

            mt[nph] = true_n_sync
            mi[nph] = dt
            cn[nph] = ch
            sb[nph] = sp
            nph += 1
    return sb[:nph], mt[:nph], mi[:nph], cn[:nph]


@nb.jit(nopython=True)
def read_pht3(
        data: np.array,
        n_rec: int,
        version: int = 1
):

    sb = np.zeros(n_rec, dtype=np.uint8)
    mt = np.zeros(n_rec, dtype=np.uint64)
    mi = np.zeros(n_rec, dtype=np.uint32)
    cn = np.zeros(n_rec, dtype=np.uint8)

    nph = 0
    t3_wrap = 65536
    over_flow = 0
    for di in data:
        ns = (di & 0b00000000000000001111111111111111) >> 0
        dt = (di & 0b00001111111111110000000000000000) >> 16
        ch = (di & 0b11110000000000000000000000000000) >> 28
        sp = 1 if ch == 0b1111 else 0
        n_sync = over_flow + ns
        if 1 <= ch <= 4:
            mt[nph] = n_sync
            mi[nph] = dt
            cn[nph] = ch + 1
            sb[nph] = sp
            nph += 1
            over_flow = 0
        else:
            if sp:
                marker = (di & 0b00000000000011110000000000000000) >> 16
                if marker == 0:
                    over_flow += t3_wrap
                else:
                    mt[nph] = n_sync
                    mi[nph] = dt
                    cn[nph] = marker
                    sb[nph] = sp
                    nph += 1
            else:
                pass  # This is an error

    return sb[:nph], mt[:nph], mi[:nph], cn[:nph]



pq_tag_types = {
    0xFFFF0008: {
        'name': 'tyEmpty8',
        'type': str
    },
    0x00000008: {
        'name': 'tyBool8',
        'type': bool
    },
    0x10000008: {
        'name': 'tyInt8',
        'type': int
    },
    0x11000008: {
        'name': 'tyBitSet64',
        'type': str
    },
    0x12000008: {
        'name': 'tyColor8',
        'type': int
    },
    0x20000008: {
        'name': 'tyFloat8',
        'type': float
    },
    0x21000008: {
        'name': 'tyTDateTime',
        'type': int
    },
    0x2001FFFF: {
        'name': 'tyFloat8Array',
        'type': list,
        'sub_type': float
    },
    0x4001FFFF: {
        'name': 'tyAnsiString',
        'type': list,
        'sub_type': str
    },
    0x4002FFFF: {
        'name': 'tyWideString',
        'type': list,
        'sub_type': str
    },
    0xFFFFFFFF: {
        'name': 'tyBinaryBlob',
        'type': list,
        'sub_type': str
    }
}


pq_hardware = {
    0x01010304: {
        'name': 'rtHydraHarp2T3',
        'HW': 'HydraHarp',
        'args': {'version': 2},
        'read': read_hht3
    },

    0x00010303: {
        'name': 'rtPicoHarpT3',
        'HW': 'PicoHarp',
        'args': {},
        'read': read_pht3
    },
    0x00010203: {
        'name': 'rtPicoHarpT2',
        'HW': 'PicoHarp',
        'args': {},
        'read': None
    },
    0x00010304: {
        'name': 'rtHydraHarpT3',
        'HW': 'HydraHarp',
        'args': {'version': 1},
        'read': read_hht3
    },
    0x00010204: {
        'name': 'rtHydraHarpT2',
        'HW': 'HydraHarp',
        'args': {},
        'read': None
    },
    0x01010204: {
        'name': 'rtHydraHarp2T2',
        'HW': 'HydraHarp',
        'args': {},
        'read': None
    },
    0x00010305: {
        'name': 'rtTimeHarp260NT3',
        'HW': 'TimeHarp260N',
        'args': {},
        'read': None
    },
    0x00010205: {
        'name': 'rtTimeHarp260NT2',
        'HW': 'TimeHarp260N',
        'args': {},
        'read': None
    },
    0x00010306: {
        'name': 'rtTimeHarp260PT3',
        'HW': 'TimeHarp260P',
        'args': {},
        'read': None
    },
    0x00010206: {
        'name': 'rtTimeHarp260PT2',
        'HW': 'TimeHarp260P',
        'args': {},
        'read': None
    }
}


def read_ptu(
        filename: str
) -> Dict:
    """Reads a PicoQuant PTU-file
    
    :param filename:
    :return: 
    
    Example
    -------
    >>> import pylab as p
    >>> filename = 'C://temp/PQSpcm_2017-04-20_13-23-53.ptu'
    >>> filename = "N:/STED_microscope/2017/04/21/PQSpcm_2017-04-21_11-03-37.ptu"
    >>> r = read_ptu(filename)
    >>> y, x = np.histogram(r['micro_time'], bins=range(0, 4096))
    >>> p.semilogy(x[1:], y+1)
    >>> p.show()
    
    """
    with open(filename, 'rb') as fp:
        magic = fp.read(8).strip("\x00")
        tags = dict()
        if magic == 'PQTTTR':
            version = fp.read(8).strip("\x00")
            sf = struct.Struct('32s i I q')
            while True:
                TagIdent, TagIdx, TagType, TagValue = sf.unpack_from(
                    fp.read(sf.size)
                )
                TagIdent = TagIdent.strip("\x00")
                if TagIdx > 0:
                    TagIdent += '(%s)' % TagIdx
                rt = pq_tag_types[TagType]
                if rt['type'] == list:
                    value = fp.read(TagValue)
                    if rt['sub_type'] == str:
                        value = value.strip("\x00")
                else:
                    value = rt['type'](TagValue)
                if TagIdent == "Header_End":
                    break
                else:
                    tags[TagIdent] = value

            hw = tags['TTResultFormat_TTTRRecType']
            n_rec = tags["TTResult_NumberOfRecords"]
            data = np.fromfile(fp, dtype=np.uint32)

            read = pq_hardware[hw]['read']
            args = pq_hardware[hw]['args']
            t3 = read(data, n_rec, **args)
            rt = dict(
                version=version,
                tags=tags,
                special_bits=t3[0],
                macro_time=t3[1],
                micro_time=t3[2],
                channel=t3[3]
            )
            return rt
        else:
            raise ValueError("No PTU file")
    raise ValueError("File not found.")

