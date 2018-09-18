import fnmatch
from collections import OrderedDict
import tables
import tempfile
from . import _tttrlib
import numpy as np
import mfm
photon_settings = mfm.settings['photons']


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


def bh132_photons(b, invert_tac=True):
    """Get the macro-time, micro-time and the routing channel number of a BH132-file contained in a
    binary numpy-array of 8-bit chars.

    :param b: numpy-array
        a numpy array of chars containing the binary information of a
        BH132-file
    :return: list
        a list containing the number of photons, numpy-array of macro-time (64-bit unsigned integers),
        numpy-array of TAC-values (32-bit unsigned integers), numpy-array of channel numbers (8-bit unsigned integers)
    """
    length = (b.shape[0] - 4) / 4
    event = np.zeros(length, dtype=np.uint64)
    mt = np.zeros(length, dtype=np.uint64)
    tac = np.zeros(length, dtype=np.uint32)
    can = np.zeros(length, dtype=np.uint8)
    g = _tttrlib.beckerMerged(b, can, tac, mt, event, length)
    if invert_tac:
        return g, mt, 4095 - tac, can
    else:
        return g, mt, tac, can


def ht3_photons(b):
    """Get the macro-time, micro-time and the routing channel number of a PQ-HT3-file (version 1) contained in a
    binary numpy-array of 8-bit chars.

    :param b: numpy-array
        a numpy array of chars containing the binary information of a
        BH132-file
    :return: list
        a list containing the number of photons, numpy-array of macro-time (64-bit unsigned integers),
        numpy-array of TAC-values (32-bit unsigned integers), numpy-array of channel numbers (8-bit unsigned integers)
    """
    length = (b.shape[0]) / 4
    event = np.zeros(length, dtype=np.uint64)
    mt = np.zeros(length, dtype=np.uint64)
    tac = np.zeros(length, dtype=np.uint32)
    can = np.zeros(length, dtype=np.uint8)
    number_of_photon = _tttrlib.ht3(b, can, tac, mt, event, length)
    return number_of_photon, mt, tac, can


def iss_photons(data, **kwargs):
    """

    # CHANNEL PHOTON MODE (first 2 bytes)
    # in brackets int values
    # H (72)one channel time mode, h (104) one channel photon mode
    # X (88) two channel time mode, x (120) two channel photon mode

    :param data:
    :param kwargs:
    :return:
    """

    verbose = kwargs.get('verbose', mfm.verbose)
    step = 1 if (data[1] == 72) or (data[1] == 104) else 2

    #  X (88) two channel time mode, x (120) two channel photon mode
    phMode = 0 if (data[1] == 72) or (data[1] == 88) else 1

    #  Data is saved as 0: 16-bit or 1: 32-bit
    data_32 = data[10]

    if data_32:
        b = data.view(dtype=np.uint32)
        offset = 256 / 2
    else:
        b = data.view(dtype=np.uint16)
        offset = 256 / 4

    if verbose:
        print "Ph-Mode (0/1):\t%s" % phMode
        print "Nbr. Ch.:\t%s" % step
        if data_32:
            print "Datasize: 32bit"
        else:
            print "Datasize: 16bit"

    length = (b.shape[0])
    mt = np.zeros(length, dtype=np.uint64)
    tac = np.zeros(length, dtype=np.uint32)
    can = np.zeros(length, dtype=np.uint8)

    if data_32:
        k = _tttrlib.iss_32(b, can, tac, mt, length, step, phMode, offset)
    else:
        k = _tttrlib.iss_16(b, can, tac, mt, length, step, phMode, offset)

    return k, mt[:k], tac[:k], can[:k]


def bh123_header(b):
    bHeader = np.unpackbits(b[0:4])
    conv8le = np.array([128, 64, 32, 16, 8, 4, 2, 1])
    conv24be = np.array([1, 256, 65536])
    bMTclock = bHeader[0:24]
    b0 = np.dot(bMTclock[0:8], conv8le)
    b1 = np.dot(bMTclock[8:16], conv8le)
    b2 = np.dot(bMTclock[16:24], conv8le)
    MTclock = np.dot(np.array([b0, b1, b2]), conv24be) / 10.
    DataInvalid = int(bHeader[31])
    return MTclock, DataInvalid


def iss_header(b):
    # acquisition frequency in Hz
    frequency = b[2:6].view(dtype=np.uint32)[0]
    MTclock = 1. / float(frequency) * 1.e9
    return MTclock, False


def ht3_header(b):
    # TODO doesnt read header properly!!!!!
    frequency = b[2:6].view(dtype=np.uint32)[0]
    MTclock = 1. / float(frequency) * 1.e9
    DataInvalid = 0
    return MTclock, DataInvalid


def make_hdf(**kwargs):
    """
    Creates a new h5-file/h5-handle for photons

    :param title: The title of the h5-group in which new h5-photon tables are created in
    :param kwargs:
    :return: hdf-file handle (pytables)
    """
    title = kwargs.get('title', str(photon_settings['title']))
    filename = kwargs.get('filename', tempfile.mktemp(".photons.h5"))
    verbose = kwargs.get('verbose', mfm.verbose)
    complib = kwargs.get('complib', str(photon_settings['complib']))
    complevel = kwargs.get('complevel', int(photon_settings['complevel']))
    if verbose:
        print("-------------------------------------------")
        print("Make Photon HDF-File")
        print(" Filename: %s" % filename)
        print("-------------------------------------------")

    h5 = tables.open_file(filename, mode="a", title=title)
    filters = tables.Filters(complib=complib, complevel=complevel)
    h5.create_group("/", title, 'Name of measurement: %s' % title)
    h5.create_table('/' + title, 'header', description=Header, filters=filters)
    h5.create_table('/' + title, 'photons', description=Photon, filters=filters)

    return h5


def spc2hdf(spc_files, routine_name="bh132", **kwargs):
    """
    Converts BH-SPC files into hdf file format

    :param spc_files: list
        A list of spc-files
    :param routine_name:
        Name of the used reading routine by default "bh132" alternatively "bh630_x48"
    :param verbose: bool
        By default False
    :param kwargs:
        If the parameter 'filename' is not provided only a temporary hdf (.h5) file is created
        If the parameter 'title' is provided the data is stored in the hdf-group provided by the parameter 'title.
        Otherwise the default 'title' spc is used to store the data within the HDF-File.
    :return: tables.file.File

    Examples
    --------
    If the HDF-File doesn't exist it will be created

    >>> import mfm
    >>> import glob
    >>> directory = "./sample_data/tttr/spc132/hGBP1_18D"
    >>> spc_files = glob.glob(directory+'/*.spc')
    >>> h5 = mfm.io.photons.spc2hdf(spc_files, filename='test.h5', title='hGBP1_18D')

    To an existing HDF-File simply a new group with the title will be created

    >>> h5 = mfm.io.photons.spc2hdf(spc_files, filename='test.h5', title='hGBP1_18D_2')

    After finished work with the HDF-File it should be closed.

    >>> h5.close()
    """
    verbose = kwargs.get('verbose', mfm.verbose)
    title = kwargs.get('title', "spc")

    if isinstance(spc_files, str):
        spc_files = [spc_files]
    read = filetypes[routine_name]['read']
    name = filetypes[routine_name]['name']
    nTAC = filetypes[routine_name]['nTAC']
    nROUT = filetypes[routine_name]['nROUT']
    if verbose:
        print("===========================================")
        print(" Reading routine - %s" % name)
    spcs = []

    fn_ending = filetypes[routine_name]['ending']
    for i, spc_file in enumerate(fnmatch.filter(spc_files, "*" + fn_ending)):
        # TODO: gzip files dont work
        with mfm.io.zipped.open_maybe_zipped(filename=spc_file, mode='r') as fp:
            b = np.fromfile(fp, dtype=np.uint8)
            header = read_header(b, routine_name)
            nPh, aMT, aTAC, aROUT = read(b)
            spc = {'filename': spc_file, 'header': header,
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

    h5 = make_hdf(**kwargs)
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
        fileID = np.zeros(spc['photon']['MT'].shape, np.uint16) + fileID
        photonA = np.rec.array((fileID, spc['photon']['MT'], spc['photon']['ROUT'], spc['photon']['TAC']))
        photontable.append(photonA)

    photontable.cols.ROUT.create_index()
    h5.flush()
    if verbose:
        print("Reading done!")
    return h5


def read_header(binary, routine_name):
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
    >>> directory = "./sample_data/tttr/spc132/hGBP1_18D"
    >>> spc_files = glob.glob(directory+'/*.spc')
    >>> b = np.fromfile(spc_files[0], dtype=np.uint8)
    >>> header = mfm.io.photons.read_header(b, 'bh132')
    >>> print header
    {'MTCLK': 13.6, 'DINV': 0, 'nEvents': 1200000}

    """
    b = binary
    if routine_name == 'bh132':
        MTclock, DataInvalid = mfm.io.tttr.bh123_header(b)
    elif routine_name == 'iss':
        MTclock, DataInvalid = mfm.io.tttr.iss_header(b)
    elif routine_name == 'ht3':
        MTclock, DataInvalid = mfm.io.tttr.ht3_header(b)
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
        'ending': '*.ht3',
        'nTAC': 65535,
        'nROUT': 255,
        'read': ht3_photons
    }),
    ("iss", {
        'name': "ISS-FCS",
        'ending': '.fcs',
        'nTAC': 1,
        'nROUT': 2,
        'read': iss_photons
    })
]
)
