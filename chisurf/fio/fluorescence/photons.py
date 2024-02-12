"""Read TTTR files.

TTTR files contain time-tagged-time-resolved records typically recorded
in single-molecule experiments and on confocal laser scanning setups.

:Author:
  `Thomas-Otavio Peulen <http://tpeulen.github.io>`_

Requirements
------------

Revisions
---------

Notes
-----
The API is not stable yet and might change between revisions.

References
----------

Examples
--------

"""


from __future__ import annotations
from chisurf import typing

import tempfile
import numpy as np
import tables
import tttrlib

import chisurf
from . import tttr


def read_burst_ids(
        filenames: typing.List[str],
        stack_files: bool = True
) -> np.array:
    """
    Reads Seidel-BID files and returns a list of numpy
    arrays. Each numpy array contains the indexes of
    the photons of the burst. These indexes can be used
    to slice a photon-stream.

    Seidel BID-files only contain the first and the
    last photon of the Burst and not all photons of
    the burst. Thus, the Seidel BID-files have to be
    converted to array-type objects containing all
    photons of the burst to be able to use standard
    Python slicing syntax to select photons.

    :param filenames:
        filename pointing to a Seidel BID-file
    :param stack_files: bool
        If stack is True the returned list is stacked
        and the numbering of the bursts is made continuous.
        This is the default behavior.
    :return:

    Examples
    --------

    >>> import chisurf.fio, glob
    >>> directory = "./test/data/tttr/spc132/hGBP1_18D/burstwise_All 0.1200#30\BID"
    >>> files = glob.glob(directory+'/*.bst')
    >>> bids = chisurf.fio.photons.read_burst_ids(files)
    >>> bids[1]
    array([20384, 20385, 20386, 20387, 20388, 20389, 20390, 20391, 20392,
       20393, 20394, 20395, 20396, 20397, 20398, 20399, 20400, 20401,
       20402, 20403, 20404, 20405, 20406, 20407, 20408, 20409, 20410,
       20411, 20412, 20413, 20414, 20415, 20416, 20417, 20418, 20419,
       20420, 20421, 20422, 20423, 20424, 20425, 20426, 20427, 20428,
       20429, 20430, 20431, 20432, 20433, 20434, 20435, 20436, 20437,
       20438, 20439, 20440, 20441, 20442, 20443, 20444, 20445, 20446,
       20447, 20448, 20449, 20450, 20451, 20452, 20453, 20454, 20455,
       20456, 20457, 20458, 20459, 20460, 20461, 20462, 20463, 20464,
       20465, 20466, 20467, 20468, 20469, 20470, 20471, 20472, 20473,
       20474, 20475, 20476, 20477, 20478, 20479, 20480, 20481, 20482,
       20483, 20484, 20485, 20486, 20487, 20488, 20489, 20490, 20491,
       20492, 20493, 20494, 20495, 20496, 20497, 20498, 20499, 20500,
       20501, 20502, 20503, 20504, 20505, 20506, 20507, 20508, 20509,
       20510, 20511, 20512, 20513, 20514, 20515, 20516, 20517, 20518,
       20519, 20520, 20521, 20522, 20523, 20524, 20525, 20526, 20527,
       20528, 20529, 20530, 20531, 20532, 20533, 20534, 20535, 20536,
       20537, 20538, 20539, 20540, 20541, 20542, 20543, 20544, 20545,
       20546, 20547, 20548, 20549, 20550, 20551, 20552, 20553, 20554,
       20555, 20556, 20557, 20558, 20559, 20560, 20561, 20562, 20563,
       20564, 20565, 20566, 20567, 20568, 20569, 20570, 20571, 20572,
       20573, 20574, 20575, 20576, 20577, 20578, 20579, 20580, 20581,
       20582, 20583, 20584, 20585, 20586, 20587, 20588, 20589, 20590,
       20591, 20592, 20593, 20594, 20595, 20596, 20597, 20598, 20599,
       20600, 20601, 20602, 20603, 20604, 20605, 20606, 20607, 20608,
       20609, 20610, 20611, 20612, 20613, 20614, 20615, 20616, 20617,
       20618, 20619, 20620, 20621, 20622, 20623, 20624, 20625, 20626,
       20627, 20628, 20629, 20630, 20631, 20632, 20633, 20634, 20635,
       20636, 20637, 20638, 20639, 20640, 20641, 20642, 20643, 20644,
       20645, 20646, 20647, 20648, 20649, 20650, 20651, 20652, 20653,
       20654, 20655], dtype=uint64)

    """
    if isinstance(filenames, str):
        filenames = [filenames]
    re = dict()
    for file in filenames:
        bids = np.loadtxt(file, dtype=np.int32)
        re[file] = [np.arange(bid[0], bid[1], dtype=np.int32) for bid in bids]
    if not stack_files:
        return re
    else:
        b = re[filenames[0]]
        if len(filenames) > 1:
            for i, fn in enumerate(filenames[1:]):
                offset = re[filenames[i]][-1][-1]
                for j in range(len(re[fn])):
                    b.append(re[fn][j] + offset)
        return b


class Photons(object):

    """

    :param p_object:
        Is either a list of filenames or a single string containing the path to
        one file. If the first argument is n
    :param reading_routine:
        The file type of the files passed using the first argument (p_object)
        is specified using the 'file_type' parameter. This string is either
        'hdf' or 'bh132', 'bh630_x48', 'ht3', 'iss'. If the file type is not
        an hdf file the files are temporarily converted to hdf-files to
        guarantee a consistent interface.
    :param kwargs:
    :return:

    Examples
    --------

    >>> import chisurf.fio, glob
    >>> directory = './test/data/tttr/BH/'
    >>> spc_files = glob.glob(directory+'/BH_SPC132.spc')
    >>> photons = chisurf.fio.photons.Photons(spc_files, file_type="bh132")
    >>> print(photons)
    File-type: bh132
    Filename(s):
            ./test/data/tttr/spc132/hGBP1_18D\m000.spc
            ./test/data/tttr/spc132/hGBP1_18D\m001.spc
            ./test/data/tttr/spc132/hGBP1_18D\m002.spc
            ./test/data/tttr/spc132/hGBP1_18D\m003.spc
    nTAC:   4095
    nROUT:  255
    MTCLK [ms]:     1.36000003815e-05

    >>> print(photons[:10])
    File-type: None
    Filename(s):    None
    nTAC:   4095
    nROUT:  255
    MTCLK [ms]:     1.36000003815e-05
    """

    def __init__(
            self,
            p_object,
            reading_routine: str = None,
            verbose: bool = chisurf.verbose
    ):
        self._tttrs = None
        self._h5 = None
        self._sample_name_hdf_tp = 'spc'
        _, self._h5_tempfile = tempfile.mkstemp('.h5')
        if isinstance(p_object, tables.file.File):
            self._h5 = p_object
            self._filenames = []
        else:
            if isinstance(p_object, str):
                self._filenames = [p_object]
            elif isinstance(p_object, list):
                p_object.sort()
                self._filenames = p_object

            # determine reading routine
            if reading_routine == 'hdf':
                self._h5 = tables.open_file(
                    p_object[0], mode='r'
                )
            elif reading_routine == 'iss':
                self._h5 = tttr.spc2hdf(
                    self._filenames,
                    routine_name=reading_routine,
                    filename=self._h5_tempfile
                )
            else:
                # tttrlib
                if reading_routine == 'bh132':
                    spcs = []
                    for i, filename in enumerate(self._filenames):
                        t = tttrlib.TTTR(filename, 'SPC-130')
                        header = t.get_header()
                        number_of_tac_channels = header.number_of_micro_time_channels
                        if i > 0:
                            t.shift_macro_time(spcs[i-1]['photon']['MT'][-1])
                        spcs.append(
                            tttr.make_spc_dict(
                                macro_times=t.get_macro_time(),
                                micro_times=t.get_micro_time(),
                                routing_channels=t.get_routing_channel(),
                                macro_time_resolution=header.macro_time_resolution,
                                number_of_events=t.get_n_valid_events(),
                                event_types=t.get_event_type(),
                                filename=filename
                            )
                        )
                    if len(self._filenames) > 0:
                        self._tttrs = spcs
                        self._h5 = chisurf.fio.fluorescence.tttr.make_tp_photon_hdf(
                            title=self._sample_name_hdf_tp,
                            filename=self._h5_tempfile,
                            verbose=verbose,
                            routine_name=reading_routine,
                            number_of_tac_channels=number_of_tac_channels,
                            number_of_routing_channels=255,
                            spcs=spcs
                        )
        self.verbose = verbose
        self._photon_array = None
        self.filetype = reading_routine
        self._selection = None
        self._number_of_tac_channels = None
        self._number_of_routing_channels = None
        self._MTCLK = None

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(
            self,
            v
    ):
        self._selection = v
        self._photon_array = self.photon_table.read_coordinates(
            self._selection
        )

    @property
    def photon_table(self) -> tables.Table:
        sample = self._h5.get_node('/' + self._sample_name_hdf_tp)
        return sample.photons

    @property
    def photon_array(self):
        if self._photon_array is None:
            self._photon_array = self.photon_table.read()
        return self._photon_array

    @property
    def filenames(self) -> typing.List[str]:
        """
        Original filename of the data
        """
        return self._filenames

    @property
    def measurement_time(self) -> float:
        """
        Total measurement time in seconds?
        """
        return self.macro_times[-1] * self.mt_clk

    @property
    def dt(self) -> float:
        """
        The micro-time calibration
        """
        return self.mt_clk / self.n_tac

    @property
    def shape(self) -> typing.Tuple[int]:
        return self.routing_channels.shape

    @property
    def nPh(self) -> int:
        """
        Total number of photons
        """
        return self.routing_channels.shape[0]

    @property
    def routing_channels(self) -> np.array:
        """
        Array containing the routing channel of the photons
        """
        return self.photon_array['ROUT']

    @property
    def micro_times(self) -> np.array:
        """
        Array containing the micro-time clock counts (TAC) of the photons
        """
        return self.photon_array['TAC']

    @property
    def event_types(self) -> np.array:
        """
        Array containing the event types
        """
        return self.photon_array['EVENT']

    @property
    def macro_times(self) -> np.array:
        """
        Array containing the macros-time clock counts of the photons
        """
        return self.photon_array['MT']

    @property
    def n_tac(self) -> int:
        """
        Number of TAC channels
        """
        sample = self._h5.get_node('/' + self._sample_name_hdf_tp)
        return sample.header[0]['nTAC']

    @property
    def mt_clk(self) -> float:
        """
        Macro-time clock of the data (time between the macrotime events
        in milli-seconds)
        """
        sample = self._h5.get_node('/' + self._sample_name_hdf_tp)
        return sample.header[0]['MTCLK'] if self._MTCLK is None else self._MTCLK

    def read_where(
            self,
            selection: np.ndarray
    ) -> Photons:
        """This function uses the pytables selection syntax

        :param selection:
        :return:
        """
        selection = np.intersect1d(
            self._selection,
            selection,
            assume_unique=True
        )

        re = Photons(None)
        re._h5 = self._h5
        re.selection = selection
        self._selection = self._selection
        self._number_of_tac_channels = self._number_of_tac_channels
        self._number_of_routing_channels = self._number_of_routing_channels
        self._MTCLK = self._MTCLK

        return re

    def take(
            self,
            keys
    ) -> Photons:
        re = Photons(None)
        if isinstance(self.selection, np.ndarray):
            selection = np.intersect1d(
                self._selection,
                keys,
                assume_unique=True
            )
        else:
            selection = keys
        re._h5 = self._h5
        re.selection = selection
        self.filetype = self.filetype
        self._sample_name_hdf_tp = self._sample_name_hdf_tp
        self._selection = self._selection
        self._number_of_tac_channels = self._number_of_tac_channels
        self._number_of_routing_channels = self._number_of_routing_channels
        self._MTCLK = self._MTCLK
        return re

    def __str__(self):
        s = ""
        s += "File-type: %s\n" % self.filetype
        s += "Filename(s):\t"
        if len(self.filenames) > 0:
            s += "\n"
            for fn in self.filenames:
                s += "\t" + fn + "\n"
        else:
            s += "None\n"
        s += "nTAC:\t%d\n" % self.n_tac
        s += "MTCLK [ms]:\t%s\n" % self.mt_clk
        return s

    # def __del__(self):
    #     if self.h5 is not None:
    #         self.h5.close()
    #         #if self.filetype in ['bh132', 'bh630_x48']:
    #         #    os.unlink(self._tempfile)

    def __len__(self):
        return self.nPh

    def __getitem__(self, key):
        if isinstance(key, int):
            key = np.array(key)
        else:
            start = 0 if key.start is None else key.start
            stop = len(self) if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            key = np.arange(start, stop, step)
        return self.take(keys=key)

