# -*- coding: utf-8 -*-
# sdtfile.py

# Copyright (c) 2007-2014, Christoph Gohlke
# Copyright (c) 2007-2014, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read Becker & Hickl SDT files.

SDT files are produced by Becker & Hickl SPCM software. They contain time
correlated single photon counting instrumentation parameters and measurement
data. Currently only the "Setup & Data" and "fcs Data" formats are supported.

`Becker & Hickl GmbH <http://www.becker-hickl.de/>`_ is a manufacturer of
equipment for photon counting.

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2014.09.05

Requirements
------------
* `CPython 2.7 or 3.4 <http://www.python.org>`_
* `Numpy 1.8.2 <http://www.numpy.org>`_

Revisions
---------
2014.09.05
    Fixed reading multiple MEASURE_INFO records.

Notes
-----
The API is not stable yet and might change between revisions.

The `Priithon <http://code.google.com/p/priithon/>`_ package includes an
alternative SDT file reader.

References
----------
(1) W Becker. The bh TCSPC Handbook. Third Edition. Becker & Hickl GmbH 2008.
    pp 401.
(2) SPC_data_file_structure.h header file. Part of the Becker & Hickl
    SPCM software.

Examples
--------
>>> sdt = SdtFile("image.sdt")
>>> sdt.header.revision
588
>>> sdt.info.id[1:-1]
b'SPC Setup & Data File'
>>> int(sdt.measure_info.scan_x)
128
>>> sdt.data[0].shape
(128, 128, 256)
>>> sdt.times[0].shape
(256,)

>>> sdt = SdtFile("fluorescein.sdt")
>>> len(sdt.data)
4
>>> sdt.data[3].shape
(1, 1024)

>>> sdt = SdtFile("fcs.sdt")
>>> sdt.info.id[1:-1]
b'SPC fcs Data File'
>>> sdt.data[0].shape
(512, 512, 256)

"""

from __future__ import division, print_function

import sys

import numpy

__version__ = '2014.09.05'
__docformat__ = 'restructuredtext en'
__all__ = "SdtFile",


class SdtFile(object):
    """Becker & Hickl SDT file.

    Attributes
    ----------
    header : numpy.rec.array of FILE_HEADER structure
        General information about the location of the setup and measurement
        data within the file.
    info : FileInfo
        General information in ASCII format.
    setup : SetupBlock
        Setup block containing all system parameters, display parameters, etc.
    measure_info : numpy.rec.array of MEASURE_INFO structure
        Measurement description blocks.
    block_headers : list of numpy.rec.array of BLOCK_HEADER structure
        Data block headers.
    data : list of 2D numpy arrays
        Photon counts at each curve point.
    times : list of 1D numpy arrays
        Time axes for each data set.

    """
    def __init__(self, arg=None):
        """Initialize instance from filename or open file."""
        if isinstance(arg, basestring):
            with open(arg, 'rb') as fh:
                self._fromfile(fh)
        elif hasattr(arg, 'seek'):
            self._fromfile(arg)

    def _fromfile(self, fh):
        """Initialize instance from open file."""
        # read file header
        self.header = numpy.rec.fromfile(fh, dtype=FILE_HEADER,
                                         shape=1, byteorder='<')[0]
        if self.header.header_valid != 0x5555:
            raise ValueError("not a SDT file")
        if self.header.no_of_data_blocks == 0x7fff:
            self.header.no_of_data_blocks = self.header.reserved1
        elif self.header.no_of_data_blocks > 0x7fff:
            raise ValueError("")

        # read file info
        fh.seek(self.header.info_offset)
        self.info = FileInfo(fh.read(self.header.info_length))
        try:
            if self.info.id not in (b"SPC Setup & Data File",
                                    b"SPC fcs Data File"):
                raise NotImplementedError(
                    "currently not supported:", self.info.id)
        except AttributeError:
            raise ValueError("invalid SDT file info\n", self.info)

        # read setup block
        fh.seek(self.header.setup_offs)
        self.setup = SetupBlock(fh.read(self.header.setup_length))

        # read measurement description blocks
        self.measure_info = []
        dtype = numpy.dtype(MEASURE_INFO)
        if dtype.itemsize > self.header.meas_desc_block_length:
            # TODO: shorten MEASURE_INFO to meas_desc_block_length
            pass
        fh.seek(self.header.meas_desc_block_offset)
        for _ in range(self.header.no_of_meas_desc_blocks):
            self.measure_info.append(
                numpy.rec.fromfile(fh, dtype=dtype, shape=1, byteorder='<'))
            fh.seek(self.header.meas_desc_block_length - dtype.itemsize, 1)

        self.times = []
        self.data = []
        self.block_headers = []

        offset = self.header.data_block_offset
        for _ in range(self.header.no_of_data_blocks):
            # read data block header
            fh.seek(offset)
            bh = numpy.rec.fromfile(
                fh,
                dtype=BLOCK_HEADER,
                shape=1,
                byteorder='<'
            )[0]
            self.block_headers.append(bh)
            # read data block
            mi = self.measure_info[bh.meas_desc_block_no]
            dtype = BlockType(bh.block_type).dtype
            dsize = bh.block_length // dtype.itemsize
            data = numpy.fromfile(fh, dtype=dtype, count=dsize)
            if dsize == mi.scan_x * mi.scan_y * mi.adc_re:
                data = data.reshape(mi.scan_x, mi.scan_y, mi.adc_re)
            elif dsize == mi.image_x * mi.image_y * mi.adc_re:
                data = data.reshape(mi.image_x, mi.image_y, mi.adc_re)
            else:
                data = data.reshape(-1, mi.adc_re[0])
            self.data.append(data)
            # generate time axis
            t = numpy.arange(mi.adc_re, dtype=numpy.float64)
            t *= mi.tac_r / float(mi.tac_g * mi.adc_re)
            self.times.append(t)
            offset = bh.next_block_offs

    def block_measure_info(self, block):
        """Return measure_info record for data block."""
        return self.measure_info[
            self.data_headers[block].meas_desc_block_no
        ]

    def __str__(self):
        """Return string containing all information about SDT file."""
        return "\n\n".join([str(i) for i in (
            self.name, self.header, self.info, self.measure_info,
            self.data_headers, self.data[0].shape)])


class FileInfo(str):
    """File info string and attributes."""
    def __init__(self, value):
        str.__init__(self)
        assert (value.startswith(b'*IDENTIFICATION') and
                value.strip().endswith(b'*END'))
        for line in value.splitlines()[1:-1]:
            try:
                key, val = line.split(b':', 1)
            except Exception:
                pass
            else:
                setattr(self, bytes2str(key.strip().lower()), val.strip())


class SetupBlock(object):
    """Setup block ascii and binary data."""
    def __init__(self, value):
        assert (value.startswith(b'*SETUP') and
                value.strip().endswith(b'*END'))
        i = value.find(b'BIN_PARA_BEGIN')
        if i:
            self.ascii = value[:i]
            self.binary = bytes(value[i+15:-10])
            # todo: parse binary data here
        else:
            self.ascii = value
            self.binary = None

    def __str__(self):
        return self.ascii


class BlockNo(object):
    """The lblock_no field of BLOCK_HEADER."""
    def __init__(self, value):
        self.data = (value & 0xFFFFFF00) >> 24
        self.module = value & 0x000000FF

    def __str__(self):
        return "Data number: %s\nModule number: %s" % (self.data, self.module)

    def __iter__(self):
        return iter((self.data, self.module))


class BlockType(object):
    """The block_type field of BLOCK_HEADER."""
    def __init__(self, value):
        self.mode = BLOCK_CREATION[value & 0xF]
        self.contents = BLOCK_CONTENT[value & 0xF0]
        self.dtype = BLOCK_DTYPE[value & 0xF00]

    def __str__(self):
        return "Mode: %s\nContent: %s\nData Type: %s" % (
            self.mode, self.contents, self.dtype)

    def __iter__(self):
        return iter((self.mode, self.contents, self.dtype))


FILE_HEADER = [
    ('revision', 'i2'),
    ('info_offset', 'i4'),
    ('info_length', 'i2'),
    ('setup_offs', 'i4'),
    ('setup_length', 'i2'),
    ('data_block_offset', 'i4'),
    ('no_of_data_blocks', 'i2'),
    ('data_block_length', 'i4'),
    ('meas_desc_block_offset', 'i4'),
    ('no_of_meas_desc_blocks', 'i2'),
    ('meas_desc_block_length', 'i2'),
    ('header_valid', 'u2'),
    ('reserved1', 'u4'),
    ('reserved2', 'u2'),
    ('chksum', 'u2')]

SETUP_BIN_HDR = [
    ('soft_rev', 'u4'),
    ('para_length', 'u4'),
    ('reserved1', 'u4'),
    ('reserved2', 'u2')]

MEASURE_STOP_INFO = [  # Info collected when measurement finished
    ('status', 'u2'),
    ('flags', 'u2'),
    ('stop_time', 'f4'),
    ('cur_step', 'i4'),
    ('cur_cycle', 'i4'),
    ('cur_page', 'i4'),
    ('min_sync_rate', 'f4'),
    ('min_cfd_rate', 'f4'),
    ('min_tac_rate', 'f4'),
    ('min_adc_rate', 'f4'),
    ('max_sync_rate', 'f4'),
    ('max_cfd_rate', 'f4'),
    ('max_tac_rate', 'f4'),
    ('max_adc_rate', 'f4'),
    ('reserved1', 'i4'),
    ('reserved2', 'f4')]

MEASURE_FCS_INFO = [  # Info collected when FIFO measurement finished
    ('chan', 'u2'),
    ('fcs_decay_calc', 'u2'),
    ('mt_resol', 'u4'),
    ('cortime', 'f4'),
    ('calc_photons', 'u4'),
    ('fcs_points', 'i4'),
    ('end_time', 'f4'),
    ('overruns', 'u2'),
    ('fcs_type', 'u2'),
    ('cross_chan', 'u2'),
    ('mod', 'u2'),
    ('cross_mod', 'u2'),
    ('cross_mt_resol', 'u4')]

HIST_INFO = [  # Extension of MeasFCSInfo for other histograms
    ('fida_time', 'f4'),
    ('filda_time', 'f4'),
    ('fida_points', 'i4'),
    ('filda_points', 'i4'),
    ('mcs_time', 'f4'),
    ('mcs_points', 'i4')]

MEASURE_INFO = [  # Measurement description blocks
    ('time', 'a9'),
    ('date', 'a11'),
    ('mod_ser_no', 'a16'),
    ('meas_mode', 'i2'),
    ('cfd_ll', 'f4'),
    ('cfd_lh', 'f4'),
    ('cfd_zc', 'f4'),
    ('cfd_hf', 'f4'),
    ('syn_zc', 'f4'),
    ('syn_fd', 'i2'),
    ('syn_hf', 'f4'),
    ('tac_r', 'f4'),
    ('tac_g', 'i2'),
    ('tac_of', 'f4'),
    ('tac_ll', 'f4'),
    ('tac_lh', 'f4'),
    ('adc_re', 'i2'),
    ('eal_de', 'i2'),
    ('ncx', 'i2'),
    ('ncy', 'i2'),
    ('page', 'u2'),
    ('col_t', 'f4'),
    ('rep_t', 'f4'),
    ('stopt', 'i2'),
    ('overfl', 'u1'),
    ('use_motor', 'i2'),
    ('steps', 'u2'),
    ('offset', 'f4'),
    ('dither', 'i2'),
    ('incr', 'i2'),
    ('mem_bank', 'i2'),
    ('mod', 'a16'),
    ('syn_th', 'f4'),
    ('dead_time_comp', 'i2'),
    ('polarity_l', 'i2'),
    ('polarity_f', 'i2'),
    ('polarity_p', 'i2'),
    ('linediv', 'i2'),
    ('accumulate', 'i2'),
    ('flbck_y', 'i4'),
    ('flbck_x', 'i4'),
    ('bord_u', 'i4'),
    ('bord_l', 'i4'),
    ('pix_time', 'f4'),
    ('pix_clk', 'i2'),
    ('trigger', 'i2'),
    ('scan_x', 'i4'),
    ('scan_y', 'i4'),
    ('scan_rx', 'i4'),
    ('scan_ry', 'i4'),
    ('fifo_typ', 'i2'),
    ('epx_div', 'i4'),
    ('mod_code', 'u2'),
    ('mod_fpga_ver', 'u2'),
    ('overflow_corr_factor', 'f4'),
    ('adc_zoom', 'i4'),
    ('cycles', 'i4'),
    ('StopInfo', MEASURE_STOP_INFO),
    ('FCSInfo', MEASURE_FCS_INFO),
    ('image_x', 'i4'),
    ('image_y', 'i4'),
    ('image_rx', 'i4'),
    ('image_ry', 'i4'),
    ('xy_gain', 'i2'),
    ('master_clock', 'i2'),
    ('adc_de', 'i2'),
    ('det', 'i2'),
    ('x_axis', 'i2'),
    ('MeasHISTInfo', HIST_INFO)
    ]

BLOCK_HEADER = [
    ('block_no', 'i2'),
    ('data_offs', 'i4'),
    ('next_block_offs', 'i4'),
    ('block_type', 'u2'),
    ('meas_desc_block_no', 'i2'),
    ('lblock_no', 'u4'),
    ('block_length', 'u4')]

BLOCK_CREATION = {  # file_type of creation
    0: 'NOT_USED',
    1: 'MEAS_DATA',
    2: 'FLOW_DATA',
    3: 'MEAS_DATA_FROM_FILE',
    4: 'CALC_DATA',
    5: 'SIM_DATA',
    8: 'FIFO_DATA',
    9: 'FIFO_DATA_FROM_FILE'}

BLOCK_CONTENT = {
    0x0: 'DECAY_BLOCK',
    0x10: 'PAGE_BLOCK',
    0x20: 'FCS_BLOCK',
    0x30: 'FIDA_BLOCK',
    0x40: 'FILDA_BLOCK',
    0x50: 'MCS_BLOCK',
    0x60: 'IMG_BLOCK'}

BLOCK_DTYPE = {  # data type
    0x000: numpy.dtype('<u2'),
    0x100: numpy.dtype('<u4'),
    0x200: numpy.dtype('<f4')}

HEADER_VALID = {
    0x1111: False,
    0x5555: True}

INFO_IDS = {
    b"SPC Setup Script File": "Setup script file_type: setup only",
    b"SPC Setup & Data File": "Normal file_type: setup + data",
    b"SPC DLL Data File": "DLL created: no setup, only data",
    b"SPC Flow Data File": "Continuous Flow file_type: no setup, only data",
    b"SPC fcs Data File":
    "FIFO file_type: setup, data blocks = Decay, fcs, FIDA, FILDA & MCS "
    "curves for each used routing channel"}

if sys.version_info[0] > 2:
    basestring = str
    bytes2str = lambda x: str(x, 'ascii')
else:
    bytes2str = str

if __name__ == "__main__":
    import doctest
    doctest.testmod()