from __future__ import annotations

import numpy as np
import os

import chisurf.data
import chisurf.fio
import chisurf.fluorescence

from chisurf import typing


class THDReader:
    """PicoQUant THD file reader.
    
    Reads PicoQUant THD files (TCSPC data) and extracts histogram data.
    
    Attributes
    ----------
    filepath : str
        Path to the THD file
    header : dict
        Dictionary containing header information
    histogram_data : numpy.ndarray
        Histogram data extracted from the file
    """
    def __init__(self, filepath):
        """Initialize THDReader with a file path.
        
        Parameters
        ----------
        filepath : str
            Path to the THD file
        """
        self.filepath = filepath
        self.header = {}
        self.histogram_data = None
        self._read_file()

    def _read_file(self):
        """Read the THD file and extract header and histogram data."""
        with open(self.filepath, "rb") as f:
            content = f.read()

        # Parse header fields
        self.header["Device"] = content[0:16].decode('ascii', errors='ignore').strip('\x00')
        self.header["Version"] = content[16:24].decode('ascii', errors='ignore').strip('\x00')
        self.header["Software"] = content[24:56].decode('ascii', errors='ignore').strip('\x00')
        self.header["Software Version"] = content[56:72].decode('ascii', errors='ignore').strip('\x00')
        self.header["Date and Time"] = content[72:104].decode('ascii', errors='ignore').strip('\x00')
        self.header["Title"] = content[104:136].decode('ascii', errors='ignore').strip('\x00')

        # Histogram starts after 688
        histogram_start = 688
        histogram_bytes = content[histogram_start:]
        num_bins = len(histogram_bytes) // 4  # 4 bytes per bin
        self.histogram_data = np.frombuffer(histogram_bytes, dtype='<u4', count=num_bins)

    def get_header(self):
        """Return the header information.
        
        Returns
        -------
        dict
            Dictionary containing header information
        """
        return self.header

    def get_histogram(self):
        """Return the histogram data.
        
        Returns
        -------
        numpy.ndarray
            Histogram data
        """
        return self.histogram_data

    def get_histogram_as_dict(self):
        """Return the histogram data as a dictionary.
        
        Returns
        -------
        dict
            Dictionary with bin indices as keys and counts as values
        """
        return dict(enumerate(self.get_histogram()))

    def __repr__(self):
        """Return string representation of the THDReader."""
        return f"<THDReader '{os.path.basename(self.filepath)}' with {len(self.histogram_data)} raw bins>"


def read_tcspc_thd(
        filename: str = None,
        dt: float = 1.0,
        rebin: typing.Tuple[int, int] = (1, 1),
        experiment: chisurf.experiments.Experiment = None,
        *args,
        **kwargs
) -> chisurf.data.DataCurveGroup:
    """Read TCSPC data from a THD file.
    
    Parameters
    ----------
    filename : str, optional
        Path to the THD file
    dt : float, optional
        Time resolution in nanoseconds, by default 1.0
    rebin : tuple of int, optional
        Rebinning factors for x and y axes, by default (1, 1)
    experiment : chisurf.experiments.Experiment, optional
        Experiment object, by default None
    
    Returns
    -------
    chisurf.data.DataCurveGroup
        DataCurveGroup containing the TCSPC data
    """
    # Load data
    rebin_x, rebin_y = rebin
    
    # Read THD file
    thd_reader = THDReader(filename)
    y = thd_reader.get_histogram()
    
    # Create time axis
    x = np.arange(len(y), dtype=np.float64) * dt
    
    # Rebin data if needed
    if rebin_y > 1:
        # Calculate new length after rebinning
        new_length = len(y) // rebin_y
        # Reshape and sum along the rebinning axis
        y = y[:new_length * rebin_y].reshape(-1, rebin_y).sum(axis=1)
        # Adjust time axis
        x = x[:new_length * rebin_y:rebin_y]
    
    # Calculate error (assuming Poisson statistics)
    ey = np.sqrt(y)
    
    # Create data curve
    data = chisurf.data.DataCurve(
        x=x,
        y=y,
        ex=np.zeros_like(x),
        ey=ey,
        experiment=experiment,
        name=filename,
        **kwargs
    )
    data.filename = filename
    
    # Create data group
    data_group = chisurf.data.DataCurveGroup([data], filename)
    
    return data_group