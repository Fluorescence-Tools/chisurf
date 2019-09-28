from __future__ import annotations
from typing import List, Tuple

import csv
import os
import numpy as np

import mfm


def save_xy(
        filename: str,
        x: np.array,
        y: np.array,
        verbose: bool = mfm.verbose,
        fmt: str = "%.3f\t%.3f",
        header_string: str = None
) -> None:
    """
    Saves data x, y to file in format (csv). x and y should have the same length.

    :param filename: string
        Target filename
    :param x: array
    :param y: array
    :param verbose: bool
    :param header_string:
    :param fmt:
    """
    if verbose:
        print("Writing histogram to file: %s" % filename)
    with open(filename, 'w') as fp:
        if header_string is not None:
            fp.write(header_string)
        for p in zip(x, y):
            fp.write(fmt % (p[0], p[1]))


def load_xy(
        filename: str,
        verbose: bool = mfm.verbose,
        usecols: Tuple[int, int] = None,
        skiprows: int = 0,
        delimiter: str = "\t"
) -> Tuple[
    np.array,
    np.array
]:
    if usecols is None:
        usecols = [0, 1]
    if verbose:
        print("Loading file: ", filename)
    data = np.loadtxt(
        filename,
        skiprows=skiprows,
        usecols=usecols,
        delimiter=delimiter
    )
    return data.T[0], data.T[1]


class Csv(object):

    """
    Csv is a class to handle coma separated value files.

    :param kwargs:

    Examples
    --------
    Two-column data

    >>> import mfm.io.ascii
    >>> csv = mfm.io.ascii.Csv(skiprows=10)
    >>> filename = './sample_data/tcspc/ibh_sample/Decay_577D.txt'
    >>> csv.load(filename)
    >>> csv.data
    array([  1.00000000e+00,   2.00000000e+00,   3.00000000e+00, ...,
     4.09400000e+03,   4.09500000e+03,   4.09600000e+03])
     >>> csv.data_y
     array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
     >>> max(csv.data_y)
     50010.0

    One-column Jordi data

    >>> csv = mfm.io.ascii.Csv(skiprows=11)
    >>> filename =  './sample_data/tcspc/ibh_sample/Decay_577D.txt'
    >>> csv.load(filename)
    >>> csv.data_x
    array([  1.00000000e+00,   2.00000000e+00,   3.00000000e+00, ...,
     4.09400000e+03,   4.09500000e+03,   4.09600000e+03])
     >>> csv.data_y
     array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
     >>> max(csv.data_y)
     50010.0

    """

    def __init__(
            self,
            *args,
            filename: str = None,
            colspecs: List[int] = None,
            use_header: bool = False,
            x_on: bool = True,
            y_on: bool = True,
            col_x: int = 0,
            col_y: int = 1,
            col_ex: int = 2,
            col_ey: int = 3,
            reverse: bool = False,
            error_x_on: bool = False,
            directory: str = '.',
            skiprows: int = 9,
            verbose: bool = mfm.verbose,
            file_type: str = 'csv',
            **kwargs
    ):
        """

        :param args:
        :param filename:
        :param colspecs:
        :param use_header:
        :param x_on:
        :param y_on:
        :param col_x:
        :param col_y:
        :param col_ex:
        :param col_ey:
        :param reverse:
        :param error_x_on:
        :param directory:
        :param skiprows:
        :param verbose:
        :param file_type:
        :param kwargs:
        """
        self._filename = filename
        self.use_header = use_header
        self.x_on = x_on
        self.error_y_on = y_on
        self.col_x = col_x
        self.col_y = col_y
        self.col_ex = col_ex
        self.col_ey = col_ey
        self.reverse = reverse
        self.error_x_on = error_x_on
        self.directory = directory
        self.skiprows = skiprows
        self.file_type = file_type
        self.verbose = verbose

        self._header = None

        if colspecs is None:
            colspecs = (15, 17, 17)
        self.colspecs = colspecs

        self._data = kwargs.get('data', None)
        if isinstance(filename, str):
            self.load(
                filename
            )

    @property
    def filename(
            self
    ) -> str:
        """
        The currently open filename (after setting this parameter the file is opened)
        """
        return self._filename

    def load(
            self,
            filename: str,
            skiprows: int = None,
            use_header: bool = None,
            verbose: bool = mfm.verbose,
            delimiter: str = None,
            file_type: str = None,
            **kwargs
    ):
        """
        This method loads a filename to the `Csv` object
        :param filename: string specifying the file
        :param skiprows: number of rows to skip in the file. By default the
        value of the instance is used
        :param verbose: The method is verbose if verbose is set to True of
        the verbose attribute of the instance is
        True.
        """
        if file_type is None:
            file_type = self.file_type
        if use_header is None:
            use_header = self.use_header
        if skiprows is None:
            skiprows = self.skiprows

        # process header
        header = 'infer' if use_header else None
        if use_header:
            with open(file=filename, mode='r') as fp:
                for _ in range(skiprows):
                    fp.readline()
                header_line = fp.readline()
            self._header = header_line.split(delimiter)
            skiprows += 1

        if os.path.isfile(filename):
            self.directory = os.path.dirname(filename)
            self._filename = filename
            colspecs = self.colspecs
            if verbose:
                print("Reading: {}".format(filename))
                print("Skip rows: {}".format(skiprows))
                print("Use header: {}".format(use_header))
                with open(filename, 'r') as csvfile:
                    print(csvfile.read()[:512])

            if file_type == 'csv':
                if delimiter is None:
                    with open(filename, 'r') as csvfile:
                        for _ in range(skiprows):
                            csvfile.readline()
                        dialect = csv.Sniffer().sniff(
                            csvfile.read(), delimiters=';,|\t '
                        )
                        delimiter = dialect.delimiter
                d = np.genfromtxt(
                    fname=filename,
                    delimiter=delimiter,
                    skip_header=skiprows,
                    **kwargs
                )
            else:
                d = np.genfromtxt(
                    skip_header=skiprows,
                    fname=filename,
                    delimiter=colspecs,
                    names=header,
                    **kwargs
                )
            self._data = d
        else:
            raise IOError

    def save(
            self,
            data: np.ndarray,
            filename: str,
            delimiter: str = '\t',
            file_type: str = 'txt',
            header: str = ''
    ):
        self._data = data
        if self.verbose:
            s = """Saving
            ------
            filename: %s
            file_type: %s
            delimiter: %s
            Object-type: %s
            """ % (filename, file_type, delimiter, type(data))
            print(s)
        # if isinstance(data, mfm.curve.Curve):
        #     d = np.array(data[:])
        # elif isinstance(data, np.ndarray):
        #     d = data
        # else:
        #     d = np.array(data)
        #
        if file_type == 'txt':
            np.savetxt(
                filename,
                data.T,
                delimiter=delimiter,
                header=header
            )
        if file_type == 'npy':
            np.save(
                filename,
                data.T
            )

    @property
    def n_cols(
            self
    ) -> int:
        """
        The number of columns
        """
        return self._data.shape[1]

    @property
    def n_rows(
            self
    ) -> int:
        """
        The number of rows
        """
        return self._data.shape[0]

    @property
    def data(
            self
    ) -> np.array:
        """
        Numpy array of the data
        """
        if self.reverse:
            return np.array(self._data, dtype=np.float64).T[::-1]
        else:
            return np.array(self._data, dtype=np.float64).T

    @property
    def header(
            self
    ) -> List[str]:
        """
        A list of the column headers
        """
        if self.use_header is not None:
            header = self._header
        else:
            header = range(self.data.shape[1])
        return [str(i) for i in header]

    @property
    def n_points(
            self
    ) -> int:
        """
        The number of data points corresponds to the number of rows
        :py:attribute`.CSV.n_rows`
        """
        return self.n_rows


