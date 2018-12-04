import os
import numpy as np
import pandas as pd

import mfm


def save_xy(filename, x, y, verbose=False, fmt="%.3f\t%.3f", header=None):
    """
    Saves data x, y to file in format (csv). x and y
    should have the same lenght.

    :param filename: string
        Target filename
    :param x: array
    :param y: array
    :param verbose: bool
    :param fmt:
    """
    if verbose:
        print("Writing histogram to file: %s" % filename)
    fp = open(filename, 'w')
    if header is not None:
        fp.write(header)
    for p in zip(x, y):
        fp.write(fmt % (p[0], p[1]))
    fp.close()


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

    def __init__(self, *args, **kwargs):
        """

        :param kwargs:
        :return:
        """
        self._filename = ""
        self._x = kwargs.get('x', None)
        self._y = kwargs.get('y', None)
        self._ex = kwargs.get('ex', None)
        self._ey = kwargs.get('ey', None)

        self.use_header = kwargs.get('use_header', False)
        self.x_on = kwargs.get('x_on', True)
        self.error_y_on = kwargs.get('y_on', False)
        self.col_x = kwargs.get('col_x', 0)
        self.col_y = kwargs.get('col_y', 1)
        self.col_ex = kwargs.get('col_ex', 2)
        self.col_ey = kwargs.get('col_ex', 3)
        self.reverse = kwargs.get('reverse', False)
        self.error_x_on = kwargs.get('error_x_on', False)
        self.directory = kwargs.get('directory', '.')
        self.skiprows = kwargs.get('skiprows', 9)
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self.mode = kwargs.get('mode', 'csv')
        self.colspecs = kwargs.get('colspecs', '(16,33), (34,51)')

        self._data = kwargs.get('data', pd.DataFrame())
        self._filename = kwargs.get('filename', "")

    @property
    def filename(self):
        """
        The currently open filename (after setting this parameter the file is opened)
        """
        return self._filename

    #@filename.setter
    #def filename(self, v):
    #    self._filename = v
    #    self.load(v)

    def load(self, filename, **kwargs):
        """
        This method loads a filename to the `Csv` object
        :param filename: string specifying the file
        :param skiprows: number of rows to skip in the file. By default the value of the instance is used
        :param verbose: The method is verbose if verbose is set to True of the verbose attribute of the instance is
        True.
        """
        verbose = kwargs.get('verbose', self.verbose)
        use_header = kwargs.pop('use_header', self.use_header)
        skiprows = kwargs.get('skiprows', self.skiprows)
        header = 'infer' if use_header else None
        self._filename = kwargs.get('filename', "")

        if os.path.isfile(filename):
            self.directory = os.path.dirname(filename)
            self._filename = filename
            colspecs = self.colspecs
            if verbose:
                print("Reading: {}".format(filename))
                print("Skip rows: {}".format(skiprows))
                print("Use header: {}".format(use_header))
            if self.mode == 'csv':
                df = pd.read_csv(filename, sep='[\s,;,,]',
                                 header=header,
                                 engine='python',
                                 **kwargs)
            else:
                df = pd.read_fwf(filename, colspecs=colspecs, header=header, **kwargs)
            self._data = df
        else:
            raise IOError

    def save(self, data, filename, delimiter='\t', mode='txt', header=''):
        if self.verbose:
            s = """Saving
            ------
            filename: %s
            mode: %s
            delimiter: %s
            Object-type: %s
            """ % (filename, mode, delimiter, type(data))
            print(s)

        if isinstance(data, mfm.curve.Curve):
            d = np.array(data[:])
        elif isinstance(data, np.ndarray):
            d = data
        else:
            d = np.array(data)

        if mode == 'txt':
            np.savetxt(filename, d.T, delimiter=delimiter, header=header)
        if mode == 'npy':
            np.save(filename, d.T)

    @property
    def n_cols(self):
        """
        The number of columns
        """
        return self._data.shape[1]

    @property
    def n_rows(self):
        """
        The number of rows
        """
        return self._data.shape[0]

    @property
    def data(self):
        """
        Numpy array of the data
        """
        if self.reverse:
            return np.array(self._data, dtype=np.float64).T[::-1]
        else:
            return np.array(self._data, dtype=np.float64).T

    #@data.setter
    #def data(self, df):
    #    self._data = df

    @property
    def header(self):
        """
        A list of the column headers
        """
        if self.use_header is not None:
            header = list(self._data.columns)
        else:
            header = range(self._data.shape[1])
        return [str(i) for i in header]

    # def reload_csv(self):
    #     """
    #     Reloads the csv as specified by :py:attribute:`.CSV.filename`
    #     """
    #     self.load(self.filename)
    #
    # This class should only read CSV files. Hence, the code below
    # should not be part of the class.
    # @property
    # def data_x(self):
    #     """
    #     The x-values of the loaded file as numpy.array
    #     """
    #     if self.x_on:
    #         try:
    #             return self.data[self.col_x]
    #         except IndexError:
    #             return np.arange(self.data_y.shape[0], dtype=np.float64)
    #     else:
    #         return np.arange(self.data_y.shape[0], dtype=np.float64)
    #
    # @property
    # def error_x(self):
    #     """
    #     The errors of the x-values of the loaded file as numpy.array
    #     """
    #     if self.error_x_on:
    #         return self.data[self.col_ex]
    #     else:
    #         return self._ex
    #
    # @error_x.setter
    # def error_x(self, prop):
    #     self.error_x_on = False
    #     self._ex = prop
    #
    # @property
    # def error_y(self):
    #     """
    #     The errors of the y-values of the loaded file as numpy.array
    #     """
    #     if self.error_y_on:
    #         return self.data[self.col_y]
    #     else:
    #         if self._ey is None:
    #             return np.ones_like(self.data_y)
    #         else:
    #             return self._ey
    #
    # @error_y.setter
    # def error_y(self, prop):
    #     self.error_y_on = False
    #     self._ey = prop
    #
    # @property
    # def data_y(self):
    #     """
    #     The y-values of the loaded file as numpy.array
    #     """
    #     if self.data is not None:
    #         return self.data[self.col_y]
    #     else:
    #         return None

    @property
    def n_points(self):
        """
        The number of data points corresponds to the number of rows :py:attribute`.CSV.n_rows`
        """
        return self.n_rows


