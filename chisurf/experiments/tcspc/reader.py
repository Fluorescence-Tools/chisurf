from __future__ import annotations

import os.path
import pathlib

import chisurf.fio.fluorescence.tcspc
from chisurf import typing

import numpy as np
import tttrlib

import chisurf.settings
import chisurf.fluorescence.tcspc
import chisurf.base
import chisurf.fluorescence
import chisurf.data
import chisurf.fio.fluorescence

from chisurf.experiments.core.reader import ExperimentReader


class TCSPCReader(ExperimentReader):

    def __init__(
            self,
            dt: float = None,
            rep_rate: float = None,
            is_jordi: bool = False,
            mode: str = 'vm',
            g_factor: float = None,
            rebin: typing.Tuple[int, int] = (1, 1),
            matrix_columns: typing.Tuple[int, int] = (0, 1),
            skiprows: int = 8,
            polarization: str = 'vm',
            use_header: bool = True,
            fit_area: float = None,
            fit_start_fraction: float = None,
            fit_count_threshold: float = None,
            reading_routine: str = 'auto',
            vh_shift: int = 0,
            *args,
            **kwargs
    ):
        """Initialize a TCSPCReader instance.

        Parameters
        ----------
        dt : float, optional
            Time resolution in nanoseconds
        rep_rate : float, optional
            Repetition rate in MHz
        is_jordi : bool, optional
            Whether the file is in Jordi format
        mode : str, optional
            Polarization mode
        g_factor : float, optional
            G-factor for anisotropy calculations
        rebin : tuple of int, optional
            Rebinning factors for x and y axes
        matrix_columns : tuple of int, optional
            Columns to use from the data matrix
        skiprows : int, optional
            Number of rows to skip at the beginning of the file
        polarization : str, optional
            Polarization mode
        use_header : bool, optional
            Whether to use the header in the file
        fit_area : float, optional
            Area to fit
        fit_start_fraction : float, optional
            Fraction of the data to use for fitting
        fit_count_threshold : float, optional
            Threshold for counts in fitting
        reading_routine : str, optional
            Reading routine to use. If set to 'auto' (default), the routine will be
            guessed based on the file extension:
            - '.thd' -> 'thd' (PicoQuant THD files)
            - '.txt', '.dat', '.csv' -> 'csv' (CSV files)
            - '.yaml', '.yml' -> 'yaml' (YAML files)
            - '.json' -> 'json' (JSON files)

        Example
        -------
        The following example performs real file I/O and is therefore
        marked as skipped for doctest:

        >>> import pylab as p  # doctest: +SKIP
        >>> import chisurf.experiments  # doctest: +SKIP
        >>> filename = "../test/data/tcspc/ibh_sample/Decay_577D.txt"  # doctest: +SKIP
        >>> ex = chisurf.experiments.core.Experiment('TCSPC')  # doctest: +SKIP
        >>> dt = 0.0141  # doctest: +SKIP
        >>> g1 = chisurf.experiments.tcspc.TCSPCReader(experiment=ex, skiprows=8, rebin=(1, 8), dt=dt)  # doctest: +SKIP
        >>> data = g1.read(filename=filename)  # doctest: +SKIP
        >>> x = data.x  # doctest: +SKIP
        >>> y = data.y  # doctest: +SKIP
        >>> p.plot(x, y)  # doctest: +SKIP
        """
        super().__init__(*args, **kwargs)
        if dt is None:
            dt = chisurf.settings.tcspc['dt']
        if g_factor is None:
            g_factor = chisurf.settings.anisotropy['g_factor']
        if rep_rate is None:
            rep_rate = chisurf.settings.tcspc['rep_rate']
        if fit_area is None:
            fit_area = chisurf.settings.tcspc['fit_area']
        if fit_count_threshold is None:
            fit_count_threshold = chisurf.settings.tcspc['fit_count_threshold']
        if fit_start_fraction is None:
            fit_start_fraction = chisurf.settings.tcspc['fit_start_fraction']
        self.dt = dt
        self.excitation_repetition_rate = rep_rate
        self.is_jordi = is_jordi
        self.polarization = mode
        self.g_factor = g_factor
        self.rep_rate = rep_rate
        self.rebin = rebin
        self.matrix_columns = matrix_columns
        self.skiprows = skiprows
        self.polarization = polarization
        self.use_header = use_header
        self.matrix_columns = matrix_columns
        self.fit_area = fit_area
        self.fit_count_threshold = fit_count_threshold
        self.fit_start_fraction = fit_start_fraction
        self.reading_routine = reading_routine
        self.vh_shift = int(vh_shift) if vh_shift is not None else 0

    def autofitrange(self, data, **kwargs) -> typing.Tuple[int, int]:
        return chisurf.fluorescence.tcspc.initial_fit_range(
            data.y,
            self.fit_count_threshold,
            self.fit_area,
            start_fraction=self.fit_start_fraction,
            verbose=chisurf.settings.cs_settings['verbose']
        )

    def _guess_reading_routine(self, filename: str) -> str:
        """Guess the reading routine based on the filename extension.

        This method extracts the file extension from the filename and maps it to
        the appropriate reading routine. If the extension is not recognized, it
        returns the default reading routine specified in the class instance.

        The current mapping is:
        - '.thd' -> 'thd' (PicoQuant THD files)
        - '.txt', '.dat', '.csv' -> 'csv' (CSV files)
        - '.yaml', '.yml' -> 'yaml' (YAML files)
        - '.json' -> 'json' (JSON files)

        Parameters
        ----------
        filename : str
            The filename to guess the reading routine for

        Returns
        -------
        str
            The guessed reading routine, or the default reading routine if the
            extension is not recognized or the filename is None
        
        Examples
        --------
        >>> from chisurf.experiments.tcspc import TCSPCReader
        >>> r = TCSPCReader(reading_routine='auto')
        >>> r._guess_reading_routine('decay_data.txt')
        'csv'
        """
        if filename is None:
            return self.reading_routine

        # Get the file extension
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        # Map file extensions to reading routines
        extension_map = {
            '.thd': 'thd',
            '.txt': 'csv',
            '.dat': 'csv',
            '.csv': 'csv',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.pqres': 'pqres'
        }

        # Return the reading routine for the extension, or the default if not found
        return extension_map.get(ext, self.reading_routine)

    def read(self, filename: str = None, *args, **kwargs) -> chisurf.data.DataCurveGroup:
        import chisurf.fio.fluorescence.thdfile
        import chisurf.fio.fluorescence.pqres
        import chisurf.data

        # Guess the reading routine if not explicitly overridden in kwargs
        reading_routine = kwargs.get('reading_routine', self.reading_routine)

        # If the reading routine is not explicitly set, guess it from the filename
        if reading_routine == 'auto' or (reading_routine == self.reading_routine and self.reading_routine == 'auto'):
            reading_routine = self._guess_reading_routine(filename)

        if reading_routine == 'csv':
            data_group: chisurf.data.DataCurveGroup = chisurf.fio.fluorescence.tcspc.read_tcspc_csv(
                filename=filename,
                skiprows=self.skiprows,
                rebin=self.rebin,
                dt=self.dt,
                matrix_columns=self.matrix_columns,
                use_header=self.use_header,
                is_jordi=self.is_jordi,
                polarization=self.polarization,
                g_factor=self.g_factor,
                experiment=self.experiment,
                data_reader=self
            )
        elif reading_routine == 'thd':
            data_group: chisurf.data.DataCurveGroup = chisurf.fio.fluorescence.thdfile.read_tcspc_thd(
                filename=filename,
                rebin=self.rebin,
                dt=self.dt,
                experiment=self.experiment,
                data_reader=self
            )
        elif reading_routine == 'pqres':
            data_group: chisurf.data.DataCurveGroup = chisurf.fio.fluorescence.pqres.read_pqres_tcspc(
                filename=filename,
                rebin=self.rebin,
                dt=self.dt,
                experiment=self.experiment,
                data_reader=self
            )
        else:
            if reading_routine == 'yaml':
                file_type = 'yaml'
                data_set = chisurf.data.DataCurve()
                data_set.load(
                    file_type=file_type,
                    filename=filename
                )
                data_set.experiment = self.experiment
                data_group = chisurf.data.DataGroup([data_set])
            else:
                chisurf.logging.warning(
                    "Reading routine '%s' not supported. "
                    "Created empty DataGroup" % reading_routine
                )
                data_group = chisurf.data.DataGroup([])
        data_group.data_reader = self
        return data_group
