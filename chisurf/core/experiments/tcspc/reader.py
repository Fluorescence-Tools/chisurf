from __future__ import annotations

import os.path
import pathlib

import chisurf.core.fio.fluorescence.tcspc as tcspc_io
from chisurf import typing

import numpy as np
import tttrlib

import chisurf.core.settings
import chisurf.core.fluorescence.tcspc
import chisurf.core.base
import chisurf.core.fluorescence
import chisurf.core.data

from chisurf.core.experiments.core.reader import ExperimentReader


class TCSPCReader(ExperimentReader):
    operation_type = "tcspc_curve_load"
    artifact_kind_source = "raw_data"
    artifact_kind_derived = "tcspc_decay"
    derived_data_format = "json"
    derived_mime_type = "application/json"

    @staticmethod
    def _safe_float(value: typing.Any, default: typing.Any):
        """Convert a value to float, returning *default* on failure.

        Parameters
        ----------
        value : any
            The value to convert.
        default : any
            Fallback value if conversion fails.

        Returns
        -------
        float
            The converted value or default.
        """
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @classmethod
    def _default_anisotropy_calibration(cls):
        """Return the global default anisotropy calibration parameters.

        Reads ``g_factor``, ``l1``, and ``l2`` from the settings.

        Returns
        -------
        dict
            Dictionary with keys ``g_factor``, ``l1``, ``l2``.
        """
        anisotropy_settings = getattr(chisurf.core.settings, 'anisotropy', {})
        tcspc_settings = getattr(chisurf.core.settings, 'tcspc', {})
        g_factor = cls._safe_float(
            anisotropy_settings.get('g_factor', tcspc_settings.get('g_factor', 1.0)),
            1.0
        )
        l1 = cls._safe_float(anisotropy_settings.get('l1', 0.0), 0.0)
        l2 = cls._safe_float(anisotropy_settings.get('l2', 0.0), 0.0)
        return {
            'g_factor': g_factor,
            'l1': l1,
            'l2': l2,
        }

    def _reader_calibration(self):
        """Return the reader-level anisotropy calibration parameters.

        Returns
        -------
        dict
            Dictionary with keys ``g_factor``, ``l1``, ``l2``, and ``source``.
        """
        return {
            'g_factor': self._safe_float(getattr(self, 'g_factor', 1.0), 1.0),
            'l1': self._safe_float(getattr(self, 'l1', 0.0), 0.0),
            'l2': self._safe_float(getattr(self, 'l2', 0.0), 0.0),
            'source': 'tcspc_reader'
        }

    def _annotate_anisotropy_calibration(self, data_group) -> None:
        """Attach anisotropy calibration metadata to a data group.

        Parameters
        ----------
        data_group : chisurf.core.data.DataGroup
            The data group to annotate.
        """
        calibration = self._reader_calibration()

        group_meta = getattr(data_group, 'meta_data', None)
        if not isinstance(group_meta, dict):
            group_meta = {}
            data_group.meta_data = group_meta
        group_meta.setdefault('g_factor', calibration['g_factor'])
        group_meta.setdefault('l1', calibration['l1'])
        group_meta.setdefault('l2', calibration['l2'])
        group_meta.setdefault('anisotropy_calibration_source', calibration['source'])

        try:
            for curve in data_group:
                try:
                    if getattr(curve, 'data_reader', None) is None:
                        curve.data_reader = self
                except Exception:
                    pass
                curve_meta = getattr(curve, 'meta_data', None)
                if not isinstance(curve_meta, dict):
                    curve_meta = {}
                    curve.meta_data = curve_meta
                curve_meta.setdefault('g_factor', group_meta.get('g_factor', calibration['g_factor']))
                curve_meta.setdefault('l1', group_meta.get('l1', calibration['l1']))
                curve_meta.setdefault('l2', group_meta.get('l2', calibration['l2']))
                curve_meta.setdefault(
                    'anisotropy_calibration_source',
                    group_meta.get('anisotropy_calibration_source', calibration['source'])
                )
        except Exception:
            pass

    def __init__(
            self,
            dt: typing.Optional[float] = None,
            rep_rate: typing.Optional[float] = None,
            is_jordi: bool = False,
            mode: str = 'vm',
            g_factor: typing.Optional[float] = None,
            l1: typing.Optional[float] = None,
            l2: typing.Optional[float] = None,
            rebin: typing.Tuple[int, int] = (1, 1),
            matrix_columns: typing.Tuple[int, int] = (0, 1),
            skiprows: int = 8,
            polarization: str = 'vm',
            use_header: bool = True,
            fit_area: typing.Optional[float] = None,
            fit_start_fraction: typing.Optional[float] = None,
            fit_count_threshold: typing.Optional[float] = None,
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
            Polarization mode. Use 'vv/vh' for stacked VV,VH jordi files
        g_factor : float, optional
            G-factor for anisotropy calculations
        l1 : float, optional
            Polarization correction factor l1
        l2 : float, optional
            Polarization correction factor l2
        rebin : tuple of int, optional
            Rebinning factors for x and y axes
        matrix_columns : tuple of int, optional
            Columns to use from the data matrix
        skiprows : int, optional
            Number of rows to skip at the beginning of the file
        polarization : str, optional
            Polarization mode (alias for mode parameter)
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
        >>> import chisurf.core.experiments  # doctest: +SKIP
        >>> filename = "../test/data/tcspc/ibh_sample/Decay_577D.txt"  # doctest: +SKIP
        >>> ex = chisurf.core.experiments.core.Experiment('TCSPC')  # doctest: +SKIP
        >>> dt = 0.0141  # doctest: +SKIP
        >>> g1 = chisurf.core.experiments.tcspc.TCSPCReader(experiment=ex, skiprows=8, rebin=(1, 8), dt=dt)  # doctest: +SKIP
        >>> data = g1.read(filename=filename)  # doctest: +SKIP
        >>> x = data.x  # doctest: +SKIP
        >>> y = data.y  # doctest: +SKIP
        >>> p.plot(x, y)  # doctest: +SKIP
        """
        super().__init__(*args, **kwargs)
        if dt is None:
            dt = chisurf.core.settings.tcspc['dt']
        calibration_defaults = self._default_anisotropy_calibration()
        if g_factor is None:
            g_factor = calibration_defaults['g_factor']
        if l1 is None:
            l1 = calibration_defaults['l1']
        if l2 is None:
            l2 = calibration_defaults['l2']
        if rep_rate is None:
            rep_rate = chisurf.core.settings.tcspc['rep_rate']
        if fit_area is None:
            fit_area = chisurf.core.settings.tcspc['fit_area']
        if fit_count_threshold is None:
            fit_count_threshold = chisurf.core.settings.tcspc['fit_count_threshold']
        if fit_start_fraction is None:
            fit_start_fraction = chisurf.core.settings.tcspc['fit_start_fraction']
        self.dt = dt
        self.excitation_repetition_rate = rep_rate
        self.is_jordi = is_jordi
        # Use mode parameter, but allow polarization as alias for backward compatibility
        self.polarization = mode if mode != 'vm' else polarization
        self.g_factor = self._safe_float(g_factor, calibration_defaults['g_factor'])
        self.l1 = self._safe_float(l1, calibration_defaults['l1'])
        self.l2 = self._safe_float(l2, calibration_defaults['l2'])
        self.rep_rate = rep_rate
        self.rebin = rebin
        self.matrix_columns = matrix_columns
        self.skiprows = skiprows
        self.use_header = use_header
        self.matrix_columns = matrix_columns
        self.fit_area = fit_area
        self.fit_count_threshold = fit_count_threshold
        self.fit_start_fraction = fit_start_fraction
        self.reading_routine = reading_routine
        self.vh_shift = int(vh_shift) if vh_shift is not None else 0

    def autofitrange(self, data, **kwargs) -> typing.Tuple[int, int]:
        """Determine the default fit range for TCSPC data.

        Delegates to :func:`chisurf.core.fluorescence.tcspc.initial_fit_range`.

        Parameters
        ----------
        data : chisurf.core.base.Data
            The experimental TCSPC data.

        Returns
        -------
        tuple of int
            ``(start, stop)`` indices for the fit region.
        """
        return chisurf.core.fluorescence.tcspc.initial_fit_range(
            data.y,
            self.fit_count_threshold,
            self.fit_area,
            start_fraction=self.fit_start_fraction,
            verbose=chisurf.core.settings.cs_settings['verbose']
        )

    def _guess_reading_routine(self, filename: typing.Optional[str]) -> str:
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
        >>> from chisurf.core.experiments.tcspc import TCSPCReader
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

    def read(self, filename: typing.Optional[str] = None, *args, **kwargs) -> typing.Any:
        """Read TCSPC data from a file.

        Supports CSV, THD, PQRES, YAML, and JSON formats. The reading
        routine is either explicitly specified or guessed from the
        file extension via :meth:`_guess_reading_routine`.

        Parameters
        ----------
        filename : str, optional
            Path to the TCSPC data file.

        Returns
        -------
        chisurf.core.data.DataGroup
            Group containing the loaded TCSPC curves.
        """
        import chisurf.core.fio.fluorescence.thdfile
        import chisurf.core.fio.fluorescence.pqres
        import chisurf.core.data

        if filename is None:
            raise ValueError("filename must be provided")

        # Guess the reading routine if not explicitly overridden in kwargs
        reading_routine = kwargs.get('reading_routine', self.reading_routine)

        # If the reading routine is not explicitly set, guess it from the filename
        if reading_routine == 'auto' or (reading_routine == self.reading_routine and self.reading_routine == 'auto'):
            reading_routine = self._guess_reading_routine(filename)

        if reading_routine == 'csv':
            data_group = tcspc_io.read_tcspc_csv(
                filename=filename,
                skiprows=self.skiprows,
                rebin=self.rebin,
                dt=self.dt,
                matrix_columns=self.matrix_columns,
                use_header=self.use_header,
                is_jordi=self.is_jordi,
                polarization=self.polarization,
                g_factor=self.g_factor,
                l1=self.l1,
                l2=self.l2,
                experiment=getattr(self, 'experiment', None),
                data_reader=self
            )
        elif reading_routine == 'thd':
            data_group = chisurf.core.fio.fluorescence.thdfile.read_tcspc_thd(
                filename=filename,
                rebin=self.rebin,
                dt=self.dt,
                experiment=getattr(self, 'experiment', None),
                data_reader=self
            )
        elif reading_routine == 'pqres':
            data_group = chisurf.core.fio.fluorescence.pqres.read_pqres_tcspc(
                filename=filename,
                rebin=self.rebin,
                dt=self.dt,
                experiment=getattr(self, 'experiment', None),
                data_reader=self
            )
        else:
            if reading_routine == 'yaml':
                file_type = 'yaml'
                data_set = chisurf.core.data.DataCurve()
                data_set.load(
                    file_type=file_type,
                    filename=filename
                )
                data_set.experiment = self.experiment
                data_group = chisurf.core.data.DataGroup([data_set])
            else:
                chisurf.logging.warning(
                    "Reading routine '%s' not supported. "
                    "Created empty DataGroup" % reading_routine
                )
                data_group = chisurf.core.data.DataGroup([])
        self._annotate_anisotropy_calibration(data_group)
        data_group.data_reader = self
        return data_group
