from __future__ import annotations

import os.path
import pathlib

import chisurf.fio.fluorescence.tcspc
from chisurf import typing

import numpy as np
import tttrlib

import chisurf.settings
import chisurf.fluorescence.tcspc
import chisurf.experiments
import chisurf.base
import chisurf.fluorescence
import chisurf.data
import chisurf.fio.fluorescence

from chisurf.experiments import reader


class TCSPCReader(reader.ExperimentReader):

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
        >>> import pylab as p
        >>> import chisurf.experiments
        >>> filename = "../test/data/tcspc/ibh_sample/Decay_577D.txt"
        >>> ex = chisurf.experiments.experiment.Experiment('TCSPC')
        >>> dt = 0.0141
        >>> g1 = chisurf.experiments.tcspc.TCSPCReader(experiment=ex, skiprows=8, rebin=(1, 8), dt=dt)
        >>> data = g1.read(filename=filename)  # reading_routine will be guessed as 'csv' based on the .txt extension
        >>> x = data.x
        >>> y = data.y
        >>> p.plot(x, y)
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
            '.json': 'json'
        }

        # Return the reading routine for the extension, or the default if not found
        return extension_map.get(ext, self.reading_routine)

    def read(self, filename: str = None, *args, **kwargs) -> chisurf.data.DataCurveGroup:
        import chisurf.fio.fluorescence.thdfile
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


class TCSPCSimulatorSetup(TCSPCReader):

    name = "TCSPC-Simulator"

    def __init__(
            self,
            *args,
            n_tac: int = 4096,
            dt: float = 0.0141,
            p0: float = 10000.0,
            rep_rate: float = 10.0,
            lifetime_spectrum: typing.List[float] = None,
            instrument_response_function: chisurf.data.DataCurve = None,
            sample_name: str = 'TCSPC-Dummy',
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.experiment = kwargs.get('experiment', None)
        if lifetime_spectrum:
            t = ','.join([str(x) for x in lifetime_spectrum])
            self.controller.lineEdit_2.setText(t)
        self.instrument_response_function = instrument_response_function
        self.sample_name = sample_name
        self.lifetime_spectrum = np.array(lifetime_spectrum, dtype=np.float64)
        self.n_tac = n_tac
        self.dt = dt
        self.p0 = p0
        self.rep_rate = rep_rate

    def read(self, filename: str = None, *args, **kwargs) -> chisurf.data.DataCurveGroup:
        if filename is None:
            filename = self.sample_name
        name = kwargs.get('name', filename)
        x = np.arange(self.n_tac) * self.dt
        time_axis, y = chisurf.fluorescence.general.calculate_fluorescence_decay(
            lifetime_spectrum=self.lifetime_spectrum,
            time_axis=x
        )
        data_set = chisurf.data.DataCurve(
            x=time_axis,
            y=y,
            ey=chisurf.fluorescence.tcspc.counting_noise(y),
            setup=self,
            name=name,
            experiment=self.experiment
        )
        return chisurf.data.DataCurveGroup(
            [data_set],
            experiment=self.experiment,
            data_reader=self
        )


class TCSPCTTTRReader(TCSPCReader):

    def read(self, filename: str = None, *args, **kwargs) -> chisurf.data.DataCurveGroup:
        if os.path.isfile(filename):
            tttr = tttrlib.TTTR(filename, self.reading_routine)
            tttr_selected = tttr.get_tttr_by_channel(self.channel_numbers)
            y, x = tttr_selected.get_microtime_histogram(self.micro_time_coarsening)
            x *= 1.0e9  # decays in ns

            y_pos = np.where(y > 0)[0]
            i_y_max = y_pos[-1]
            y = y[:i_y_max]
            t = x[:i_y_max]

            # Prepare name
            fn, _ = os.path.splitext(filename)
            name = fn + "_ch("
            name += ",".join([str(i) for i in self.channel_numbers])
            name += ")"

            data_set = chisurf.data.DataCurve(
                x=t, y=y, name=name,
                experiment=self.experiment,
                data_reader=self,
                ey=chisurf.fluorescence.tcspc.counting_noise(y)
            )
            data_group = chisurf.data.DataGroup([data_set])
            return data_group
        else:
            raise FileNotFoundError

    def __int__(
            self,
            tttr_filename: pathlib.Path,
            channel_numbers=None,
            reading_routine: str = 'PTU',
            micro_time_coarsening: int = 1,
            *args,
            **kwargs
    ):
        super().__int__(*args, **kwargs)
        if channel_numbers is None:
            channel_numbers = []
        self.reading_routine = reading_routine
        self.channel_numbers = channel_numbers
        self.micro_time_coarsening = micro_time_coarsening
