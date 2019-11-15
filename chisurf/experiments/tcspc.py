from __future__ import annotations
import typing

import numpy as np

import chisurf.settings
import chisurf.fluorescence.tcspc
import chisurf.experiments
import chisurf.base
import chisurf.fluorescence
import chisurf.experiments.data
import chisurf.fio.fluorescence

from chisurf.experiments import reader


class TCSPCReader(
    reader.ExperimentReader
):

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
            fit_count_threshold: float = None,
            *args,
            **kwargs
    ):
        """

        :param dt:
        :param rep_rate:
        :param is_jordi:
        :param mode:
        :param g_factor:
        :param rebin:
        :param matrix_columns:
        :param skiprows:
        :param polarization:
        :param use_header:
        :param fit_area:
        :param fit_count_threshold:
        :param args:
        :param kwargs:

        Example
        -------

        >>> import chisurf.experiments
        >>> filename = "./test/data/tcspc/ibh_sample/Decay_577D.txt"
        >>> ex = chisurf.experiments.experiment.Experiment('TCSPC')
        >>> dt = 0.0141
        >>> g1 = chisurf.experiments.tcspc.TCSPCReader(experiment=ex, skiprows=8, rebin=(1, 8), dt=dt)

        """
        super().__init__(
            *args,
            **kwargs
        )
        if dt is None:
            dt = chisurf.settings.tcspc['dt']
        if g_factor is None:
            g_factor = chisurf.settings.tcspc['g_factor']
        if rep_rate is None:
            rep_rate = chisurf.settings.tcspc['rep_rate']
        if fit_area is None:
            fit_area = chisurf.settings.tcspc['fit_area']
        if fit_count_threshold is None:
            fit_count_threshold = chisurf.settings.tcspc['fit_count_threshold']

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

    def autofitrange(
            self,
            data,
            **kwargs
    ) -> typing.Tuple[int, int]:
        return chisurf.fluorescence.tcspc.fitrange(
            data.y,
            self.fit_count_threshold,
            self.fit_area
        )

    def read(
            self,
            filename: str = None,
            *args,
            **kwargs
    ) -> chisurf.experiments.data.DataCurveGroup:
        data_group = chisurf.fio.fluorescence.read_tcspc_csv(
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
        data_group.data_reader = self
        return data_group


class TCSPCSetupDummy(
    TCSPCReader
):

    name = "Dummy-TCSPC"

    def __init__(
            self,
            *args,
            n_tac: int = 4096,
            dt: float = 0.0141,
            p0: float = 10000.0,
            rep_rate: float = 10.0,
            lifetime_spectrum: typing.List[float] = None,
            instrument_response_function: chisurf.experiments.data.DataCurve = None,
            sample_name: str = 'TCSPC-Dummy',
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        if lifetime_spectrum is None:
            lifetime_spectrum = [1.0, 4.1]
        self.instrument_response_function = instrument_response_function
        self.sample_name = sample_name
        self.lifetime_spectrum = np.array(
            lifetime_spectrum,
            dtype=np.float
        )
        self.n_tac = n_tac
        self.dt = dt
        self.p0 = p0
        self.rep_rate = rep_rate

    def read(
            self,
            filename: str = None,
            **kwargs
    ) -> chisurf.experiments.data.DataCurveGroup:
        if filename is None:
            filename = self.sample_name

        x = np.arange(self.n_tac) * self.dt
        time_axis, y = chisurf.fluorescence.general.calculate_fluorescence_decay(
            lifetime_spectrum=self.lifetime_spectrum,
            time_axis=x
        )
        d = chisurf.experiments.data.DataCurve(
            x=time_axis,
            y=y,
            ey=1./chisurf.fluorescence.tcspc.weights(y),
            setup=self,
            name=filename
        )
        d.setup = self
        return d