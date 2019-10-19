from __future__ import annotations
from typing import Tuple

import chisurf.mfm.settings
import chisurf.mfm as mfm
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
            rebin: Tuple[int, int] = (1, 1),
            matrix_columns: Tuple[int, int] = (0, 1),
            skiprows: int = 7,
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
            dt = mfm.settings.tcspc['dt']
        if g_factor is None:
            g_factor = mfm.settings.tcspc['g_factor']
        if rep_rate is None:
            rep_rate = mfm.settings.tcspc['rep_rate']
        if fit_area is None:
            fit_area = mfm.settings.tcspc['fit_area']
        if fit_count_threshold is None:
            fit_count_threshold = mfm.settings.tcspc['fit_count_threshold']

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
    ) -> Tuple[int, int]:
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
    ) -> experiments.data.DataCurveGroup:
        return chisurf.fio.fluorescence.read_tcspc_csv(
            filename=filename,
            skiprows=self.skiprows,
            rebin=self.rebin,
            dt=self.dt,
            matrix_columns=self.matrix_columns,
            use_header=self.use_header,
            is_jordi=self.is_jordi,
            polarization=self.polarization,
            g_factor=self.g_factor,
            setup=self,
        )


