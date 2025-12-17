from __future__ import annotations

import numpy as np

import chisurf.data
import chisurf.fluorescence
import chisurf.fluorescence.tcspc

from chisurf import typing

from .reader import TCSPCReader


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
