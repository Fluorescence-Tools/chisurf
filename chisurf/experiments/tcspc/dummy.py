from __future__ import annotations
from typing import List

import numpy as np

import chisurf.fluorescence
from chisurf.experiments.tcspc import TCSPCReader


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
            lifetime_spectrum: List[float] = None,
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


