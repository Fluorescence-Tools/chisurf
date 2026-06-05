from __future__ import annotations

import numpy as np

import chisurf.core.data
import chisurf.core.fluorescence
import chisurf.core.fluorescence.tcspc

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
            instrument_response_function: chisurf.core.data.DataCurve = None,
            sample_name: str = 'TCSPC-Dummy',
            **kwargs
    ):
        """Initialize a TCSPC simulator.

        Parameters
        ----------
        n_tac : int
            Number of TAC bins (time channels).
        dt : float
            Time resolution per bin in nanoseconds.
        p0 : float
            Initial peak photon count.
        rep_rate : float
            Laser repetition rate in MHz.
        lifetime_spectrum : list of float, optional
            Lifetime components for the simulated decay.
        instrument_response_function : DataCurve, optional
            IRF to convolve with the decay.
        sample_name : str
            Name for the simulated dataset.
        """
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

    def read(self, filename: str = None, *args, **kwargs) -> chisurf.core.data.DataCurveGroup:
        """Generate a simulated TCSPC decay curve.

        Parameters
        ----------
        filename : str, optional
            Ignored; the simulated curve uses ``self.sample_name``.

        Returns
        -------
        chisurf.core.data.DataCurveGroup
            Group containing the simulated decay.
        """
        if filename is None:
            filename = self.sample_name
        name = kwargs.get('name', filename)
        x = np.arange(self.n_tac) * self.dt
        time_axis, y = chisurf.core.fluorescence.general.calculate_fluorescence_decay(
            lifetime_spectrum=self.lifetime_spectrum,
            time_axis=x
        )
        data_set = chisurf.core.data.DataCurve(
            x=time_axis,
            y=y,
            ey=chisurf.core.fluorescence.tcspc.counting_noise(y),
            setup=self,
            name=name,
            experiment=self.experiment
        )
        return chisurf.core.data.DataCurveGroup(
            [data_set],
            experiment=self.experiment,
            data_reader=self
        )
