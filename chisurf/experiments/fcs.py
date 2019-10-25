from __future__ import annotations

import chisurf.fio.ascii
import chisurf.widgets
import chisurf.fluorescence.fcs
import chisurf.experiments.data
import chisurf.fio.fluorescence
from . import reader


class FCS(
    reader.ExperimentReader
):

    def __init__(
            self,
            name: str = 'FCS',
            use_header: bool = False,
            experiment_reader='Kristine',
            skiprows: int = 0,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.name = name
        self.skiprows = skiprows
        self.use_header = use_header
        self.experiment_reader = experiment_reader

    def read(
            self,
            filename: str = None,
            verbose: bool = None,
            **kwargs
    ) -> chisurf.experiments.data.ExperimentDataCurveGroup:

        if self.experiment_reader == 'Kristine':
            r = chisurf.fio.fluorescence.read_fcs_kristine(
                filename=filename,
                verbose=verbose
            )
        elif self.experiment_reader == 'China-mat':
            r = chisurf.fio.fluorescence.read_fcs_china_mat(
                filename=filename,
                verbose=verbose
            )
        elif self.experiment_reader == 'ALV':
            r = chisurf.fio.fluorescence.read_fcs_alv(
                filename=filename,
                verbose=verbose
            )
        else:
            r = chisurf.fio.fluorescence.read_fcs(
                filename=filename,
                setup=self,
                skiprows=self.skiprows,
                use_header=self.use_header
            )
        r.experiment = self
        return r


