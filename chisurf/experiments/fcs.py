from __future__ import annotations

import chisurf.fio.ascii
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
        self.experiment_reader = experiment_reader.lower()

    def read(
            self,
            filename: str = None,
            verbose: bool = None,
            **kwargs
    ) -> chisurf.experiments.data.ExperimentDataCurveGroup:
        r = chisurf.fio.fluorescence.read_fcs(
            filename=filename,
            data_reader=self,
            skiprows=self.skiprows,
            use_header=self.use_header,
            reader_name=self.experiment_reader,
            experiment=self.experiment
        )
        return r


