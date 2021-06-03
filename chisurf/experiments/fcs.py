from __future__ import annotations

import chisurf.fio.ascii
import chisurf.fio.fluorescence.fcs
import chisurf.fluorescence.fcs
import chisurf.data
import chisurf.fio.fluorescence
from . import reader


class FCS(
    reader.ExperimentReader
):

    name: str = "FCS"
    skiprows: int = 0
    use_header: bool = False

    def __init__(
            self,
            name: str = 'FCS',
            use_header: bool = False,
            experiment_reader='kristine',
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
    ) -> chisurf.data.ExperimentDataCurveGroup:
        r = chisurf.fio.fluorescence.fcs.read_fcs(
            filename=filename,
            data_reader=self,
            skiprows=self.skiprows,
            use_header=self.use_header,
            reader_name=self.experiment_reader,
            experiment=self.experiment
        )
        return r


