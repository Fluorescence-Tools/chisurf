from __future__ import annotations

import pathlib
import numpy as np

from chisurf import typing

import tttrlib

import chisurf.settings
import chisurf.fluorescence.tcspc
import chisurf.experiments
import chisurf.base
import chisurf.fluorescence
import chisurf.data
import chisurf.curve
import chisurf.fio.fluorescence

from chisurf.experiments import reader


class PdaReader(reader.ExperimentReader):

    def __init__(
            self,
            channels: typing.Tuple[typing.List[int], typing.List[int]],
            micro_time_ranges: typing.List[typing.Tuple[int, int]],
            reading_routine: str = 'PTU',
            maximum_number_of_photons: int = 148,
            minimum_number_of_photons: int = 20,
            minimum_time_window_length: float = 1e-4,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.reading_routine = reading_routine
        self.micro_time_ranges = micro_time_ranges
        self.maximum_number_of_photons = maximum_number_of_photons
        self.minimum_number_of_photons = minimum_number_of_photons
        self.minimum_time_window_length = minimum_time_window_length
        self.channels = channels

    def autofitrange(self, data, **kwargs) -> typing.Tuple[int, int]:
        chisurf.logging.warning("PDA autofitrange not yet implemented")
        return 0, len(data.y.flatten())

    def read(self, filename: typing.List[str] = None, *args, **kwargs) -> chisurf.data.ExperimentDataGroup:
        if isinstance(filename, str):
            filename = [filename]

        filename.sort()
        fn = pathlib.Path(filename[0])
        data_group = chisurf.data.ExperimentDataGroup([])

        t = None
        if fn.is_file():
            t = tttrlib.TTTR(fn.as_posix(), self.reading_routine)
            for fn in filename[1:]:
                fn = pathlib.Path(fn)
                if fn.is_file():
                    d = tttrlib.TTTR(fn.as_posix(), self.reading_routine)
                    t.append(d)

        if t is not None:
            channels_1 = self.channels[0]
            channels_2 = self.channels[1]
            s1s2_e, ps, tttr_indices = tttrlib.Pda.compute_experimental_histograms(
                tttr_data=t,
                channels_1=channels_1,
                channels_2=channels_2,
                maximum_number_of_photons=self.maximum_number_of_photons,
                minimum_number_of_photons=self.minimum_number_of_photons,
                minimum_time_window_length=self.minimum_time_window_length
            )

            row_indices, col_indices = list(), list()
            for r in range(self.maximum_number_of_photons):
                for c in range(self.maximum_number_of_photons - r):
                    row_indices.append(r)
                    col_indices.append(c)

            d = {
                'maximum_number_of_photons': self.maximum_number_of_photons,
                'minimum_number_of_photons': self.minimum_number_of_photons,
                'minimum_time_window_length': self.minimum_time_window_length,
                'channels': self.channels,
                's1s2': s1s2_e,
                'ps': ps,
                'row_indices': row_indices,
                'col_indices': col_indices
            }
            # Use s1s2 matrix as data only use upper left triangle of matrix
            y = s1s2_e[row_indices, col_indices]
            x = np.arange(len(y))
            data = chisurf.data.DataCurve(
                name=fn.stem,
                data_reader=self,
                pda=d,
                y=y, x=x,
                ey=chisurf.fluorescence.tcspc.counting_noise(y)
            )
            data_group.append(data)

        data_group.data_reader = self
        return data_group
