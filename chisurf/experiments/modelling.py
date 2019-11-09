"""

"""
from __future__ import annotations
import typing

import chisurf.base
import chisurf.decorators
import chisurf.fio
import chisurf.structure
from . import reader


class StructureReader(
    reader.ExperimentReader
):

    def __init__(
            self,
            compute_internal_coordinates: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.compute_internal_coordinates = compute_internal_coordinates

    @staticmethod
    def autofitrange(
            data: chisurf.base.Data,
            **kwargs
    ) -> typing.Tuple[int, int]:
        return 0, 0

    def read(
            self,
            filename: str = None,
            *args,
            **kwargs
    ) -> chisurf.experiments.data.ExperimentDataGroup:
        structure = chisurf.structure.Structure(
            p_object=filename,
            make_coarse=self.compute_internal_coordinates
        )
        data_group = chisurf.experiments.data.ExperimentDataGroup(
            seq=[structure]
        )
        data_group.data_reader = self
        return data_group


