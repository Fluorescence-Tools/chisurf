"""

"""
from __future__ import annotations
from chisurf import typing

import chisurf.base
import chisurf.decorators
import chisurf.fio
import chisurf.structure
from chisurf.experiments.core.reader import ExperimentReader


class StructureReader(
    ExperimentReader
):

    def __init__(
            self,
            compute_internal_coordinates: bool = False,
            *args,
            **kwargs
    ):
        """Initialize a structure reader.

        Parameters
        ----------
        compute_internal_coordinates : bool
            Whether to compute internal coordinates (dihedral angles etc.)
            when loading a structure.
        """
        super().__init__(
            *args,
            **kwargs
        )
        self.compute_internal_coordinates = compute_internal_coordinates

    @staticmethod
    def autofitrange(data: chisurf.base.Data, **kwargs) -> typing.Tuple[int, int]:
        """Return a trivial fit range (not applicable for structure data).

        Parameters
        ----------
        data : chisurf.base.Data
            The experimental data (unused).

        Returns
        -------
        tuple of int
            Always ``(0, 0)``.
        """
        return 0, 0

    def read(
            self,
            filename: str = None,
            *args,
            **kwargs
    ) -> chisurf.data.ExperimentDataGroup:
        """Load a molecular structure from a file.

        Parameters
        ----------
        filename : str, optional
            Path to the structure file (e.g. PDB).

        Returns
        -------
        chisurf.data.ExperimentDataGroup
            Group containing the loaded :class:`Structure`.
        """
        structure = chisurf.structure.Structure(
            p_object=filename,
            make_coarse=self.compute_internal_coordinates
        )
        data_group = chisurf.data.ExperimentDataGroup(
            seq=[structure]
        )
        data_group.data_reader = self
        return data_group


