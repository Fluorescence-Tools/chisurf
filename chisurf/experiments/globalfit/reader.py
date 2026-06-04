from __future__ import annotations
from chisurf import typing

import chisurf.data
from chisurf.experiments.core.reader import ExperimentReader


class GlobalFitSetup(ExperimentReader):

    def __init__(self, *args, **kwargs):
        """Initialize a global-fit setup reader."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def autofitrange(*args, **kwargs) -> typing.Tuple[int, int]:
        """Return a trivial fit range (not applicable for global-fit setups).

        Returns
        -------
        tuple of int
            Always ``(0, 0)``.
        """
        return 0, 0

    def read(
            self,
            name: str = "Global-fit",
            *args,
            **kwargs
    ):
        """Create a placeholder data curve for global fitting.

        Parameters
        ----------
        name : str
            Name for the placeholder data curve.

        Returns
        -------
        chisurf.data.DataCurve
            A singleton data curve with zero-valued x/y.
        """
        return chisurf.data.DataCurve(
            x=[0], y=[0],
            setup=self,
            name=name
        )

    def __str__(self):
        """Human-readable string representation."""
        s = 'Global-Fit\n'
        s += 'Name: \t%s \n' % self.name
        return s


