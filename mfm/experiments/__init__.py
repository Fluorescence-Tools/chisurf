"""
This module is responsible for all experiments/fits

The :py:mod:`experiments` module contains the fitting model and the setups (assembled reading routines) for
different experimental setups. Furthermore, it contains a set of plotting libraries.


"""
from __future__ import annotations
from typing import List

from . import experiment
from . import data
from . import reader
from . import fcs
from . import tcspc
from . import globalfit
from . import modelling


def get_data(
        curve_type: str = 'experiment',
        data_set: List[data.ExperimentalData] = list()
):
    """
    Returns all curves `mfm.curve.DataCurve` except if the curve is names "Global-fit"
    """
    if curve_type == 'all':
        return [d for d in data_set
                if isinstance(d, data.ExperimentalData) or
                isinstance(d, data.ExperimentDataGroup)
                ]
    elif curve_type == 'experiment':
        return [
            d for d in data_set if (isinstance(d, data.ExperimentalData) or
                                    isinstance(d, data.ExperimentDataGroup))
                                   and d.name != "Global-fit"]


"""
# This needs to move to the QtApplication or it needs to be
# independent as new Widgets can only be created once a QApplication has been created
tcspc_setups = [
    tcspc.TCSPCSetupWidget(name="CSV/PQ/IBH", **mfm.cs_settings['tcspc_csv']),
    tcspc.TCSPCSetupSDTWidget(),
    tcspc.TCSPCSetupDummyWidget()
]

fcs_setups = [
    fcs.FCSKristine(experiment=fcs),
    fcs.FCSCsv(experiment=fcs)
]

structure_setups = [
    modelling.PDBLoad()
        #.FCSKristine(experiment=fcs)
]
"""