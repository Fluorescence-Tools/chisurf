"""
The structure module contains most functions and classes to handle structural models of proteins. It contains basically
two submodules:

#. :py:mod:`.mfm.structure.structure`
#. :py:mod:`.mfm.structure.trajectory`
#. :py:mod:`.mfm.potential`

The module :py:mod:`.mfm.potential` provides a set of potentials. The module :py:mod:`.mfm.structure.structure`
provides a set of functions and classes to work with structures and trajectories.

"""
from chisurf.core.structure.structure import *
from chisurf.core.structure.trajectory import *
from chisurf.core.structure.protein import *

from . import labeled_structure


def __getattr__(name: str):
    """Lazy-load heavy submodules on first access."""
    if name == "av":
        import chisurf.core.structure.av
        return chisurf.core.structure.av
    if name == "potential":
        import chisurf.core.structure.potential
        return chisurf.core.structure.potential
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

