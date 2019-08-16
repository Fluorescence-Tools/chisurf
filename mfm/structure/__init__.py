"""
The structure module contains most functions and classes to handle structural models of proteins. It contains basically
two submodules:

#. :py:mod:`.mfm.structure.structure`
#. :py:mod:`.mfm.structure.trajectory`
#. :py:mod:`.mfm.potential`

The module :py:mod:`.mfm.potential` provides a set of potentials. The module :py:mod:`.mfm.structure.structure`
provides a set of functions and classes to work with structures and trajectories.

"""
import mfm.structure.trajectory
from mfm.structure.protein import *
from mfm.structure.structure import *
