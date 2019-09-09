"""
The structure module contains most functions and classes to handle structural models of proteins. It contains basically
two submodules:

#. :py:mod:`.mfm.structure.structure`
#. :py:mod:`.mfm.structure.trajectory`
#. :py:mod:`.mfm.potential`

The module :py:mod:`.mfm.potential` provides a set of potentials. The module :py:mod:`.mfm.structure.structure`
provides a set of functions and classes to work with structures and trajectories.

"""
from __future__ import annotations

import mfm.structure.structure
import mfm.structure.trajectory
import mfm.structure.protein
import mfm.structure.potential

