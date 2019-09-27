from __future__ import annotations
from typing import List

import os
import numpy as np
from collections.abc import Iterable

#######################################################
#        SETTINGS  & CONSTANTS                        #
#######################################################
import mfm.settings
package_directory = os.path.dirname(os.path.abspath(__file__))
verbose = mfm.settings.cs_settings['verbose']
__version__ = mfm.settings.cs_settings['version']
__name__ = mfm.settings.cs_settings['name']
working_path = ''
eps = np.sqrt(np.finfo(float).eps)

#######################################################
#        LIST OF FITS, DATA, EXPERIMENTS              #
#######################################################
fits = list()
fit_windows = list()
experiment = list()
imported_datasets = list()
run = lambda x: x   # This is replaced during initialization to execute commands via a command line interface
cs = object         # The current instance of ChiSurf
console = object


def c(
        t,
        st: str,
        parameters: List[mfm.parameter.Parameter]
):
    """This function facilitates the connection of qt-events to the mfm-commandline. Whenever the qt-event
    is called the string passed as argument is executed at the mfm-commandline with the provided parameters.
    Here the parameters are either callable functions of strings.

    Example
    -------
    >>> mfm.c(self.checkBox.stateChanged, models, self.checkBox.isChecked)
    >>> cs.current_fit.model.update_rmsd=True

    :param t: The signal of the qt-widget
    :param st: The string passed to the mfm-commandline
    :param parameters: the parameters passed to the commandline string
    """
    if isinstance(parameters, Iterable):
        t.connect(lambda: run(st % [p if isinstance(p, str) else p() for p in parameters]))
    else:
        t.connect(lambda: run(st % parameters()))

