from __future__ import annotations
from collections import Iterable
from typing import List

import mfm.base
import mfm.io
import mfm.parameter
import mfm.curve
import mfm.experiments
import mfm.settings

import mfm.models
import mfm.fitting
#import mfm.structure
#import mfm.fluorescence

#from mfm.settings import cs_settings, colors


#######################################################
#        LIST OF FITS, DATA, EXPERIMENTS              #
#######################################################
fits = []
fit_windows = []
experiment = []
imported_datasets = []
run = None   # This is replaced during initialization to execute commands via a command line interface
cs = None    # The current instance of ChiSurf
console = None


#######################################################
#        SETTINGS  & CONSTANTS                        #
#######################################################

verbose = mfm.settings.cs_settings['verbose']
__version__ = mfm.settings.cs_settings['version']
__name__ = mfm.settings.cs_settings['name']
working_path = ''
eps = 1e-8
cs = None


def find_objects(
        search_list: List,
        object_type,
        remove_double: bool = True):
    """Traverse a list recursively a an return all objects of type `object_type` as
    a list

    :param search_list: list
    :param object_type: an object type
    :param remove_double: boolean
    :return: list of objects with certain object type
    """
    re = []
    for value in search_list:
        if isinstance(value, object_type):
            re.append(value)
        elif isinstance(value, list):
            re += find_objects(value, object_type)
    if remove_double:
        return list(set(re))
    else:
        return re


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
    >>> mfm.c(self.checkBox.stateChanged, models", self.checkBox.isChecked)
    >>> cs.current_fit.models.update_rmsd=True

    :param t: The signal of the qt-widget
    :param st: The string passed to the mfm-commandline
    :param parameters: the parameters passed to the commandline string
    """
    if isinstance(parameters, Iterable):
        t.connect(lambda: run(st % [p if isinstance(p, str) else p() for p in parameters]))
    else:
        t.connect(lambda: run(st % parameters()))

