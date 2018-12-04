from collections import Iterable

#######################################################
#       OPENCL                                        #
#######################################################
#import pyopencl.array
#cl_platform = pyopencl.get_platforms()[0]
#cl_device = cl_platform.get_devices()[0]

#######################################################
#        LIST OF FITS, DATA, EXPERIMENTS              #
#######################################################
fits = []
fit_windows = []
experiment = []
data_sets = []
run = None   # This is replaced during initialization to execute commands via a command line interface
cs = None    # The current instance of ChiSurf
console = None


#######################################################
#        SETTINGS  & CONSTANTS                        #
#######################################################
from settings import cs_settings, colors

verbose = cs_settings['verbose']
__version__ = cs_settings['version']
__name__ = cs_settings['name']
working_path = ''

import mfm.curve
from mfm.base import *
import mfm.fitting
import mfm.structure
import mfm.fluorescence


def find_fit_idx_of_model(model):
    """Returns index of the fit of a model in mfm.fits array

    :param model:
    :return:
    """
    for idx, f in enumerate(fits):
        if f.model == model:
            return idx


def get_data(curve_type='experiment'):
    """
    Returns all curves `mfm.curve.DataCurve` except if the curve is names "Global-fit"
    """
    if curve_type == 'all':
        return [d for d in data_sets if isinstance(d, curve.ExperimentalData) or isinstance(d, curve.ExperimentDataGroup)]
    elif curve_type == 'experiment':
        return [d for d in data_sets if (isinstance(d, curve.ExperimentalData) or isinstance(d, curve.ExperimentDataGroup)) and d.name != "Global-fit"]


def find_objects(search_list, object_type, remove_double=True):
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


def c(t, st, parameters):
    """This function facilitates the connection of qt-events to the mfm-commandline. Whenever the qt-event
    is called the string passed as argument is executed at the mfm-commandline with the provided parameters.
    Here the parameters are either callable functions of strings.

    Example
    -------
    >>> mfm.c(self.checkBox.stateChanged, "cs.current_fit.model.update_rmsd=%s", self.checkBox.isChecked)
    >>> cs.current_fit.model.update_rmsd=True

    :param t: The signal of the qt-widget
    :param st: The string passed to the mfm-commandline
    :param parameters: the parameters passed to the commandline string
    """
    if isinstance(parameters, Iterable):
        t.connect(lambda: run(st % [p if isinstance(p, str) else p() for p in parameters]))
    else:
        t.connect(lambda: run(st % parameters()))

