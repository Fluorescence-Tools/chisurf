"""This

"""
import numpy as np
import json
import os
from collections import Iterable
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relation, sessionmaker
import weakref
import sip
import yaml

sip.setapi('QDate', 2)
sip.setapi('QDateTime', 2)
sip.setapi('QString', 2)
sip.setapi('QTextStream', 2)
sip.setapi('QTime', 2)
sip.setapi('QUrl', 2)
sip.setapi('QVariant', 2)
#######################################################
#        LIST OF FITS, DATA, EXPERIMENTS              #
#######################################################
fits = []
fit_windows = []
experiment = []
data_sets = []
run = None   # This is replaced in later stage and used to execute commands via a command line interface

#######################################################
#        SETTINGS                                     #
#######################################################
package_directory = os.path.dirname(os.path.abspath(__file__))
settings_file = os.path.join(package_directory, 'settings/chisurf.json')

#settings_file = os.path.join(package_directory, 'settings/chisurf.yaml')
colors = json.load(open(os.path.join(package_directory, 'settings/colors.json')))
settings = json.load(open(settings_file))
style_sheet_file = os.path.join(package_directory, settings['gui']['style_sheet'])
#settings = yaml.safe_load(open(settings_file))
__version__ = settings['version']
__name__ = settings['name']
verbose = settings['verbose']
style_sheet = open(style_sheet_file, 'r').read()

working_path = ''
console = None
eps = np.sqrt(np.finfo(float).eps)
rda_axis = np.linspace(settings['fret']['rda_min'],
                       settings['fret']['rda_max'],
                       settings['fret']['rda_resolution'], dtype=np.float64)

#######################################################
#       OPENCL                                        #
#######################################################
try:
    import pyopencl
    import pyopencl.array
    cl_platform = pyopencl.get_platforms()[0]
    cl_device = cl_platform.get_devices()[0]
except ImportError:
    pyopencl = None
    cl_platform = None
    cl_device = None


#######################################################
#        DATABASE INITIALIZATION                      #
#######################################################
SQLBase = declarative_base()
engine = create_engine(settings['database']['engine'])

#######################################################
#        PLOTTING INITIALIZATION                      #
#######################################################
import pyqtgraph as pg
plot_settings = settings['gui']['plot']
pyqtgraph_settings = plot_settings["pyqtgraph"]
for setting in pyqtgraph_settings:
    pg.setConfigOption(setting, pyqtgraph_settings[setting])


class Base(object):

    @property
    def name(self):
        if self._name is None:
            return 'None'
        return self._name

    @name.setter
    def name(self, v):
        self._name = v

    def save(self, filename):
        txt = self.to_json()
        with open(filename, 'w') as fp:
            fp.write(txt)

    def to_dict(self):
        return {'name': self.name}

    def from_dict(self, v):
        try:
            self._name = v['name']
        except AttributeError:
            print "Values in dictionary missing"

    def to_json(self):
        #return json.dumps(self.to_dict(), indent=4, sort_keys=True)
        return json.dumps(self.to_dict())

    def from_json(self, json_string=None, filename=None):
        """Reads the content of a JSON file into the object.

        :param json_string:
        :param filename:
        :return:

        Example
        -------

        >>> dc = mfm.curve.DataCurve()
        >>> dc.from_json(filename='./sample_data/internal_types/datacurve.json')
        """
        j = dict()
        if filename is not None:
            with open(filename, 'r') as fp:
                j = json.load(fp)
        elif json_string is not None:
            j = json.loads(json_string)
        else:
            pass
        self.from_dict(j)

    def __setattr__(self, k, v):
        super(Base, self).__setattr__(k, v)

    def __getattr__(self, k):
        super(Base, self).__getattr__(k)

    def __str__(self):
        return self.name + ": " + str(type(self)) + "\n"

    def __init__(self, *args, **kwargs):
        object.__init__(self)
        self.verbose = kwargs.get('verbose', verbose)
        try:
            name = args[0]
        except IndexError:
            name = None
        self._name = kwargs.get('name', name)


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

from . import curve
from . import widgets
from . import structure
from . import io
from . import fluorescence
from . import experiments
from . import math
from . import ui
from . import parameter
from . import fitting
from . import plots
from . import tools
from . import common

import zmq
import random
import sys
import time

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:%s" % port)

#######################################################
#        DATABASE FINALIZE                            #
#######################################################
SQLBase.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db_session = Session()


