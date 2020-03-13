from __future__ import annotations

import sys
import logging
import pathlib

try:
    if sys.version_info >= (3, 8):
        import typing
    else:
        import typing_extensions as typing
except ModuleNotFoundError:
    print("WARNING typing_extensions not found")
    typing = None

import chisurf.settings

__name__ = 'chisurf'
__author__ = "Thomas-Otavio Peulen"
__version__ = '20.3.9'
__copyright__ = "Copyright (C) 2020 Thomas-Otavio Peulen"
__credits__ = ["Thomas-Otavio Peulen"]
__maintainer__ = "Thomas-Otavio Peulen"
__email__ = "thomas.otavio.peulen@gmail.com"
__license__ = 'GPL2.1'
__status__ = "Dev"
__description__ = "ChiSurf is an interactive global analysis platform for " \
                  "time-resolved fluorescence data."


fits = list()
imported_datasets = list()
run = lambda x: x   # This is replaced during initialization to execute commands via a command line interface
cs = object         # The current instance of ChiSurf
console = object
experiment = dict()
fit_windows = list()
working_path = pathlib.Path().home()
verbose = chisurf.settings.verbose


logging.basicConfig(
    filename=settings.session_log,
    level=logging.DEBUG
)


def c(
        t,
        st: str,
        parameters: typing.List
):
    """This function facilitates the connection of events to the mfm-command
    line. Whenever the qt-event is called the string passed as argument is
    executed at the mfm-commandline with the provided parameters.
    Here the parameters are either callable functions of strings.

    Example
    -------
    >>> import chisurf
    >>> chisurf.c(self.checkBox.stateChanged, models, self.checkBox.isChecked)
    >>> cs.current_fit.model.update_rmsd=True

    :param t: The signal of the qt-widget
    :param st: The string passed to the mfm-commandline
    :param parameters: the parameters passed to the commandline string
    """
    if isinstance(parameters, typing.Iterable):
        t.connect(
            lambda: run(
                st % [p if isinstance(p, str) else p() for p in parameters]
            )
        )
    else:
        t.connect(lambda: run(st % parameters()))


