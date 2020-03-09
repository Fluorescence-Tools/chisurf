import typing

import chisurf.settings as settings
import logging

from chisurf.version import __version__

def c(
        t,
        st: str,
        parameters: typing.List
):
    """This function facilitates the connection of qt-events to the mfm-command
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


fits = list()
imported_datasets = list()
run = lambda x: x   # This is replaced during initialization to execute commands via a command line interface
cs = object         # The current instance of ChiSurf
console = object
experiment = dict()
fit_windows = list()
working_path = ''
verbose = settings.cs_settings['verbose']

logging.basicConfig(
    filename=settings.session_log,
    level=logging.DEBUG
)
import chisurf.decorators
