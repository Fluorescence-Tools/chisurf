from typing import List, Iterable
import chisurf.settings as settings

__version__ = settings.cs_settings['version']
__name__ = settings.cs_settings['name']


def c(
        t,
        st: str,
        parameters: List
):
    """This function facilitates the connection of qt-events to the mfm-command
    line. Whenever the qt-event is called the string passed as argument is
    executed at the mfm-commandline with the provided parameters.
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
experiment = list()
fit_windows = list()
working_path = ''
