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
    print(
        "WARNING typing_extensions not found",
        file=sys.stderr
    )
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


