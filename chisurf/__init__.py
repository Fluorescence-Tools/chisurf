"""ChiSurf is sth.

"""
from __future__ import annotations

import sys
import logging
import pathlib

try:
    if sys.version_info >= (3, 8):
        import typing
    elif sys.version_info >= (3, 7):
        # monkey patch the 3.7 typing system as
        # TypedDict etc. is missing
        import typing_extensions
        import typing
        for key in typing_extensions.__dict__.keys():
            f = typing_extensions.__dict__[key]
            if callable(f):
                typing.__dict__[key] = f
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
__url__ = 'https://fluorescence-tools.github.io/chisurf/'
__license__ = 'GPL2.1'
__status__ = "Dev"
__description__ = """ChiSurf: an interactive global analysis platform for fluorescence \
data."""
__app_id__ = "F25DCFFA-1234-4643-BC4F-2C3A20495937"


fits: typing.List[chisurf.fitting.fit.FitGroup] = list()
imported_datasets: typing.List[chisurf.data.DataGroup] = list()
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


