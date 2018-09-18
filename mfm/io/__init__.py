"""
The :py:mod:`mfm.io` module contains all classes, functions and modules relevant for file input and outputs.
In particular three kinds of file-types are handled:

1. Comma-separated files :py:mod:`mfm.io.csv_file`
2. PDB-file :py:mod:`mfm.io.pdb_file`
3. TTTR-files containing photon data :py:mod:`mfm.io.photons`


"""

from . import ascii
from . import sdtfile
from . import pdb
from . import widgets
from . import photons
from . import tttr
from . import zipped

