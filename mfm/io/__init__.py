"""
The :py:mod:`mfm.io` module contains all classes, functions and modules relevant for file input and outputs.
In particular three kinds of file-types are handled:

1. Comma-separated files :py:mod:`mfm.io.csv_file`
2. PDB-file :py:mod:`mfm.io.pdb_file`
3. TTTR-files containing photon data :py:mod:`mfm.io.photons`


"""

import mfm.io.ascii
import mfm.io.pdb
# import mfm.io.widgets
import mfm.io.photons
import mfm.io.sdtfile
import mfm.io.tttr
import mfm.io.zipped

