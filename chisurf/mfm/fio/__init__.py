"""
The :py:mod:`mfm.fio` module contains all classes, functions and modules relevant for file input and outputs.
In particular three kinds of file-types are handled:

1. Comma-separated files :py:mod:`mfm.fio.ascii`
2. PDB-file :py:mod:`mfm.fio.pdb`
3. TTTR-files containing photon data :py:mod:`mfm.fio.photons`
4. XYZ-files containing coordinates :py:mod:`mfm.fio.xyz`
5. DX-files containing densities :py:mod:`mfm.fio.dx`
6. SDT-files containing time-resolved fluorescence decays :py:mod:`mfm.fio.sdtfile`

"""

import mfm.fio.ascii
import mfm.fio.coordinates
import mfm.fio.tttr
import mfm.fio.photons
import mfm.fio.sdtfile
import mfm.fio.zipped
#import mfm.fio.widgets
