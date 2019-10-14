"""
The :py:mod:`mfm.io` module contains all classes, functions and modules relevant for file input and outputs.
In particular three kinds of file-types are handled:

1. Comma-separated files :py:mod:`mfm.io.ascii`
2. PDB-file :py:mod:`mfm.io.pdb`
3. TTTR-files containing photon data :py:mod:`mfm.io.photons`
4. XYZ-files containing coordinates :py:mod:`mfm.io.xyz`
5. DX-files containing densities :py:mod:`mfm.io.dx`
6. SDT-files containing time-resolved fluorescence decays :py:mod:`mfm.io.sdtfile`

"""

import mfm.io.ascii
import mfm.io.coordinates
import mfm.io.tttr
import mfm.io.photons
import mfm.io.sdtfile
import mfm.io.zipped
#import mfm.io.widgets
