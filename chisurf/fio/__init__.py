"""
The :py:mod:`chisurf.fio` module contains all classes, functions and modules relevant for file input and outputs.
In particular three kinds of file-types are handled:

1. Comma-separated files :py:mod:`chisurf.fio.ascii`
2. PDB-file :py:mod:`chisurf.fio.pdb`
3. TTTR-files containing photon data :py:mod:`chisurf.fio.photons`
4. XYZ-files containing coordinates :py:mod:`chisurf.fio.xyz`
5. DX-files containing densities :py:mod:`chisurf.fio.dx`
6. SDT-files containing time-resolved fluorescence decays :py:mod:`chisurf.fio.sdtfile`

"""

import chisurf.fio.zipped
import chisurf.fio.ascii
import chisurf.fio.coordinates
import chisurf.fio.tttr
import chisurf.fio.photons
import chisurf.fio.sdtfile
import chisurf.fio.fluorescence
