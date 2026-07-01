"""Qt-free core for the anisotropy wizard.

Groups the plugin's pure logic:

* :mod:`.irf` — IRF background subtraction + intensity normalisation.
* :mod:`.spectra` — lifetime/rotation spectrum persistence (``*.spk.json``).
* :mod:`.fits` — the VV/VH parameter link/constraint plan.
"""

from . import fits, irf, spectra

__all__ = ["fits", "irf", "spectra"]
