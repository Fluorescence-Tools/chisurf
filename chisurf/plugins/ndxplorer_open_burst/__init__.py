"""Open Burst Selection in ndXplorer (from MFDB).

A thin launcher plugin (PRD-28): pick a registered burst selection via the MFDB
sample/measurement selection widget and open it in ndXplorer. All logic lives in
``chisurf.plugins.ndxplorer.mfdb_launcher``; this is just the menu entry.
"""

name = "Tools:Open Burst in ndXplorer"

import chisurf as cs

log = cs.logging.info


if __name__ == "plugin":
    try:
        from chisurf.plugins.ndxplorer.mfdb_launcher import open_burst_selection_from_mfdb

        open_burst_selection_from_mfdb()
    except Exception:
        log("Could not open burst selection in ndXplorer")
        raise
