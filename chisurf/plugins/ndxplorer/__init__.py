"""
ndXplorer

This plugin provides a powerful interface for analyzing and visualizing multidimensional 
fluorescence data within ChiSurf.

Features:
- Burst analysis for single-molecule fluorescence experiments
- Multiparameter fluorescence detection (MFD) analysis
- Interactive selection and filtering of burst events
- Visualization of multidimensional data through histograms and plots
- Support for FRET efficiency calculations and proximity ratio analysis
- Application to both solution-based measurements and image spectroscopy data

The ndXplorer tool is particularly useful for analyzing complex fluorescence datasets 
where multiple parameters need to be correlated, such as fluorescence intensity, 
lifetime, anisotropy, and spectral information. It provides an intuitive interface 
for exploring relationships between different fluorescence parameters.

For single-molecule experiments, ndXplorer enables detailed burst analysis with 
capabilities to select, filter, and categorize individual molecule detection events 
based on multiple criteria. The tool also supports advanced FRET analysis with 
various correction factors and calculation methods.

When working with image spectroscopy data, ndXplorer allows pixel-by-pixel analysis 
of multiparameter fluorescence information, enabling spatial correlation of 
spectroscopic properties.
"""

name = "Main:Tools:ndXplorer"

import chisurf as cs
log = cs.logging.info


if __name__ == '__main__':
    import sys
    from qtpy.QtWidgets import QApplication
    import ndxplorer
    app = QApplication(sys.argv)
    ndx = ndxplorer.NDXplorer()
    ndx.show()
    ndx.raise_()
    ndx.activateWindow()
    sys.exit(app.exec())

if __name__ == "plugin":
    import sys
    import pathlib
    _ndxplorer_module = pathlib.Path(__file__).resolve().parents[3] / "modules" / "ndxplorer"
    if _ndxplorer_module.is_dir():
        p = str(_ndxplorer_module)
        if p not in sys.path:
            sys.path.insert(0, p)
    import ndxplorer
    try:
        # Inject the in-process ChiSurf client so the phasor / FRET-line toolbar
        # is available; falls back to a plain window if the RPC stack is missing.
        from chisurf.plugins.ndxplorer.rpc_bridge import make_ndxplorer

        ndx = make_ndxplorer()
    except Exception:
        log("Could not load ndXplorer plugin (missing optional dependencies)")
        raise
    ndx.show()
    ndx.raise_()
    ndx.activateWindow()

    # Add MFDB toolbar button if MFDB is connected
    try:
        from mfdb.admin.gui.client import MFDBClient
        from chisurf.plugins.ndxplorer.mfdb_launcher import (
            BURST_FORMATS, BURST_KINDS, resolve_dataset_path,
        )
        from chisurf.gui.widgets.mfdb.dataset_browser import MfdbDatasetPickerDialog
        from ndxplorer.__main__ import open_path_like_drop
        from qtpy import QtCore

        client = MFDBClient(inprocess=True)
        client.status()  # raises if MFDB database is not accessible

        def _open_burst_in_current_ndx() -> None:
            sel = MfdbDatasetPickerDialog.pick_dataset(
                parent=ndx,
                kinds=BURST_KINDS,
                formats=BURST_FORMATS,
                scope="all",
                client=client,
            )
            if sel is None:
                return
            path = resolve_dataset_path(client, sel.artifact_id)
            if not path:
                return
            QtCore.QTimer.singleShot(
                0, lambda: open_path_like_drop(ndx, str(path))
            )

        toolbar = ndx.addToolBar("MFDB")
        toolbar.setObjectName("ndxplorerMfdbToolbar")
        mfdb_action = toolbar.addAction("🗄️ Open from MFDB")
        mfdb_action.setToolTip("Open a burst selection registered in MFDB")
        mfdb_action.triggered.connect(_open_burst_in_current_ndx)
    except Exception:
        pass  # MFDB not available — skip toolbar button


cli_entrypoint = "ndxplorer=chisurf.plugins.ndxplorer.cli:cli"

