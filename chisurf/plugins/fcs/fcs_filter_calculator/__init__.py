"""
Filtered FCS Lifetime Filter Calculator

This plugin provides a small GUI tool to calculate filtered-FCS (fFCS) lifetime
filters from microtime decay patterns, similar to the Calc_fFCS_Filters
functionality in PAM. It operates on precomputed decay histograms for several
species and a total decay histogram and computes:

- Species-specific filters
- Reconstructed total decay
- Weighted residuals

The plugin is intentionally file-based and self-contained so that it can be
used independently of other wizards. Histogram files are expected to contain a
single column of counts (one value per TAC/microtime bin).

Features
--------
- Load total decay and multiple species decay histograms
- Compute lifetime filters using weighted least-squares (PAM method)
- Visualize filters, reconstruction, and residuals
- Export results to fcs_filter.json for use in correlation analysis

Workflow
--------
1. Load total decay histogram (Browse button)
2. Load one or more species decay histograms (Browse button)
3. Click "Compute filters" to calculate filters
4. Review plots of filters and reconstruction quality
5. Click "Export to JSON" to save filters as fcs_filter.json
"""

import sys

# Export public API
from .api import (
    compute_filters,
    compute_filters_from_files,
    compute_filters_mfd,
    compute_filters_mfd_from_files,
    load_histogram,
    FilterResult,
    FilterResultMFD,
    DetectionMode,
)

from pathlib import Path as _Path

from chisurf.core.plugin import load_manifest as _load_manifest

from .gui.client import FilterCalcClient  # noqa: F401
from .gui_parts.main_window import FcsFilterCalculatorWidget  # noqa: F401

__all__ = [
    "compute_filters",
    "compute_filters_from_files",
    "compute_filters_mfd",
    "compute_filters_mfd_from_files",
    "load_histogram",
    "FilterResult",
    "FilterResultMFD",
    "DetectionMode",
    "FcsFilterCalculatorWidget",
    "FilterCalcClient",
    "menu_hidden",
    "name",
]


_manifest = _load_manifest(_Path(__file__).with_name("manifest.json"))
name = (
    _manifest.display_name if _manifest is not None
    else "Spectroscopy:Fluorescence Correlation Spectroscopy:FCS Filter Calculator"
)

# Hidden from the menu: surfaced inside the FCS Toolbox meta tool.
menu_hidden = True


# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed by the ChiSurf plugin system.
if __name__ == "plugin":
    from .gui import FcsFilterCalculatorWidget
    window = FcsFilterCalculatorWidget()
    window.show()


if __name__ == "__main__":
    import sys
    from qtpy import QtWidgets
    from .gui import FcsFilterCalculatorWidget
    
    app = QtWidgets.QApplication(sys.argv)
    win = FcsFilterCalculatorWidget()
    win.show()
    sys.exit(app.exec_())
