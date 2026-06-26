"""
2D-Fluorescence Lifetime Correlation Spectroscopy (2D-FLCS) Plugin

This plugin implements 2D-FLCS analysis for ChiSurf, translating the MATLAB code
from the 2D-FLC-code directory into a Python implementation.

Core functionality:
- 2D-FDC (Fluorescence Decay Correlation) matrix creation
- 2D-MEM (Maximum Entropy Method) fitting
- Multi-exponential decay analysis
- Photon arrival time correlation analysis

Based on MATLAB implementation by T. K. (TK_Create2DFDC_04, TK_FitF_2DMEM_07, etc.)
"""

import logging
from pathlib import Path
from qtpy.QtWidgets import QWidget
from qtpy.QtGui import QIcon

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c


try:
    from chisurf.plugins.fcs.fcs_2d.gui import TwoDFCSWizard
except ImportError:
    # Fallback for direct testing
    try:
        from .gui import TwoDFCSWizard
    except ImportError:
        TwoDFCSWizard = None

name = "Spectroscopy:Fluorescence Correlation Spectroscopy:2D-FLCS"

# Hidden from the menu: surfaced inside the FCS Toolbox meta tool.
menu_hidden = True

# Plugin icon
try:
    _plugin_dir = Path(__file__).parent
    _png = _plugin_dir / "icon.png"
    if _png.exists():
        icon = QIcon(str(_png))
    else:
        icon = QIcon()
except Exception:
    icon = QIcon()

# Main widget class that ChiSurf will instantiate
if TwoDFCSWizard is not None:
    @persist_plugin_state("fcs_2d")
    class TwoDFCSPlugin(TwoDFCSWizard):
        """Main 2D-FLCS plugin widget."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("2D-FLCS Analysis")
            try:
                self.setWindowIcon(icon)
            except Exception:
                pass
            
            logging.getLogger(__name__).info("2D-FLCS Plugin initialized")
else:
    # Fallback widget if imports fail
    class TwoDFCSPlugin(QWidget):
        """Fallback widget when dependencies are missing."""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("2D-FLCS Plugin - Import Error")
            logging.getLogger(__name__).error("Failed to import TwoDFCSWizard")

# Expose the main widget class for ChiSurf
window = TwoDFCSPlugin

# ChiSurf plugin entry point
if __name__ == "plugin":
    try:
        import sys
        print(f"DEBUG: __init__.py loaded with __name__ = 'plugin'")
        print(f"DEBUG: TwoDFCSPlugin available: {TwoDFCSPlugin is not None}")
        
        plugin_window = TwoDFCSPlugin()
        plugin_window.show()
        plugin_window.resize(800, 600)
        window = plugin_window
        print(f"DEBUG: Plugin window created and shown from __init__.py")
        logging.getLogger(__name__).info("2D-FLCS Plugin window created and shown")
    except Exception as e:
        import traceback
        print(f"DEBUG: Exception in __init__.py plugin entry: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        logging.getLogger(__name__).error(f"Failed to create 2D-FLCS Plugin window: {e}")
        # Create fallback error window
        from qtpy.QtWidgets import QMessageBox
        error_widget = QWidget()
        error_widget.setWindowTitle("2D-FLCS Plugin Error")
        QMessageBox.critical(error_widget, "Plugin Error", f"Failed to initialize 2D-FLCS Plugin:\n{str(e)}")
        window = error_widget

elif __name__ == "__main__":
    # Standalone test mode
    import sys
    from qtpy.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    try:
        test_window = TwoDFCSPlugin()
        test_window.show()
        test_window.resize(800, 600)
        print("2D-FLCS Plugin test window opened successfully")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Failed to create test window: {e}")
        sys.exit(1)
