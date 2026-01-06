import chisurf.gui.widgets.general
from chisurf import logging


class MenuSwitchWidget:
    """
    A simple widget that provides a one-click switch between menu and ribbon interfaces.
    """
    
    def __init__(self, main_window=None):
        """
        Initialize the menu switch widget.
        
        Parameters
        ----------
        main_window : QtWidgets.QMainWindow, optional
            The main window instance. If None, will try to get the current main window.
        """
        self.main_window = main_window
        if self.main_window is None:
            # Try to get the current main window using multiple methods
            self.main_window = self._find_main_window()
        
        if self.main_window is None:
            raise ValueError("Cannot find main window instance")
    
    def _find_main_window(self):
        """
        Find the main window instance using multiple fallback methods.
        """
        import chisurf
        from chisurf.gui import QtWidgets
        
        # Method 1: Try chisurf.cs (the standard way)
        main_window = getattr(chisurf, 'cs', None)
        if main_window is not None:
            return main_window
        
        # Method 2: Try to find the main window by title
        try:
            for w in QtWidgets.QApplication.topLevelWidgets():
                title = w.windowTitle() if hasattr(w, 'windowTitle') else ''
                if w.isVisible() and ("Chi" in title or "Fit" in title or "PCH" in title or "FIDA" in title):
                    return w
        except Exception:
            pass
        
        # Method 3: Try the active window
        try:
            aw = QtWidgets.QApplication.activeWindow()
            if aw and aw.isVisible():
                return aw
        except Exception:
            pass
        
        # Method 4: Try to find any QMainWindow
        try:
            for w in QtWidgets.QApplication.topLevelWidgets():
                if isinstance(w, QtWidgets.QMainWindow) and w.isVisible():
                    return w
        except Exception:
            pass
        
        return None
    
    def switch_menu_mode(self):
        """
        Switch between menu and ribbon interfaces.
        This method automatically detects the current state and switches to the opposite.
        """
        try:
            # Check current state by looking for ribbon integration
            has_ribbon = getattr(self.main_window, '_ribbon_integration', None) is not None
            
            if has_ribbon:
                # Currently using ribbon, switch to normal menu
                self.main_window.toggle_ribbon_interface(False)
                chisurf.logging.info("Switched to normal menu interface")
            else:
                # Currently using normal menu, switch to ribbon
                self.main_window.toggle_ribbon_interface(True)
                chisurf.logging.info("Switched to ribbon interface")
                
        except Exception as e:
            chisurf.logging.error(f"Failed to switch menu mode: {e}")
    
    def _show_message(self, message):
        """Show an info message to the user."""
        try:
            chisurf.gui.widgets.general.MyMessageBox(
                label="Menu Switch",
                info=message,
                show_fortune=False
            )
        except Exception:
            pass
    
    def _show_error(self, error_message):
        """Show an error message to the user."""
        try:
            chisurf.gui.widgets.general.MyMessageBox(
                label="Menu Switch Error",
                info=error_message,
                show_fortune=False
            )
        except Exception:
            pass


def run():
    """
    Main entry point for the menu switch plugin.
    This function is called when the plugin is activated.
    """
    try:
        widget = MenuSwitchWidget()
        widget.switch_menu_mode()
    except ValueError as e:
        # Handle the case where main window is not found
        error_msg = str(e)
        if "Cannot find main window instance" in error_msg:
            detailed_msg = ("Cannot find main window instance. This may happen if the plugin is "
                          "loaded before the GUI is fully initialized. Try running the menu switch "
                          "again after ChiSurf has finished loading, or use the menu option "
                          "instead of the plugin auto-execution.")
            chisurf.logging.error(f"Menu Switch plugin failed: {detailed_msg}")
            try:
                chisurf.gui.widgets.general.MyMessageBox(
                    label="Menu Switch Error",
                    info=f"Failed to run menu switch: {detailed_msg}",
                    show_fortune=False
                )
            except Exception:
                pass
        else:
            # Re-raise other ValueError exceptions
            raise
    except Exception as e:
        chisurf.logging.error(f"Menu Switch plugin failed: {e}")
        try:
            chisurf.gui.widgets.general.MyMessageBox(
                label="Menu Switch Error",
                info=f"Failed to run menu switch: {e}",
                show_fortune=False
            )
        except Exception:
            pass
