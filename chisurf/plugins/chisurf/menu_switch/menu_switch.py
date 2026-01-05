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
            # Try to get the current main window from chisurf
            import chisurf
            self.main_window = getattr(chisurf, 'cs', None)
        
        if self.main_window is None:
            raise ValueError("Cannot find main window instance")
    
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
                self._show_message("Switched to normal menu interface")
            else:
                # Currently using normal menu, switch to ribbon
                self.main_window.toggle_ribbon_interface(True)
                chisurf.logging.info("Switched to ribbon interface")
                self._show_message("Switched to ribbon interface")
                
        except Exception as e:
            chisurf.logging.error(f"Failed to switch menu mode: {e}")
            self._show_error(f"Failed to switch menu mode: {e}")
    
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
