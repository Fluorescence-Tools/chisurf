"""TTTR Audifier Plugin"""

name = "TTTR:Analysis:Audifier"

# Dynamic icon that changes based on plugin state
# This will be updated by the GUI when state changes
icon = "🎵"  # Default musical note emoji

# Import dynamic icon utilities
try:
    from .dynamic_icons import get_icon_manager
    _icon_manager = get_icon_manager()
    
    def get_current_icon():
        """Get the current dynamic icon."""
        return _icon_manager.get_current_icon()
    
    def set_icon_state(state: str):
        """Set the icon state (idle, playing, paused, processing, error, loaded)."""
        _icon_manager.set_state(state)
        
except ImportError:
    # Fallback if dynamic icons not available
    def get_current_icon():
        return icon
    
    def set_icon_state(state: str):
        pass


def load():
    """Return the plugin's main widget instance."""
    from .gui import TTTRAudifierWidget

    return TTTRAudifierWidget()


# Import lifetime analysis functions for direct access
try:
    from .lifetime_analysis import (
        lifetime_spectrum_ilt,
        compute_lifetime_waterfall,
        plot_lifetime_waterfall,
        plot_lifetime_waterfall_multichannel
    )
except ImportError:
    # Fallback if lifetime analysis not available
    lifetime_spectrum_ilt = None
    compute_lifetime_waterfall = None
    plot_lifetime_waterfall = None
    plot_lifetime_waterfall_multichannel = None


if __name__ == "plugin":  # pragma: no cover - GUI bootstrap
    from qtpy import QtCore
    # Assign to a global variable 'widget' to ensure the object is not
    # garbage collected. The main window maintains a persistent dictionary
    # ('_plugin_contexts') for each plugin directory where this global lives.
    widget = load()
    try:
        widget.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
    except Exception:
        pass
    try:
        widget.resize(1000, 800)
    except Exception:
        pass
    widget.show()
    try:
        widget.raise_()
        widget.activateWindow()
    except Exception:
        pass


__all__ = [
    "name", "load", "icon", "get_current_icon", "set_icon_state",
    "lifetime_spectrum_ilt", "compute_lifetime_waterfall", 
    "plot_lifetime_waterfall", "plot_lifetime_waterfall_multichannel"
]
