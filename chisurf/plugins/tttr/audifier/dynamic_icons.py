"""
Dynamic icon utilities for TTTR Audifier plugin.
Provides context-aware icons that change based on plugin state.
"""

from qtpy.QtCore import QObject, QTimer, Signal
from qtpy.QtGui import QIcon

from chisurf.plugins.icon_utils import create_emoji_icon, create_text_icon


class DynamicIconManager(QObject):
    """Manages dynamic icons that can change based on application state."""

    icon_changed = Signal(QIcon)  # Emitted when icon changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_icon = None
        self._state = "idle"  # idle, playing, paused, processing, error
        self._base_emoji = "🎵"
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_animated_icon)
        self._animation_frame = 0

    def set_state(self, state: str):
        """Set the current state and update icon accordingly."""
        if self._state != state:
            self._state = state
            self._update_icon()

    def get_current_icon(self, size: int = 64) -> QIcon:
        """Get the current icon for the given state."""
        return self._create_state_icon(self._state, size)

    def _create_state_icon(self, state: str, size: int = 64) -> QIcon:
        """Create icon based on current state."""
        if state == "playing":
            # Animated music notes
            emojis = ["🎵", "🎶", "🎼", "🎤"]
            emoji = emojis[self._animation_frame % len(emojis)]
            return create_emoji_icon(emoji, size=size, bg_color="#4CAF50")
        elif state == "paused":
            return create_emoji_icon("⏸️", size=size, bg_color="#FF9800")
        elif state == "processing":
            # Animated processing indicator
            processing_emojis = ["Wait", "Processing", "Done"]
            emoji = processing_emojis[self._animation_frame % len(processing_emojis)]
            return create_emoji_icon(emoji, size=size, bg_color="#2196F3")
        elif state == "error":
            return create_emoji_icon("❌", size=size, bg_color="#F44336")
        elif state == "loaded":
            return create_emoji_icon("🎵", size=size, bg_color="#9C27B0")
        else:  # idle
            return create_emoji_icon(self._base_emoji, size=size)

    def _update_icon(self):
        """Update the current icon and emit change signal."""
        new_icon = self._create_state_icon(self._state)
        if new_icon != self._current_icon:
            self._current_icon = new_icon
            self.icon_changed.emit(new_icon)

    def _update_animated_icon(self):
        """Update animation frame for animated states."""
        if self._state in ["playing", "processing"]:
            self._animation_frame += 1
            self._update_icon()
        else:
            self._update_timer.stop()
            self._animation_frame = 0

    def start_animation(self):
        """Start icon animation for states that need it."""
        if self._state in ["playing", "processing"]:
            self._update_timer.start(500)  # Update every 500ms

    def stop_animation(self):
        """Stop icon animation."""
        self._update_timer.stop()
        self._animation_frame = 0


# Global icon manager instance
_icon_manager = None


def get_icon_manager() -> DynamicIconManager:
    """Get the global icon manager instance."""
    global _icon_manager
    if _icon_manager is None:
        _icon_manager = DynamicIconManager()
    return _icon_manager


def get_dynamic_icon(state: str = "idle", size: int = 64) -> QIcon:
    """Get a dynamic icon for a specific state."""
    manager = get_icon_manager()
    return manager._create_state_icon(state, size)


def create_context_aware_icon(
    base_text: str = "AUD",
    data_loaded: bool = False,
    is_playing: bool = False,
    is_processing: bool = False,
    has_error: bool = False,
    size: int = 64,
) -> QIcon:
    """
    Create a context-aware icon based on the current plugin state.

    Parameters
    ----------
    base_text : str
        Base text for the icon
    data_loaded : bool
        Whether TTTR data is loaded
    is_playing : bool
        Whether audio is currently playing
    is_processing : bool
        Whether the plugin is processing data
    has_error : bool
        Whether there's an error state
    size : int
        Icon size

    Returns
    -------
    QIcon
        Context-aware icon
    """
    if has_error:
        return create_text_icon("ERR", size=size, bg_color="#F44336", text_color="white")
    elif is_playing:
        return create_text_icon(
            "▶", size=size, bg_color="#4CAF50", text_color="white", shape="circle"
        )
    elif is_processing:
        return create_text_icon("⏳", size=size, bg_color="#2196F3", text_color="white")
    elif data_loaded:
        return create_text_icon(base_text, size=size, bg_color="#9C27B0", text_color="white")
    else:
        return create_text_icon(base_text, size=size, bg_color="#607D8B", text_color="white")


def create_audifier_icon_from_state(
    data_loaded: bool = False, selected_channels: int = 0, is_playing: bool = False, size: int = 64
) -> QIcon:
    """
    Create audifier-specific icon based on current state.

    Parameters
    ----------
    data_loaded : bool
        Whether TTTR data is loaded
    selected_channels : int
        Number of selected channels
    is_playing : bool
        Whether audio is playing
    size : int
        Icon size

    Returns
    -------
    QIcon
        State-specific audifier icon
    """
    if is_playing:
        # Show music note when playing
        return create_emoji_icon("🎵", size=size, bg_color="#4CAF50")
    elif data_loaded and selected_channels > 0:
        # Show channel count when data is loaded
        return create_text_icon(
            f"{selected_channels}", size=size, bg_color="#9C27B0", text_color="white"
        )
    elif data_loaded:
        # Show loaded state
        return create_text_icon("LD", size=size, bg_color="#9C27B0", text_color="white")
    else:
        # Default idle state
        return create_emoji_icon("🎵", size=size)
