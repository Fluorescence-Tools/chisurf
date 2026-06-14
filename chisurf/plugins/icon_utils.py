"""Enhanced icon utilities for ChiSurf plugins supporting image, emoji, and text icons."""
import pathlib

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap

ICON_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".svg", ".ico"}


def create_text_icon(
    text: str,
    size: int = 64,
    bg_color: str | None = None,
    text_color: str = "#000000",
    font_size: int | None = None,
    font_family: str = "Arial",
    shape: str = "square"  # "square", "circle", "rounded"
) -> QIcon:
    """
    Create an icon from text or emoji.

    Parameters
    ----------
    text : str
        Text or emoji to display
    size : int
        Icon size in pixels
    bg_color : str, optional
        Background color (hex or name). If None, uses transparent background
    text_color : str
        Text color (hex or name)
    font_size : int, optional
        Font size. If None, auto-calculated based on size
    font_family : str
        Font family name
    shape : str
        Icon shape: "square", "circle", or "rounded"

    Returns
    -------
    QIcon
        Generated icon
    """
    # Create pixmap
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)

    # Setup painter
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)

    # Draw background if specified
    if bg_color:
        bg_color_obj = QColor(bg_color)
        painter.setBrush(bg_color_obj)
        painter.setPen(Qt.NoPen)

        if shape == "circle":
            painter.drawEllipse(0, 0, size, size)
        elif shape == "rounded":
            radius = size // 8
            painter.drawRoundedRect(0, 0, size, size, radius, radius)
        else:  # square
            painter.drawRect(0, 0, size, size)

    # Setup font
    font = QFont(font_family)
    if font_size is None:
        # Auto-calculate font size based on icon size and text length
        if len(text) <= 2:
            font_size = int(size * 0.6)
        elif len(text) <= 4:
            font_size = int(size * 0.4)
        else:
            font_size = int(size * 0.3)
    font.setPointSize(font_size)
    font.setBold(True)
    painter.setFont(font)

    # Setup text color
    text_color_obj = QColor(text_color)
    painter.setPen(QPen(text_color_obj))

    # Draw text centered
    rect = QtCore.QRect(0, 0, size, size)
    flags = Qt.AlignCenter | Qt.AlignVCenter
    painter.drawText(rect, flags, text)

    painter.end()

    return QIcon(pm)


def create_emoji_icon(emoji: str, size: int = 64, bg_color: str | None = None) -> QIcon:
    """
    Create an icon from an emoji character.

    Parameters
    ----------
    emoji : str
        Emoji character(s)
    size : int
        Icon size in pixels
    bg_color : str, optional
        Background color. If None, uses transparent background

    Returns
    -------
    QIcon
        Generated icon
    """
    return create_text_icon(
        text=emoji,
        size=size,
        bg_color=bg_color,
        text_color="#000000",  # Emojis are typically colored already
        font_size=int(size * 0.7),
        font_family="Segoe UI Emoji",  # Good emoji font on Windows
        shape="circle"
    )


def resolve_plugin_icon(
    icon_value: str | QIcon | None,
    size: int = 64,
    fallback_text: str | None = None,
    base_dir: str | pathlib.Path | None = None,
) -> QIcon:
    """
    Resolve various icon formats into a QIcon object.

    Parameters
    ----------
    icon_value : str, QIcon, or None
        Icon specification:
        - str: Can be emoji, text, file path, or color name
        - QIcon: Used directly
        - None: Creates default icon
    size : int
        Default size for generated icons
    fallback_text : str, optional
        Text to use if icon_value is None
    base_dir : str or pathlib.Path, optional
        Directory used to resolve relative image paths.

    Returns
    -------
    QIcon
        Resolved icon
    """
    if icon_value is None:
        if fallback_text:
            return create_text_icon(fallback_text, size=size)
        else:
            return create_text_icon("?", size=size, bg_color="#cccccc")

    if isinstance(icon_value, QIcon):
        return icon_value

    if isinstance(icon_value, str):
        # Check if it's an emoji (contains Unicode emoji characters)
        if any(ord(char) > 0x1F000 for char in icon_value):
            return create_emoji_icon(icon_value, size=size)

        # Check if it's a file path
        path = pathlib.Path(icon_value).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = pathlib.Path(base_dir) / path
        if path.exists() and path.suffix.lower() in ICON_IMAGE_SUFFIXES:
            return QIcon(str(path))

        # Check if it's a color name (create a solid color icon)
        color = QColor(icon_value)
        if color.isValid():
            pm = QPixmap(size, size)
            pm.fill(color)
            return QIcon(pm)

        # Treat as text
        return create_text_icon(icon_value, size=size)

    # Fallback
    return create_text_icon(str(icon_value), size=size)


def create_plugin_icon_with_fallback(
    module,
    package_dir: str | pathlib.Path,
    size: int = 64,
    manifest=None,
) -> QIcon:
    """
    Create plugin icon with comprehensive fallback system.

    Parameters
    ----------
    module : module
        Plugin module object
    package_dir : str or pathlib.Path
        Plugin package directory
    size : int
        Icon size
    manifest : PluginManifest, optional
        Plugin manifest. If present, its icon field is preferred over legacy
        module attributes because it is user-editable metadata.

    Returns
    -------
    QIcon
        Resolved icon
    """
    package_dir = pathlib.Path(package_dir)

    # 1. Prefer manifest icon metadata when available.
    if manifest is not None and getattr(manifest, "icon", None):
        try:
            return resolve_plugin_icon(manifest.icon, size=size, base_dir=package_dir)
        except Exception:
            pass

    # 2. Check for common image files before module-level text or emoji icons.
    for icon_name in ("icon.png", "icon.svg", "icon.jpg", "icon.jpeg", "icon.ico"):
        icon_path = package_dir / icon_name
        if icon_path.exists():
            try:
                return QIcon(str(icon_path))
            except Exception:
                pass

    # 3. Check for module-level icon attribute
    if hasattr(module, 'icon'):
        try:
            return resolve_plugin_icon(module.icon, size=size, base_dir=package_dir)
        except Exception:
            pass

    # 4. Create fallback from plugin name
    plugin_name = getattr(module, 'name', None)
    if plugin_name:
        return create_text_icon(_plugin_icon_label(plugin_name), size=size, bg_color="#e0e0e0")

    # 5. Ultimate fallback
    return create_text_icon("?", size=size, bg_color="#cccccc")


def plugin_icon_path(package_dir: str | pathlib.Path) -> pathlib.Path:
    """
    Return the canonical editable icon image path for a plugin.

    Parameters
    ----------
    package_dir : str or pathlib.Path
        Plugin package directory.

    Returns
    -------
    pathlib.Path
        Path to ``icon.png`` in the plugin package.
    """
    return pathlib.Path(package_dir) / "icon.png"


def _plugin_icon_label(plugin_name: str) -> str:
    """
    Return a compact text label for generated plugin icons.

    Parameters
    ----------
    plugin_name : str
        Human-readable plugin name.

    Returns
    -------
    str
        Uppercase one- to three-letter label.
    """
    name_part = plugin_name.split(":")[-1].strip()
    words = [word for word in name_part.replace("-", " ").replace("_", " ").split() if word]
    if len(words) >= 2:
        return "".join(word[0] for word in words[:3]).upper()
    return (name_part[:3] or "?").upper()


# Utility functions for common icon patterns
def create_analysis_icon(text: str, size: int = 64) -> QIcon:
    """Create an analysis-themed icon with blue background."""
    return create_text_icon(
        text=text,
        size=size,
        bg_color="#2196F3",
        text_color="white",
        shape="rounded"
    )


def create_tool_icon(text: str, size: int = 64) -> QIcon:
    """Create a tool-themed icon with green background."""
    return create_text_icon(
        text=text,
        size=size,
        bg_color="#4CAF50",
        text_color="white",
        shape="square"
    )


def create_dev_icon(text: str, size: int = 64) -> QIcon:
    """Create a development-themed icon with orange background."""
    return create_text_icon(
        text=text,
        size=size,
        bg_color="#FF9800",
        text_color="white",
        shape="rounded"
    )
