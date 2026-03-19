"""
Enhanced icon utilities for ChiSurf plugins supporting emojis and text icons.
"""
import os
import pathlib
from typing import Union, Optional
from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtGui import QIcon, QPixmap, QPainter, QFont, QColor, QPen
from qtpy.QtCore import Qt, QSize


def create_text_icon(
    text: str,
    size: int = 64,
    bg_color: Optional[str] = None,
    text_color: str = "#000000",
    font_size: Optional[int] = None,
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


def create_emoji_icon(emoji: str, size: int = 64, bg_color: Optional[str] = None) -> QIcon:
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
    icon_value: Union[str, QIcon, None],
    size: int = 64,
    fallback_text: Optional[str] = None
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
        if os.path.exists(icon_value):
            return QIcon(icon_value)
        
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
    package_dir: Union[str, pathlib.Path],
    size: int = 64
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
    
    Returns
    -------
    QIcon
        Resolved icon
    """
    package_dir = pathlib.Path(package_dir)
    
    # 1. Check for module-level icon attribute
    if hasattr(module, 'icon'):
        try:
            return resolve_plugin_icon(module.icon, size=size)
        except Exception:
            pass
    
    # 2. Check for icon.png file
    icon_png_path = package_dir / 'icon.png'
    if icon_png_path.exists():
        try:
            return QIcon(str(icon_png_path))
        except Exception:
            pass
    
    # 3. Check for icon.svg file
    icon_svg_path = package_dir / 'icon.svg'
    if icon_svg_path.exists():
        try:
            return QIcon(str(icon_svg_path))
        except Exception:
            pass
    
    # 4. Create fallback from plugin name
    plugin_name = getattr(module, 'name', None)
    if plugin_name:
        # Extract first letter or create abbreviation
        if ':' in plugin_name:
            # Use the last part after colon for abbreviation
            parts = plugin_name.split(':')
            name_part = parts[-1]
        else:
            name_part = plugin_name
        
        # Create abbreviation (first 2-3 letters)
        if len(name_part) >= 3:
            fallback_text = name_part[:3].upper()
        else:
            fallback_text = name_part.upper()
        
        return create_text_icon(fallback_text, size=size, bg_color="#e0e0e0")
    
    # 5. Ultimate fallback
    return create_text_icon("?", size=size, bg_color="#cccccc")


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
