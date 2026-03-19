# Enhanced Plugin Icon System

This document describes the enhanced icon system for ChiSurf plugins, which supports emojis, text, colors, and traditional image files.

## Overview

The enhanced icon system allows plugins to specify icons using various formats:

1. **Emoji icons** - Unicode emoji characters (e.g., "🎵", "⭐", "🔧")
2. **Text icons** - Short text strings (e.g., "CALC", "ANA", "DEV")
3. **Color icons** - Hex color codes or color names (e.g., "#FF5722", "red")
4. **File icons** - Traditional PNG/SVG image files
5. **QIcon objects** - Direct QIcon instances

## Usage

### Basic Emoji Icon

```python
# In your plugin's __init__.py
name = "My:Plugin:Name"
icon = "🎵"  # Musical note emoji
```

### Text Icon

```python
name = "My:Analysis:Plugin"
icon = "ANA"  # Will be styled with analysis theme
```

### Color Icon

```python
name = "My:Plugin:Name"
icon = "#FF5722"  # Red color
```

### Advanced Icon Creation

For more control, you can use the icon utilities directly:

```python
from chisurf.plugins.icon_utils import (
    create_emoji_icon, create_text_icon, create_analysis_icon
)

# Create a custom emoji icon with background
icon = create_emoji_icon("🎵", size=64, bg_color="#2196F3")

# Create a custom text icon
icon = create_text_icon("CALC", size=64, bg_color="#4CAF50", shape="circle")

# Create a themed icon
icon = create_analysis_icon("ANA", size=64)
```

## Available Functions

### Core Functions

- `create_text_icon(text, size, bg_color, text_color, font_size, font_family, shape)`
- `create_emoji_icon(emoji, size, bg_color)`
- `resolve_plugin_icon(icon_value, size, fallback_text)`
- `create_plugin_icon_with_fallback(module, package_dir, size)`

### Themed Icons

- `create_analysis_icon(text, size)` - Blue background, rounded corners
- `create_tool_icon(text, size)` - Green background, square
- `create_dev_icon(text, size)` - Orange background, rounded corners

## Icon Shapes

Text icons support different shapes:

- `"square"` - Square shape (default)
- `"circle"` - Circular shape
- `"rounded"` - Rounded rectangle

## Fallback System

The enhanced icon system provides automatic fallbacks:

1. **Module icon attribute** - `icon = "🎵"` or `icon = create_text_icon(...)`
2. **icon.png file** - Traditional image file
3. **icon.svg file** - Vector image file
4. **Plugin name abbreviation** - Generated from plugin name
5. **Default icon** - Question mark in gray

## Examples

### Audifier Plugin (Enhanced)

```python
"""TTTR Audifier Plugin"""

name = "TTTR:Analysis:Audifier"
icon = "🎵"  # Musical note emoji with automatic styling
```

### Analysis Plugin

```python
"""Data Analysis Plugin"""

name = "Analysis:Statistics"
icon = "STAT"  # Will be styled with analysis theme
```

### Development Plugin

```python
"""Development Tools Plugin"""

name = "Dev:Tools"
icon = "🔧"  # Wrench emoji for tools
```

### Color-Coded Plugin

```python
"""Important Plugin"""

name = "Critical:Alert"
icon = "#F44336"  # Red color for importance
```

## Testing

Run the test script to validate the icon system:

```bash
cd chisurf/plugins
python test_icon_system.py
```

This will show a window with various icon examples and test results.

## Migration Guide

### From Traditional Icons

**Before:**
```python
# Need icon.png file in plugin directory
name = "My:Plugin"
# No icon specified
```

**After:**
```python
name = "My:Plugin"
icon = "🎵"  # Simple emoji
```

### From Complex QIcon Creation

**Before:**
```python
from qtpy.QtGui import QIcon, QPixmap, QPainter
# Complex painting code...
icon = my_custom_icon
```

**After:**
```python
from chisurf.plugins.icon_utils import create_emoji_icon
icon = create_emoji_icon("🎵", size=64, bg_color="#2196F3")
```

## Best Practices

1. **Use meaningful emojis** - Choose emojis that represent your plugin's function
2. **Keep text short** - Use 2-4 characters for text icons
3. **Consider accessibility** - Ensure icons are distinguishable
4. **Test different sizes** - Icons should work at 16x16, 32x32, and 64x64
5. **Use themes** - Prefer themed icons for consistency

## Troubleshooting

### Emoji Not Displaying

- Ensure system has emoji font support
- Try different emoji characters
- Check Qt version compatibility

### Text Not Visible

- Adjust text color for contrast with background
- Increase font size for better readability
- Use different shape if text is cut off

### Icon Not Loading

- Check import path for icon_utils
- Verify icon string format
- Test with fallback system

## Technical Details

The enhanced icon system uses Qt's painting system to render icons dynamically:

- Emojis are rendered using text rendering with emoji fonts
- Text icons use QFont for optimal rendering
- Colors are applied using QColor
- Shapes are drawn with QPainter primitives

This approach provides consistent, scalable icons without requiring external image files for most use cases.
