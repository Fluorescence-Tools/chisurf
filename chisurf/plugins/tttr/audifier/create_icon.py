"""
Create a fallback icon for the audifier plugin.
This ensures the icon displays even if the dynamic system fails.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import pathlib

def create_audifier_icon():
    """Create a musical note icon for the audifier plugin."""
    size = 64
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple musical note shape
    # Background circle
    bg_color = (76, 175, 80, 255)  # Green
    draw.ellipse([4, 4, size-4, size-4], fill=bg_color, outline=(0, 0, 0, 255), width=2)
    
    # Draw musical note
    note_color = (255, 255, 255, 255)
    
    # Note head (ellipse)
    draw.ellipse([16, 40, 28, 52], fill=note_color, outline=(0, 0, 0, 255))
    
    # Note stem
    draw.rectangle([26, 20, 28, 40], fill=note_color)
    
    # Note flag
    points = [(28, 20), (40, 24), (40, 32), (28, 28)]
    draw.polygon(points, fill=note_color)
    
    # Save the icon
    here = pathlib.Path(__file__).parent
    icon_path = here / "icon.png"
    img.save(icon_path)
    print(f"Audifier icon created at {icon_path}")
    return icon_path

if __name__ == "__main__":
    create_audifier_icon()
