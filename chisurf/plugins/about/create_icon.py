"""
Create an icon for the About plugin.

This script generates a simple icon for the About plugin.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create a new image with a transparent background
size = 64
image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)

# Draw a circular background
center = size // 2
radius = size // 2 - 4
draw.ellipse((center - radius, center - radius, center + radius, center + radius), 
             fill=(0, 120, 215, 255))

# Draw the letter "i" for information
# Draw the dot of the "i"
dot_radius = size // 10
draw.ellipse((center - dot_radius, center - radius + 8, 
              center + dot_radius, center - radius + 8 + dot_radius * 2), 
             fill=(255, 255, 255, 255))

# Draw the body of the "i"
rect_width = size // 8
rect_height = size // 2
draw.rectangle((center - rect_width // 2, center - radius + 8 + dot_radius * 3, 
                center + rect_width // 2, center - radius + 8 + dot_radius * 3 + rect_height), 
               fill=(255, 255, 255, 255))

# Save the image
image.save(os.path.join(os.path.dirname(__file__), 'icon.png'))
print(f"Icon saved to {os.path.join(os.path.dirname(__file__), 'icon.png')}")