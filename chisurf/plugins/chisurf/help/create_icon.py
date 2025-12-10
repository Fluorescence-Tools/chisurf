"""
Create an icon for the Help plugin.

This script generates a simple icon for the Help plugin.
"""

import os
import numpy as np
from PIL import Image, ImageDraw

# Create a new image with a transparent background
size = 64
image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)

# Draw a circular background
center = size // 2
radius = size // 2 - 4
draw.ellipse((center - radius, center - radius, center + radius, center + radius), 
             fill=(0, 120, 215, 255))

# Draw a question mark
# Draw the curve of the question mark
curve_points = []
for i in range(0, 180):
    angle = (i + 45) * np.pi / 180
    x = center + int((radius * 0.6) * np.cos(angle))
    y = center - int((radius * 0.6) * np.sin(angle)) - radius // 4
    curve_points.append((x, y))

# Add the stem of the question mark
stem_top = curve_points[-1]
stem_bottom = (stem_top[0], stem_top[1] + radius // 3)
curve_points.append(stem_bottom)

# Draw the curve with a thick line
for i in range(len(curve_points) - 1):
    draw.line([curve_points[i], curve_points[i+1]], fill=(255, 255, 255, 255), width=size // 8)

# Draw the dot of the question mark
dot_radius = size // 10
dot_center = (center, center + radius // 2)
draw.ellipse((dot_center[0] - dot_radius, dot_center[1] - dot_radius, 
              dot_center[0] + dot_radius, dot_center[1] + dot_radius), 
             fill=(255, 255, 255, 255))

# Save the image
image.save(os.path.join(os.path.dirname(__file__), 'icon.png'))
print(f"Icon saved to {os.path.join(os.path.dirname(__file__), 'icon.png')}")