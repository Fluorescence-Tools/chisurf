"""
Create an icon for the updater plugin.

This script generates a simple icon for the updater plugin.
"""

import os
import numpy as np
from PIL import Image, ImageDraw

# Create a new image with a transparent background
size = 64
image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)

# Draw a circular arrow to represent updating
center = size // 2
radius = size // 3
arrow_width = size // 10

# Draw the circular part of the arrow
for i in range(0, 270):
    angle = i * np.pi / 180
    x = center + int(radius * np.cos(angle))
    y = center + int(radius * np.sin(angle))
    draw.ellipse((x - arrow_width//2, y - arrow_width//2, 
                  x + arrow_width//2, y + arrow_width//2), 
                 fill=(0, 120, 215, 255))

# Draw the arrow head
arrow_head_length = size // 4
arrow_head_width = size // 6
angle = 270 * np.pi / 180
x_end = center + int(radius * np.cos(angle))
y_end = center + int(radius * np.sin(angle))

# Calculate arrow head points
x1 = x_end + int(arrow_head_length * np.cos(angle - np.pi/6))
y1 = y_end + int(arrow_head_length * np.sin(angle - np.pi/6))
x2 = x_end + int(arrow_head_length * np.cos(angle + np.pi/6))
y2 = y_end + int(arrow_head_length * np.sin(angle + np.pi/6))

# Draw the arrow head
draw.polygon([(x_end, y_end), (x1, y1), (x2, y2)], fill=(0, 120, 215, 255))

# Save the image
image.save(os.path.join(os.path.dirname(__file__), 'icon.png'))
print(f"Icon saved to {os.path.join(os.path.dirname(__file__), 'icon.png')}")