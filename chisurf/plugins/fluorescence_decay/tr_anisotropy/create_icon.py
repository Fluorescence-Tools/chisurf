"""
Create an icon for the Time-Resolved Anisotropy plugin.

This script generates an icon representing fluorescence anisotropy with
polarization components and decay curves.
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
             fill=(70, 130, 180, 255))  # Steel blue color

# Draw a stylized representation of anisotropy decay
# Starting point for the curves
start_x = 10
end_x = size - 10
mid_y = size // 2

# Draw VV (parallel) decay curve - higher intensity, slower decay
points_vv = []
for x in range(start_x, end_x + 1, 2):
    # Exponential decay curve for VV
    rel_x = (x - start_x) / (end_x - start_x)
    y = mid_y - 15 * np.exp(-3 * rel_x)
    points_vv.append((x, y))

# Draw VH (perpendicular) decay curve - lower intensity, faster approach to equilibrium
points_vh = []
for x in range(start_x, end_x + 1, 2):
    # Exponential growth curve for VH (approaching equilibrium from below)
    rel_x = (x - start_x) / (end_x - start_x)
    y = mid_y + 5 * (1 - np.exp(-5 * rel_x))
    points_vh.append((x, y))

# Draw the curves with thicker lines
# VV curve in blue
for i in range(len(points_vv) - 1):
    draw.line([points_vv[i], points_vv[i+1]], fill=(0, 0, 255, 255), width=2)

# VH curve in red
for i in range(len(points_vh) - 1):
    draw.line([points_vh[i], points_vh[i+1]], fill=(255, 0, 0, 255), width=2)

# Draw a small polarization symbol in the top right
pol_center_x = size - 15
pol_center_y = 15
pol_size = 8

# Vertical polarization line
draw.line([(pol_center_x, pol_center_y - pol_size), 
           (pol_center_x, pol_center_y + pol_size)], 
          fill=(255, 255, 255, 255), width=2)

# Horizontal polarization line
draw.line([(pol_center_x - pol_size, pol_center_y), 
           (pol_center_x + pol_size, pol_center_y)], 
          fill=(255, 255, 255, 255), width=2)

# Save the image
icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
image.save(icon_path)
print(f"Icon saved to {icon_path}")
