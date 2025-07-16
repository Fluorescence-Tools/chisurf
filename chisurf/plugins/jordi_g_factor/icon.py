"""
Script to generate an icon for the Jordi G-Factor Calculator plugin.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create a 128x128 image with a transparent background
img = Image.new('RGBA', (128, 128), color=(0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Draw a blue and red curve representing parallel and perpendicular data
width, height = img.size
center_y = height // 2
x_values = np.linspace(0, width, 100)

# Draw blue curve (parallel)
blue_y_values = [center_y - 20 * np.exp(-((x - width/2) / (width/4))**2) for x in x_values]
for i in range(len(x_values) - 1):
    draw.line(
        (x_values[i], blue_y_values[i], x_values[i+1], blue_y_values[i+1]),
        fill=(0, 0, 255, 255),
        width=3
    )

# Draw red curve (perpendicular)
red_y_values = [center_y + 20 * np.exp(-((x - width/2) / (width/4))**2) for x in x_values]
for i in range(len(x_values) - 1):
    draw.line(
        (x_values[i], red_y_values[i], x_values[i+1], red_y_values[i+1]),
        fill=(255, 0, 0, 255),
        width=3
    )

# Draw a vertical line representing the g-factor
draw.line(
    (width/2, 20, width/2, height-20),
    fill=(0, 200, 0, 200),
    width=2
)

# Draw a "G" letter
try:
    font = ImageFont.truetype("arial.ttf", 40)
except IOError:
    font = ImageFont.load_default()

draw.text((width/2-15, height/2-20), "G", fill=(0, 0, 0, 255), font=font)

# Save the image
img.save(os.path.join(os.path.dirname(__file__), 'icon.png'))

print("Icon created successfully!")

if __name__ == "__main__":
    print("Icon saved to:", os.path.join(os.path.dirname(__file__), 'icon.png'))