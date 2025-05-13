"""
Create an icon for the UCFRET plugin.

This script creates a simple icon for the UCFRET plugin.
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

# Create a new image with a white background
width, height = 128, 128
image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
draw = ImageDraw.Draw(image)

# Define colors
blue = (41, 128, 185, 255)  # Donor color
red = (231, 76, 60, 255)    # Acceptor color
green = (46, 204, 113, 255) # Energy transfer color

# Draw a circle for the donor
donor_center = (width // 3, height // 3)
donor_radius = 25
draw.ellipse(
    (
        donor_center[0] - donor_radius,
        donor_center[1] - donor_radius,
        donor_center[0] + donor_radius,
        donor_center[1] + donor_radius
    ),
    fill=blue
)

# Draw a circle for the acceptor
acceptor_center = (2 * width // 3, 2 * height // 3)
acceptor_radius = 25
draw.ellipse(
    (
        acceptor_center[0] - acceptor_radius,
        acceptor_center[1] - acceptor_radius,
        acceptor_center[0] + acceptor_radius,
        acceptor_center[1] + acceptor_radius
    ),
    fill=red
)

# Draw an arrow for energy transfer
arrow_width = 8
arrow_points = [
    (donor_center[0] + donor_radius - 5, donor_center[1] + donor_radius - 5),  # Start
    (acceptor_center[0] - acceptor_radius + 5, acceptor_center[1] - acceptor_radius + 5)  # End
]
draw.line(arrow_points, fill=green, width=arrow_width)

# Draw arrowhead
arrowhead_size = 12
arrowhead_points = [
    (arrow_points[1][0], arrow_points[1][1]),
    (arrow_points[1][0] - arrowhead_size, arrow_points[1][1] - arrowhead_size // 2),
    (arrow_points[1][0] - arrowhead_size // 2, arrow_points[1][1] - arrowhead_size)
]
draw.polygon(arrowhead_points, fill=green)

# Add "FRET" text
try:
    font = ImageFont.truetype("arial.ttf", 24)
except IOError:
    font = ImageFont.load_default()

text = "FRET"
text_width, text_height = draw.textsize(text, font=font)
text_position = ((width - text_width) // 2, height - text_height - 10)
draw.text(text_position, text, fill=(0, 0, 0, 255), font=font)

# Save the image
icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
image.save(icon_path)

print(f"Icon saved to {icon_path}")

if __name__ == "__main__":
    # If run directly, show the image
    image.show()