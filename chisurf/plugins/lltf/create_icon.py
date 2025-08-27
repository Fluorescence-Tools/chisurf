"""
Create an icon for the Lazy Lifetime Analysis (LLTF) plugin.

Adds decay curves, fitted line, clock, tau symbol, and a 'Z' to represent laziness.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_lltf_icon(filename='icon.png'):
    size = 64
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Background circle
    center = size // 2
    radius = size // 2 - 4
    draw.ellipse((center - radius, center - radius, center + radius, center + radius),
                 fill=(50, 100, 170, 255))  # Dark blue

    # Decay data + fit
    start_x = 10
    end_x = size - 10
    mid_y = size // 2
    np.random.seed(42)

    # Draw noisy decay data points
    points_data = []
    for x in range(start_x, end_x + 1, 2):
        rel_x = (x - start_x) / (end_x - start_x)
        y = mid_y - 20 * np.exp(-2.5 * rel_x) + np.random.normal(0, 0.5)
        points_data.append((x, y))

    # Fitted decay curve
    points_fit = [(x, mid_y - 20 * np.exp(-2.5 * (x - start_x) / (end_x - start_x)))
                  for x in range(start_x, end_x + 1)]

    # Draw points
    for x, y in points_data:
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(230, 230, 230, 255))

    # Draw fit line
    for i in range(len(points_fit) - 1):
        draw.line([points_fit[i], points_fit[i + 1]], fill=(0, 255, 0, 255), width=2)

    # Tilted clock for laziness
    clock_cx, clock_cy = size - 14, 14
    r = 6
    bbox = [clock_cx - r, clock_cy - r, clock_cx + r, clock_cy + r]
    draw.ellipse(bbox, fill=(255, 255, 255, 230), outline=(0, 0, 0, 255))

    # Tilted clock hands
    draw.line([(clock_cx, clock_cy), (clock_cx + r * 0.4, clock_cy + r * 0.3)],
              fill=(0, 0, 0, 255), width=1)
    draw.line([(clock_cx, clock_cy), (clock_cx - r * 0.6, clock_cy + r * 0.1)],
              fill=(0, 0, 0, 255), width=1)

    # Add a small "Z" above the clock
    try:
        font_z = ImageFont.truetype("arial.ttf", 10)
    except:
        font_z = ImageFont.load_default()
    draw.text((clock_cx - 2, clock_cy - r - 8), "Z", font=font_z, fill=(255, 255, 255, 255))

    # Draw a tau symbol (italic τ)
    try:
        font_tau = ImageFont.truetype("ariali.ttf", 14)  # italic if available
    except:
        font_tau = ImageFont.load_default()
    draw.text((14, 12), "τ", font=font_tau, fill=(255, 255, 255, 255))

    # Save icon
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    image.save(icon_path)
    print(f"Icon saved to {icon_path}")

if __name__ == '__main__':
    create_lltf_icon()
