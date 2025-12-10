"""
Script to generate an icon for the Count Rate Analysis plugin.

This creates a 64x64 pixel icon that visually represents count rate analysis
with a bar chart, mean line, and error indicators.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create a 64x64 image with a transparent background
img = Image.new('RGBA', (64, 64), color=(0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Draw a light background rectangle
draw.rectangle([(6, 6), (58, 58)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Define the chart area
chart_x, chart_y = 10, 10
chart_width, chart_height = 44, 44

# Draw axes
draw.line([(chart_x, chart_y + chart_height), (chart_x + chart_width, chart_y + chart_height)], 
          fill=(0, 0, 0, 200), width=2)  # x-axis
draw.line([(chart_x, chart_y), (chart_x, chart_y + chart_height)], 
          fill=(0, 0, 0, 200), width=2)  # y-axis

# Draw axis labels
try:
    font = ImageFont.truetype("arial.ttf", 10)
except IOError:
    font = ImageFont.load_default()

draw.text((chart_x + chart_width - 8, chart_y + chart_height + 2), "t", fill=(0, 0, 0, 255), font=font)  # time
draw.text((chart_x - 8, chart_y), "Hz", fill=(0, 0, 0, 255), font=font)  # count rate

# Generate some random data for the bars (representing count rates)
np.random.seed(42)  # For reproducibility
n_bars = 8
bar_width = chart_width / (n_bars * 1.5)
bar_spacing = bar_width * 0.5
# Make sure bar heights are within the chart area
max_height = chart_height - 5  # Leave some margin
bar_heights = np.random.uniform(5, max_height, n_bars)
mean_height = np.mean(bar_heights)
std_height = np.std(bar_heights)

# Colors for different channels
colors = [(0, 0, 200, 200),  # Blue
          (200, 0, 0, 200),  # Red
          (0, 150, 0, 200)]  # Green

# Draw the bars (representing count rates for different files)
for i in range(n_bars):
    x = chart_x + (i * (bar_width + bar_spacing)) + bar_spacing
    height = min(bar_heights[i], chart_height - 1)  # Ensure height doesn't exceed chart
    y = chart_y + chart_height - height
    color = colors[i % len(colors)]
    
    # Draw the bar (ensure coordinates are in the right order: [(x0, y0), (x1, y1)] where x0 <= x1 and y0 <= y1)
    top_left = (x, y)
    bottom_right = (x + bar_width, chart_y + chart_height)
    draw.rectangle([top_left, bottom_right], 
                   fill=color, outline=(0, 0, 0, 255))

# Draw the mean line
draw.line([(chart_x, chart_y + chart_height - mean_height), 
           (chart_x + chart_width, chart_y + chart_height - mean_height)], 
          fill=(0, 0, 0, 200), width=1, joint="curve")

# Draw error bars (standard deviation)
error_top = chart_y + chart_height - (mean_height + std_height)
error_bottom = chart_y + chart_height - (mean_height - std_height)
draw.line([(chart_x + 5, error_top), (chart_x + chart_width - 5, error_top)], 
          fill=(100, 100, 100, 150), width=1, joint="curve")
draw.line([(chart_x + 5, error_bottom), (chart_x + chart_width - 5, error_bottom)], 
          fill=(100, 100, 100, 150), width=1, joint="curve")

# Draw vertical lines connecting the error bars
for x in [chart_x + 5, chart_x + chart_width - 5]:
    draw.line([(x, error_top), (x, error_bottom)], fill=(100, 100, 100, 150), width=1)

# Save the image
img.save(os.path.join(os.path.dirname(__file__), 'icon.png'))

print("Count Rate Analysis icon created successfully!")

if __name__ == "__main__":
    print("Icon saved to:", os.path.join(os.path.dirname(__file__), 'icon.png'))