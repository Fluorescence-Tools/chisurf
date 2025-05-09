from PIL import Image, ImageDraw
import os
import random
import math
import numpy as np

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a representation of 2D Fluorescence Intensity Distribution Analysis (FIDA2D)
# Create a 2D histogram-like visualization
hist_x, hist_y = 16, 40
hist_width, hist_height = 32, 24

# Draw histogram axes
draw.line([(hist_x, hist_y), (hist_x + hist_width, hist_y)], fill=(0, 0, 0, 200), width=2)  # x-axis
draw.line([(hist_x, hist_y), (hist_x, hist_y - hist_height)], fill=(0, 0, 0, 200), width=2)  # y-axis
# Draw z-axis (for 3D effect)
draw.line([(hist_x, hist_y), (hist_x - 8, hist_y + 4)], fill=(0, 0, 0, 200), width=2)  # z-axis

# Draw axis labels
draw.text((hist_x + hist_width - 8, hist_y + 2), "k₁", fill=(0, 0, 0, 255))  # channel 1
draw.text((hist_x - 12, hist_y - hist_height + 2), "k₂", fill=(0, 0, 0, 255))  # channel 2
draw.text((hist_x - 10, hist_y + 6), "P", fill=(0, 0, 0, 255))  # probability

# Create a 2D grid for the histogram
grid_size = 8
cell_size = 3

# Generate a 2D Gaussian-like distribution
def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude):
    return amplitude * math.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))

# Draw 2D histogram cells with varying colors based on intensity
for i in range(grid_size):
    for j in range(grid_size):
        # Calculate cell position
        x = hist_x + i * cell_size
        y = hist_y - j * cell_size
        
        # Skip if outside the main axes
        if x > hist_x + hist_width or y < hist_y - hist_height:
            continue
            
        # Calculate intensity using a 2D Gaussian
        # Create two peaks to represent two species
        intensity1 = gaussian_2d(i, j, 2, 2, 1.5, 1.5, 1.0)
        intensity2 = gaussian_2d(i, j, 5, 5, 1.5, 1.5, 0.7)
        intensity = intensity1 + intensity2
        
        # Scale intensity to color
        intensity_scaled = min(255, int(intensity * 200))
        
        # Use a color gradient from blue to red
        r = intensity_scaled
        g = 0
        b = 255 - intensity_scaled
        
        # Draw the cell
        if intensity_scaled > 20:  # Only draw cells with significant intensity
            draw.rectangle(
                [(x, y - cell_size), (x + cell_size, y)],
                fill=(r, g, b, 150),
                outline=(r//2, g, b//2, 200)
            )

# Draw a small fluorescence intensity trace in the corner
trace_x, trace_y = 14, 14
trace_width, trace_height = 10, 8

# Draw a small box for the trace
draw.rectangle(
    [(trace_x, trace_y), (trace_x + trace_width, trace_y + trace_height)],
    fill=(255, 255, 255, 150),
    outline=(0, 0, 0, 255)
)

# Draw two-color fluorescence intensity spikes (representing two channels)
for i in range(trace_width):
    x = trace_x + i
    # Random heights to simulate fluorescence bursts in two channels
    if random.random() < 0.4:  # 40% chance of a burst
        height1 = random.randint(1, trace_height - 2)
        height2 = random.randint(1, trace_height - 2)
        # Channel 1 (red)
        draw.line(
            [(x, trace_y + trace_height), (x, trace_y + trace_height - height1)],
            fill=(200, 0, 0, 200),
            width=1
        )
        # Channel 2 (blue) - slightly offset
        draw.line(
            [(x + 0.5, trace_y + trace_height), (x + 0.5, trace_y + trace_height - height2)],
            fill=(0, 0, 200, 200),
            width=1
        )

# Draw a small "2D" text to indicate 2D-FIDA
draw.text((48, 16), "2D", fill=(0, 0, 0, 255))

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")