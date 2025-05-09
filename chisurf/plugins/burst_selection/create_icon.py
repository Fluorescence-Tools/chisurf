from PIL import Image, ImageDraw
import os
import math
import random

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a time trace with bursts
# X and Y axes
draw.line([(12, 44), (52, 44)], fill=(0, 0, 0, 255), width=1)  # X-axis (time)
draw.line([(12, 12), (12, 44)], fill=(0, 0, 0, 255), width=1)  # Y-axis (intensity)

# Add small ticks to the axes
for i in range(4):
    x_tick = 12 + (i+1) * 10
    draw.line([(x_tick, 44), (x_tick, 46)], fill=(0, 0, 0, 255), width=1)  # X-axis ticks
    
    y_tick = 44 - (i+1) * 8
    draw.line([(10, y_tick), (12, y_tick)], fill=(0, 0, 0, 255), width=1)  # Y-axis ticks

# Generate a time trace with bursts
# Background noise
noise_level = 42
points = []
for x in range(12, 53):
    y = noise_level + random.uniform(-1, 1)
    points.append((x, y))

# Add bursts
burst_centers = [20, 35, 45]
burst_heights = [15, 25, 20]
burst_widths = [3, 4, 3]

for center, height, width in zip(burst_centers, burst_heights, burst_widths):
    for x in range(max(12, center - width), min(53, center + width + 1)):
        idx = x - 12
        if 0 <= idx < len(points):
            # Gaussian-like burst shape
            distance = abs(x - center)
            intensity = height * math.exp(-(distance**2) / (2 * (width/2)**2))
            points[idx] = (x, noise_level - intensity)

# Draw the time trace
for i in range(len(points) - 1):
    draw.line([points[i], points[i+1]], fill=(0, 0, 200, 255), width=1)

# Highlight the bursts with selection boxes
for center, height, width in zip(burst_centers, burst_heights, burst_widths):
    x1 = max(12, center - width)
    y1 = noise_level - height - 2
    x2 = min(52, center + width)
    y2 = noise_level + 2
    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0, 200), width=1)

# Add a small histogram in the corner to represent burst size distribution
hist_x1, hist_y1 = 40, 15
hist_x2, hist_y2 = 52, 30

# Histogram background
draw.rectangle([(hist_x1, hist_y1), (hist_x2, hist_y2)], 
               fill=(220, 220, 220, 150), 
               outline=(0, 0, 0, 200))

# Histogram bars
bar_widths = [2, 3, 4, 2]
bar_heights = [3, 6, 4, 2]
bar_spacing = 1
start_x = hist_x1 + 1

for width, height in zip(bar_widths, bar_heights):
    x1 = start_x
    y1 = hist_y2 - height
    x2 = x1 + width
    y2 = hist_y2
    draw.rectangle([(x1, y1), (x2, y2)], 
                   fill=(255, 100, 100, 200), 
                   outline=(0, 0, 0, 200))
    start_x += width + bar_spacing

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")