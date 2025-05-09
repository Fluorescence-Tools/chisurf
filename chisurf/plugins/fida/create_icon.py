from PIL import Image, ImageDraw
import os
import random
import math

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a representation of Fluorescence Intensity Distribution Analysis (FIDA)
# Create a histogram-like visualization
hist_x, hist_y = 16, 40
hist_width, hist_height = 32, 24

# Draw histogram axes
draw.line([(hist_x, hist_y), (hist_x + hist_width, hist_y)], fill=(0, 0, 0, 200), width=2)  # x-axis
draw.line([(hist_x, hist_y), (hist_x, hist_y - hist_height)], fill=(0, 0, 0, 200), width=2)  # y-axis

# Draw axis labels
draw.text((hist_x + hist_width - 8, hist_y + 2), "k", fill=(0, 0, 0, 255))  # photon counts
draw.text((hist_x - 8, hist_y - hist_height + 2), "P", fill=(0, 0, 0, 255))  # probability

# Draw a photon count histogram (Poisson-like distribution)
bar_width = 2
num_bars = 12
max_height = hist_height - 4

# Use a simpler approach for the histogram
bar_heights = [5, 10, 15, 18, 20, 16, 12, 8, 6, 4, 3, 2]  # Predefined heights

# Draw histogram bars
for i in range(num_bars):
    x = hist_x + i * bar_width + 2
    height = bar_heights[i] if i < len(bar_heights) else 0

    # Ensure height is within bounds
    height = min(height, max_height)

    # Draw bar (ensure y1 > y0)
    if height > 0:
        y_top = hist_y - height
        y_bottom = hist_y
        draw.rectangle(
            [(x, y_top), (x + bar_width - 1, y_bottom)],
            fill=(200, 0, 0, 150),
            outline=(100, 0, 0, 200)
        )

# Draw a fitted model curve (smooth curve through the histogram)
curve_points = []
lambda_val = 5  # Define lambda_val for the curve

# Use a simpler approach for the curve - a smooth bell curve
for i in range(hist_width):
    x = hist_x + i
    # Simple bell curve centered at x=16
    center = 16
    width = 10
    height = max_height * 0.8 * math.exp(-((i - center) ** 2) / (2 * width ** 2))
    y = hist_y - height
    curve_points.append((x, y))

# Draw the model curve
draw.line(curve_points, fill=(0, 0, 200, 200), width=2)

# Draw a small fluorescence intensity trace in the corner
trace_x, trace_y = 14, 14
trace_width, trace_height = 10, 8

# Draw a small box for the trace
draw.rectangle(
    [(trace_x, trace_y), (trace_x + trace_width, trace_y + trace_height)],
    fill=(255, 255, 255, 150),
    outline=(0, 0, 0, 255)
)

# Draw fluorescence intensity spikes (representing single molecules)
for i in range(trace_width):
    x = trace_x + i
    # Random heights to simulate fluorescence bursts
    if random.random() < 0.4:  # 40% chance of a burst
        height = random.randint(2, trace_height - 1)
        draw.line(
            [(x, trace_y + trace_height), (x, trace_y + trace_height - height)],
            fill=(0, 200, 0, 200),
            width=1
        )

# Draw a small molecule icon to represent single-molecule analysis
mol_x, mol_y = 48, 16
mol_radius = 4

# Draw central atom
draw.ellipse(
    [(mol_x - mol_radius, mol_y - mol_radius), 
     (mol_x + mol_radius, mol_y + mol_radius)],
    fill=(0, 150, 0, 200),
    outline=(0, 0, 0, 255)
)

# Draw "glow" to represent fluorescence
for r in range(2, 5):
    draw.ellipse(
        [(mol_x - mol_radius - r, mol_y - mol_radius - r),
         (mol_x + mol_radius + r, mol_y + mol_radius + r)],
        outline=(0, 255, 0, 100 - r * 20)
    )

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")
