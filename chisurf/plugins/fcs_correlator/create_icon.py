from PIL import Image, ImageDraw
import os
import math

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a representation of a correlation function
# Create axes for the correlation plot
axes_x, axes_y = 16, 32
axes_width, axes_height = 32, 20

# Draw x and y axes
draw.line([(axes_x, axes_y), (axes_x + axes_width, axes_y)], fill=(0, 0, 0, 200), width=2)  # x-axis
draw.line([(axes_x, axes_y), (axes_x, axes_y - axes_height)], fill=(0, 0, 0, 200), width=2)  # y-axis

# Draw axis labels
draw.text((axes_x + axes_width - 8, axes_y + 2), "τ", fill=(0, 0, 0, 255))  # tau (time)
draw.text((axes_x - 8, axes_y - axes_height + 2), "G", fill=(0, 0, 0, 255))  # G (correlation)

# Draw a typical correlation curve (exponential decay)
curve_points = []
for i in range(30):
    x = axes_x + i
    if i == 0:
        y = axes_y - axes_height + 2  # Start high
    else:
        # Exponential decay: y = a * exp(-b * x) + c
        decay = math.exp(-i * 0.15)
        y = axes_y - (axes_height - 2) * decay
    curve_points.append((x, y))

# Draw the correlation curve
draw.line(curve_points, fill=(200, 0, 0, 200), width=2)

# Draw a second correlation curve (slightly different parameters)
curve2_points = []
for i in range(30):
    x = axes_x + i
    if i == 0:
        y = axes_y - axes_height + 4  # Start slightly lower
    else:
        # Different decay rate
        decay = math.exp(-i * 0.12)
        y = axes_y - (axes_height - 4) * decay
    curve2_points.append((x, y))

# Draw the second correlation curve
draw.line(curve2_points, fill=(0, 0, 200, 200), width=2)

# Draw a small photon trace in the corner to represent TTTR data
trace_x, trace_y = 14, 14
trace_width, trace_height = 10, 8

# Draw a small box for the trace
draw.rectangle([(trace_x, trace_y), (trace_x + trace_width, trace_y + trace_height)], 
               fill=(255, 255, 255, 150), outline=(0, 0, 0, 255))

# Draw photon spikes
for i in range(1, 10):
    x = trace_x + i
    height = 2 + (i % 3) * 2  # Varying heights
    if i % 2 == 0:  # Only draw some spikes
        draw.line([(x, trace_y + trace_height), (x, trace_y + trace_height - height)], 
                  fill=(0, 200, 0, 200), width=1)

# Draw a small "merge" symbol to represent correlation merging
merge_x, merge_y = 48, 16
arrow_size = 4

# Draw arrows pointing to center
for angle in [45, 135, 225, 315]:
    rad = math.radians(angle)
    x1 = merge_x + arrow_size * math.cos(rad)
    y1 = merge_y + arrow_size * math.sin(rad)
    x2 = merge_x + (arrow_size - 2) * math.cos(rad)
    y2 = merge_y + (arrow_size - 2) * math.sin(rad)
    draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0, 200), width=1)

# Draw center circle
draw.ellipse([(merge_x - 2, merge_y - 2), (merge_x + 2, merge_y + 2)], 
             fill=(0, 100, 200, 200), outline=(0, 0, 0, 255))

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")