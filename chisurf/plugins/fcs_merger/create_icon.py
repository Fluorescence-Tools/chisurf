from PIL import Image, ImageDraw
import os
import math

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a representation of merging correlation curves
# Create axes for the correlation plot
axes_x, axes_y = 16, 40
axes_width, axes_height = 32, 24

# Draw x and y axes
draw.line([(axes_x, axes_y), (axes_x + axes_width, axes_y)], fill=(0, 0, 0, 200), width=2)  # x-axis
draw.line([(axes_x, axes_y), (axes_x, axes_y - axes_height)], fill=(0, 0, 0, 200), width=2)  # y-axis

# Draw multiple correlation curves with different colors
curve_colors = [
    (200, 0, 0, 150),    # Red
    (0, 200, 0, 150),    # Green
    (0, 0, 200, 150),    # Blue
]

# Draw multiple correlation curves (exponential decays with noise)
for curve_idx, color in enumerate(curve_colors):
    curve_points = []
    # Add some random offset to make curves look different
    offset = (curve_idx - 1) * 2
    
    for i in range(30):
        x = axes_x + i
        if i == 0:
            y = axes_y - axes_height + 2 + offset  # Start high with offset
        else:
            # Exponential decay with some noise
            decay = math.exp(-i * (0.13 + curve_idx * 0.02))
            noise = (i % 3) * 0.5 * (curve_idx + 1)
            y = axes_y - (axes_height - 2 - offset) * decay + noise
            # Keep within axes
            y = max(axes_y - axes_height, min(axes_y, y))
        curve_points.append((x, y))
    
    # Draw the correlation curve
    draw.line(curve_points, fill=color, width=1)

# Draw the merged/averaged curve (thicker line)
merged_points = []
for i in range(30):
    x = axes_x + i
    if i == 0:
        y = axes_y - axes_height + 2  # Start high
    else:
        # Smoother exponential decay (average of the others)
        decay = math.exp(-i * 0.14)
        y = axes_y - (axes_height - 2) * decay
    merged_points.append((x, y))

# Draw the merged correlation curve (thicker)
draw.line(merged_points, fill=(0, 0, 0, 200), width=2)

# Draw a merge symbol at the top
merge_x, merge_y = 32, 16
arrow_size = 6

# Draw arrows pointing to center
for angle in [150, 180, 210]:
    rad = math.radians(angle)
    x1 = merge_x + arrow_size * math.cos(rad)
    y1 = merge_y + arrow_size * math.sin(rad)
    x2 = merge_x + (arrow_size - 3) * math.cos(rad)
    y2 = merge_y + (arrow_size - 3) * math.sin(rad)
    draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0, 200), width=2)

# Draw center circle
draw.ellipse([(merge_x - 3, merge_y - 3), (merge_x + 3, merge_y + 3)], 
             fill=(100, 100, 100, 200), outline=(0, 0, 0, 255))

# Draw a downward arrow from the merge circle to the graph
draw.line([(merge_x, merge_y + 3), (merge_x, axes_y - axes_height - 2)], 
          fill=(0, 0, 0, 200), width=1)
# Draw arrowhead
draw.polygon([(merge_x - 2, axes_y - axes_height - 2), 
              (merge_x + 2, axes_y - axes_height - 2),
              (merge_x, axes_y - axes_height + 1)], 
             fill=(0, 0, 0, 200))

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")