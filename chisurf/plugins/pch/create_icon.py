from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a histogram-like shape
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Histogram bars
bar_colors = [(255, 100, 100, 200), (100, 255, 100, 200), (100, 100, 255, 200)]
bar_heights = [30, 20, 15, 25, 10]
bar_width = 8
bar_spacing = 2
start_x = 10

for i, height in enumerate(bar_heights):
    color = bar_colors[i % len(bar_colors)]
    x1 = start_x + i * (bar_width + bar_spacing)
    y1 = 56 - height
    x2 = x1 + bar_width
    y2 = 56
    draw.rectangle([(x1, y1), (x2, y2)], fill=color, outline=(0, 0, 0, 255))

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")