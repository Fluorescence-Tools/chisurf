from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a FRET trajectory icon
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a trajectory line
points = [(12, 40), (20, 30), (28, 45), (36, 20), (44, 35), (52, 15)]
draw.line(points, fill=(0, 0, 200, 255), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in points:
    draw.ellipse([(x-3, y-3), (x+3, y+3)], fill=(200, 0, 0, 200), outline=(0, 0, 0, 255))

# Draw FRET arrows between some points
draw.line([(20, 30), (28, 45)], fill=(0, 200, 0, 200), width=2)
draw.line([(36, 20), (44, 35)], fill=(0, 200, 0, 200), width=2)

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")