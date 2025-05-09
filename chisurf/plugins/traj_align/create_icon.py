from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a trajectory alignment icon
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw two trajectories - one original (red) and one aligned (blue)
original_points = [(12, 40), (20, 30), (28, 45), (36, 20), (44, 35), (52, 15)]
aligned_points = [(12, 30), (20, 20), (28, 35), (36, 10), (44, 25), (52, 5)]

# Draw the original trajectory
draw.line(original_points, fill=(200, 0, 0, 200), width=2)
# Draw the aligned trajectory
draw.line(aligned_points, fill=(0, 0, 200, 200), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in original_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(200, 0, 0, 200), outline=(0, 0, 0, 255))
for x, y in aligned_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(0, 0, 200, 200), outline=(0, 0, 0, 255))

# Draw alignment arrows between some corresponding points
draw.line([(20, 30), (20, 20)], fill=(0, 200, 0, 200), width=1)
draw.line([(36, 20), (36, 10)], fill=(0, 200, 0, 200), width=1)
draw.line([(52, 15), (52, 5)], fill=(0, 200, 0, 200), width=1)

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")