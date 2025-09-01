from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a trajectory conversion icon
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a trajectory on the left side
traj_points = [(12, 40), (18, 30), (24, 45)]
draw.line(traj_points, fill=(200, 0, 0, 200), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in traj_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(200, 0, 0, 200), outline=(0, 0, 0, 255))

# Draw a conversion arrow in the middle
draw.line([(28, 32), (36, 32)], fill=(0, 200, 0, 200), width=2)
# Arrow head
draw.line([(34, 30), (36, 32)], fill=(0, 200, 0, 200), width=2)
draw.line([(34, 34), (36, 32)], fill=(0, 200, 0, 200), width=2)

# Draw a different representation on the right side (e.g., a PDB structure)
# Draw a simple protein-like structure
draw.line([(40, 25), (45, 35), (50, 25)], fill=(0, 0, 200, 200), width=2)
draw.line([(45, 35), (45, 45)], fill=(0, 0, 200, 200), width=2)
# Draw some atoms
draw.ellipse([(39, 24), (41, 26)], fill=(0, 0, 200, 200), outline=(0, 0, 0, 255))
draw.ellipse([(44, 34), (46, 36)], fill=(0, 0, 200, 200), outline=(0, 0, 0, 255))
draw.ellipse([(49, 24), (51, 26)], fill=(0, 0, 200, 200), outline=(0, 0, 0, 255))
draw.ellipse([(44, 44), (46, 46)], fill=(0, 0, 200, 200), outline=(0, 0, 0, 255))

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")