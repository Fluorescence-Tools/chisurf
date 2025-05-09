from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a trajectory join icon
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw first trajectory (top)
traj1_points = [(12, 20), (20, 15), (28, 25), (36, 18)]
draw.line(traj1_points, fill=(200, 0, 0, 200), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in traj1_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(200, 0, 0, 200), outline=(0, 0, 0, 255))

# Draw second trajectory (bottom)
traj2_points = [(12, 40), (20, 45), (28, 35), (36, 42)]
draw.line(traj2_points, fill=(0, 0, 200, 200), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in traj2_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(0, 0, 200, 200), outline=(0, 0, 0, 255))

# Draw joining arrow
draw.line([(40, 30), (48, 30)], fill=(0, 200, 0, 200), width=2)
# Arrow head
draw.line([(46, 28), (48, 30)], fill=(0, 200, 0, 200), width=2)
draw.line([(46, 32), (48, 30)], fill=(0, 200, 0, 200), width=2)

# Draw joined trajectory (right)
joined_points = [(50, 20), (52, 25), (54, 30), (52, 35), (50, 40)]
draw.line(joined_points, fill=(100, 100, 0, 200), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in joined_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(100, 100, 0, 200), outline=(0, 0, 0, 255))

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")