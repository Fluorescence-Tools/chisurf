from PIL import Image, ImageDraw
import os
import math

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a trajectory rotate/translate icon
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw original trajectory
traj_points = [(15, 40), (20, 35), (25, 40), (30, 30), (35, 35)]
draw.line(traj_points, fill=(200, 0, 0, 200), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in traj_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(200, 0, 0, 200), outline=(0, 0, 0, 255))

# Draw rotation arrow (circular arrow)
center_x, center_y = 25, 25
radius = 8
start_angle = 0
end_angle = 270
# Draw arc
for angle in range(start_angle, end_angle + 1, 5):
    rad = math.radians(angle)
    x = center_x + radius * math.cos(rad)
    y = center_y + radius * math.sin(rad)
    if angle == start_angle:
        start_point = (x, y)
    else:
        end_point = (x, y)
        draw.line([start_point, end_point], fill=(0, 0, 200, 200), width=1)
        start_point = end_point

# Draw arrowhead for rotation
arrow_angle = math.radians(end_angle + 10)
arrow_x = center_x + radius * math.cos(arrow_angle)
arrow_y = center_y + radius * math.sin(arrow_angle)
draw.line([end_point, (arrow_x, arrow_y)], fill=(0, 0, 200, 200), width=1)

# Draw translation arrow
draw.line([(40, 35), (50, 25)], fill=(0, 200, 0, 200), width=2)
# Arrow head
draw.line([(47, 25), (50, 25)], fill=(0, 200, 0, 200), width=2)
draw.line([(50, 25), (50, 28)], fill=(0, 200, 0, 200), width=2)

# Draw transformed trajectory (rotated and translated)
transformed_points = [(35, 25), (40, 20), (45, 25), (50, 15), (55, 20)]
draw.line(transformed_points, fill=(0, 0, 200, 200), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in transformed_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(0, 0, 200, 200), outline=(0, 0, 0, 255))

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")