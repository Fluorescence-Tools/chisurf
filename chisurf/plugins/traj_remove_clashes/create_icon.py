from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a trajectory remove clashed frames icon
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a trajectory with some clashed frames
traj_points = [(12, 40), (20, 30), (28, 45), (36, 20), (44, 35), (52, 15)]
draw.line(traj_points, fill=(200, 0, 0, 200), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in traj_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(200, 0, 0, 200), outline=(0, 0, 0, 255))

# Mark clashed frames with X
clashed_points = [(28, 45), (44, 35)]
for x, y in clashed_points:
    # Draw X over the clashed frame
    draw.line([(x-3, y-3), (x+3, y+3)], fill=(0, 0, 0, 255), width=1)
    draw.line([(x-3, y+3), (x+3, y-3)], fill=(0, 0, 0, 255), width=1)
    
    # Draw a red circle around the clashed frame
    draw.ellipse([(x-4, y-4), (x+4, y+4)], outline=(255, 0, 0, 255), width=1)

# Draw a "cleaned" trajectory below (without the clashed frames)
clean_points = [(12, 50), (20, 40), (36, 30), (52, 25)]
draw.line(clean_points, fill=(0, 200, 0, 200), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in clean_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(0, 200, 0, 200), outline=(0, 0, 0, 255))

# Draw an arrow from the original to the cleaned trajectory
draw.line([(32, 35), (32, 45)], fill=(0, 0, 200, 200), width=1)
# Arrow head
draw.line([(30, 43), (32, 45)], fill=(0, 0, 200, 200), width=1)
draw.line([(34, 43), (32, 45)], fill=(0, 0, 200, 200), width=1)

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")