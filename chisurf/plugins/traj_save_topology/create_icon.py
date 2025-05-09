from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a trajectory save topology icon
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a trajectory
traj_points = [(12, 40), (20, 30), (28, 45), (36, 20), (44, 35), (52, 15)]
draw.line(traj_points, fill=(200, 0, 0, 200), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in traj_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(200, 0, 0, 200), outline=(0, 0, 0, 255))

# Draw a document/file icon to represent saving
file_x, file_y = 32, 32
file_width, file_height = 20, 25
# Main rectangle of the file
draw.rectangle([(file_x - file_width//2, file_y - file_height//2), 
                (file_x + file_width//2, file_y + file_height//2)], 
               fill=(255, 255, 255, 220), outline=(0, 0, 0, 255))

# Folded corner of the file
corner_size = 5
draw.polygon([(file_x + file_width//2 - corner_size, file_y - file_height//2),
              (file_x + file_width//2, file_y - file_height//2 + corner_size),
              (file_x + file_width//2, file_y - file_height//2)],
             fill=(200, 200, 200, 220), outline=(0, 0, 0, 255))

# Draw some lines to represent text in the file (topology data)
line_y_positions = [file_y - file_height//4, file_y, file_y + file_height//4]
for y_pos in line_y_positions:
    draw.line([(file_x - file_width//3, y_pos), (file_x + file_width//3, y_pos)], 
              fill=(0, 0, 200, 200), width=1)

# Draw a small arrow pointing from the trajectory to the file
draw.line([(40, 25), (35, 30)], fill=(0, 200, 0, 200), width=1)
# Arrow head
draw.line([(35, 30), (37, 28)], fill=(0, 200, 0, 200), width=1)
draw.line([(35, 30), (36, 32)], fill=(0, 200, 0, 200), width=1)

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")