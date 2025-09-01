from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a trajectory energy calculator icon
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a trajectory
traj_points = [(12, 40), (20, 30), (28, 45), (36, 20), (44, 35), (52, 15)]
draw.line(traj_points, fill=(200, 0, 0, 200), width=2)

# Draw dots at each point to represent trajectory frames
for x, y in traj_points:
    draw.ellipse([(x-2, y-2), (x+2, y+2)], fill=(200, 0, 0, 200), outline=(0, 0, 0, 255))

# Draw energy values above some points (as small bars of different heights)
energy_points = [(20, 30), (36, 20), (52, 15)]
for i, (x, y) in enumerate(energy_points):
    # Draw energy bar
    height = 10 + i * 5  # Varying heights for different energy levels
    draw.rectangle([(x-3, y-height), (x+3, y)], fill=(0, 200, 0, 150), outline=(0, 0, 0, 255))
    
    # Draw small "E" letter to indicate energy
    draw.text((x-2, y-height-10), "E", fill=(0, 0, 0, 255))

# Draw a small graph in the corner to represent energy calculation
graph_x = 12
graph_y = 15
graph_width = 15
graph_height = 10
draw.rectangle([(graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height)], 
               fill=(255, 255, 255, 200), outline=(0, 0, 0, 255))

# Draw graph axes
draw.line([(graph_x, graph_y + graph_height), (graph_x + graph_width, graph_y + graph_height)], 
          fill=(0, 0, 0, 255), width=1)  # x-axis
draw.line([(graph_x, graph_y), (graph_x, graph_y + graph_height)], 
          fill=(0, 0, 0, 255), width=1)  # y-axis

# Draw energy curve on graph
curve_points = [(graph_x + 2, graph_y + 8), 
                (graph_x + 5, graph_y + 5),
                (graph_x + 8, graph_y + 7),
                (graph_x + 12, graph_y + 3)]
draw.line(curve_points, fill=(0, 0, 200, 255), width=1)

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")