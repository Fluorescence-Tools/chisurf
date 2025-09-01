from PIL import Image, ImageDraw
import os
import math

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a plugin manager icon
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a settings gear
center_x, center_y = 32, 32
outer_radius = 15
inner_radius = 10
tooth_length = 5
num_teeth = 8

# Draw the gear teeth
for i in range(num_teeth):
    angle = i * (360 / num_teeth)
    # Convert angle to radians
    angle_rad = angle * 3.14159 / 180

    # Calculate outer point of tooth
    outer_x = center_x + (outer_radius + tooth_length) * math.cos(angle_rad)
    outer_y = center_y + (outer_radius + tooth_length) * math.sin(angle_rad)

    # Calculate inner points of tooth
    inner1_angle = (angle - 15) * 3.14159 / 180
    inner1_x = center_x + outer_radius * math.cos(inner1_angle)
    inner1_y = center_y + outer_radius * math.sin(inner1_angle)

    inner2_angle = (angle + 15) * 3.14159 / 180
    inner2_x = center_x + outer_radius * math.cos(inner2_angle)
    inner2_y = center_y + outer_radius * math.sin(inner2_angle)

    # Draw tooth
    draw.polygon([(inner1_x, inner1_y), (outer_x, outer_y), (inner2_x, inner2_y)], 
                 fill=(100, 100, 200, 200), outline=(0, 0, 0, 255))

# Draw the gear center
draw.ellipse([(center_x - inner_radius, center_y - inner_radius), 
              (center_x + inner_radius, center_y + inner_radius)], 
             fill=(100, 100, 200, 200), outline=(0, 0, 0, 255))

# Draw small plugin icons around the gear
plugin_radius = 8
plugin_distance = 20

# Plugin positions around the gear
plugin_positions = [
    (center_x, center_y - plugin_distance),  # Top
    (center_x + plugin_distance, center_y),  # Right
    (center_x, center_y + plugin_distance),  # Bottom
    (center_x - plugin_distance, center_y),  # Left
]

# Draw plugin icons (small squares with checkmarks or X marks)
for i, (x, y) in enumerate(plugin_positions):
    # Draw plugin square
    draw.rectangle([(x - plugin_radius, y - plugin_radius), 
                    (x + plugin_radius, y + plugin_radius)], 
                   fill=(255, 255, 255, 200), outline=(0, 0, 0, 255))

    # Alternate between checkmark and X
    if i % 2 == 0:
        # Draw checkmark (enabled plugin)
        draw.line([(x - 4, y), (x - 1, y + 3)], fill=(0, 200, 0, 255), width=2)
        draw.line([(x - 1, y + 3), (x + 4, y - 3)], fill=(0, 200, 0, 255), width=2)
    else:
        # Draw X (disabled plugin)
        draw.line([(x - 3, y - 3), (x + 3, y + 3)], fill=(200, 0, 0, 255), width=2)
        draw.line([(x - 3, y + 3), (x + 3, y - 3)], fill=(200, 0, 0, 255), width=2)

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")
