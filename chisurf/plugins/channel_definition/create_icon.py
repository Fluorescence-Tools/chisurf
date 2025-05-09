from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a representation of channel definition
# Draw detector channels as rectangles with different colors
channel_colors = [
    (200, 0, 0, 200),    # Red
    (0, 200, 0, 200),    # Green
    (0, 0, 200, 200),    # Blue
    (200, 200, 0, 200),  # Yellow
]

# Draw channel rectangles
channel_width = 8
channel_spacing = 4
start_x = 16

for i, color in enumerate(channel_colors):
    x = start_x + i * (channel_width + channel_spacing)
    # Draw channel rectangle
    draw.rectangle(
        [(x, 20), (x + channel_width, 44)], 
        fill=color, 
        outline=(0, 0, 0, 255)
    )
    
    # Draw channel number
    draw.text((x + 2, 30), str(i), fill=(255, 255, 255, 255))

# Draw connecting lines to a central hub (representing channel assignment)
hub_x, hub_y = 32, 50
hub_radius = 5
draw.ellipse(
    [(hub_x - hub_radius, hub_y - hub_radius), 
     (hub_x + hub_radius, hub_y + hub_radius)], 
    fill=(100, 100, 100, 200), 
    outline=(0, 0, 0, 255)
)

# Draw lines from channels to hub
for i in range(len(channel_colors)):
    x = start_x + i * (channel_width + channel_spacing) + channel_width // 2
    draw.line([(x, 44), (hub_x, hub_y - hub_radius)], fill=(0, 0, 0, 200), width=1)

# Draw a small settings gear in the corner to represent configuration
gear_x, gear_y = 48, 16
gear_radius = 6
# Draw gear circle
draw.ellipse(
    [(gear_x - gear_radius, gear_y - gear_radius), 
     (gear_x + gear_radius, gear_y + gear_radius)], 
    fill=(180, 180, 180, 200), 
    outline=(0, 0, 0, 255)
)
# Draw gear teeth
for i in range(8):
    angle = i * 45
    import math
    x1 = gear_x + (gear_radius - 1) * math.cos(math.radians(angle))
    y1 = gear_y + (gear_radius - 1) * math.sin(math.radians(angle))
    x2 = gear_x + (gear_radius + 2) * math.cos(math.radians(angle))
    y2 = gear_y + (gear_radius + 2) * math.sin(math.radians(angle))
    draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0, 255), width=1)

# Draw a small center in the gear
draw.ellipse(
    [(gear_x - 2, gear_y - 2), 
     (gear_x + 2, gear_y + 2)], 
    fill=(100, 100, 100, 200), 
    outline=(0, 0, 0, 255)
)

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")