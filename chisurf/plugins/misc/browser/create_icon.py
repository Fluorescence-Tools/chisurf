from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a browser window
# Background rectangle (browser window)
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Address bar at the top
address_bar_y1, address_bar_y2 = 12, 18
draw.rectangle([(12, address_bar_y1), (52, address_bar_y2)], 
               fill=(255, 255, 255, 255), 
               outline=(180, 180, 180, 255))

# Navigation buttons (back, forward, refresh)
button_size = 4
button_y = 15
button_spacing = 6

# Back button
draw.polygon([(12, button_y), (15, button_y - 2), (15, button_y + 2)], 
             fill=(100, 100, 200, 200))

# Forward button
draw.polygon([(12 + button_spacing, button_y), (12 + button_spacing - 3, button_y - 2), 
              (12 + button_spacing - 3, button_y + 2)], 
             fill=(100, 100, 200, 200))

# Refresh button
refresh_x = 12 + button_spacing * 2
draw.arc([(refresh_x - 2, button_y - 2), (refresh_x + 2, button_y + 2)], 
         0, 270, fill=(100, 100, 200, 200))

# Draw a small "http://" in the address bar
draw.line([(24, button_y), (45, button_y)], fill=(180, 180, 180, 200), width=1)

# Content area
content_y1 = address_bar_y2 + 2
content_y2 = 52

# Draw a simplified webpage in the content area
# Header
draw.rectangle([(12, content_y1), (52, content_y1 + 6)], 
               fill=(100, 150, 250, 150))

# Content blocks
block_heights = [4, 3, 5, 3]
block_y = content_y1 + 8

for height in block_heights:
    draw.rectangle([(15, block_y), (49, block_y + height)], 
                   fill=(220, 220, 220, 150))
    block_y += height + 2

# Draw a small globe icon to represent the web
globe_x, globe_y = 32, 38
globe_radius = 6
draw.ellipse([(globe_x - globe_radius, globe_y - globe_radius), 
              (globe_x + globe_radius, globe_y + globe_radius)], 
             outline=(0, 120, 200, 200))

# Draw latitude lines on the globe
for offset in [-3, 0, 3]:
    draw.line([(globe_x - 5, globe_y + offset), (globe_x + 5, globe_y + offset)], 
              fill=(0, 120, 200, 200), width=1)

# Draw longitude lines on the globe
draw.arc([(globe_x - 3, globe_y - 6), (globe_x + 3, globe_y + 6)], 
         0, 360, fill=(0, 120, 200, 200))

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")