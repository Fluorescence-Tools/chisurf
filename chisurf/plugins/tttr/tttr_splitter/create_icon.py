from PIL import Image, ImageDraw, ImageFont
import os
import math

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a large file at the top
large_file_color = (100, 150, 250, 230)  # Blue-ish
large_file_outline = (50, 100, 200, 255)
draw.rectangle([(16, 12), (48, 24)], fill=large_file_color, outline=large_file_outline, width=1)

# Add file icon details to the large file
draw.line([(16, 16), (22, 16), (22, 12)], fill=large_file_outline, width=1)
draw.text((28, 14), "PTU", fill=(0, 0, 0, 255))

# Draw an arrow pointing down
arrow_color = (80, 80, 80, 255)
arrow_start = (32, 26)
arrow_end = (32, 32)
draw.line([arrow_start, arrow_end], fill=arrow_color, width=2)
# Arrow head
draw.line([(28, 28), (32, 32), (36, 28)], fill=arrow_color, width=2)

# Draw smaller files at the bottom (split results)
small_file_positions = [(14, 36), (28, 36), (42, 36)]
small_file_colors = [
    (250, 150, 100, 230),  # Orange-ish
    (150, 250, 100, 230),  # Green-ish
    (250, 100, 250, 230)   # Purple-ish
]
small_file_outlines = [
    (200, 100, 50, 255),
    (100, 200, 50, 255),
    (200, 50, 200, 255)
]

for i, (pos, color, outline) in enumerate(zip(small_file_positions, small_file_colors, small_file_outlines)):
    # Draw small file rectangle
    draw.rectangle([(pos[0], pos[1]), (pos[0] + 8, pos[1] + 12)], 
                   fill=color, outline=outline, width=1)
    
    # Add file icon details to each small file
    draw.line([(pos[0], pos[1] + 3), (pos[0] + 3, pos[1] + 3), (pos[0] + 3, pos[1])], 
              fill=outline, width=1)
    
    # Add a small number to each file to indicate sequence
    draw.text((pos[0] + 2, pos[1] + 5), str(i+1), fill=(0, 0, 0, 255))

# Add dotted lines connecting the large file to the small files
for pos in small_file_positions:
    center_x = pos[0] + 4
    # Draw dotted line
    for y in range(32, 36, 2):
        draw.point((center_x, y), fill=arrow_color)

# Add a small label at the bottom
draw.text((18, 50), "PTU Splitter", fill=(0, 0, 0, 255))

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")