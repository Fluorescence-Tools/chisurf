from PIL import Image, ImageDraw, ImageFont
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a JSON file icon
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw curly braces to represent JSON
draw.line([(16, 16), (16, 48), (24, 48)], fill=(0, 0, 0, 255), width=2)  # Left brace
draw.line([(48, 16), (48, 48), (40, 48)], fill=(0, 0, 0, 255), width=2)  # Right brace

# Draw some "key-value" lines to represent JSON content
draw.line([(20, 24), (44, 24)], fill=(0, 100, 200, 255), width=2)  # Line 1
draw.line([(20, 32), (44, 32)], fill=(0, 100, 200, 255), width=2)  # Line 2
draw.line([(20, 40), (44, 40)], fill=(0, 100, 200, 255), width=2)  # Line 3

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")