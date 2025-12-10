from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw multiple stacked documents to represent batch processing
num_docs = 4
doc_width, doc_height = 30, 35
start_x, start_y = 17, 15

# Draw documents from back to front
for i in range(num_docs):
    offset = i * 3  # Offset each document slightly
    x1 = start_x + offset
    y1 = start_y + offset
    x2 = x1 + doc_width
    y2 = y1 + doc_height
    
    # Document background
    draw.rectangle([(x1, y1), (x2, y2)], 
                   fill=(255, 255, 255, 230), 
                   outline=(0, 0, 0, 255))
    
    # Document lines (representing text)
    line_y_positions = [y1 + 5, y1 + 10, y1 + 15, y1 + 20, y1 + 25]
    for line_y in line_y_positions:
        line_length = doc_width - 6
        draw.line([(x1 + 3, line_y), (x1 + 3 + line_length, line_y)], 
                  fill=(100, 100, 100, 150), width=1)

# Draw a progress bar at the bottom
progress_x1, progress_y1 = 15, 52
progress_x2, progress_y2 = 49, 56
progress_fill_x2 = progress_x1 + (progress_x2 - progress_x1) * 0.7  # 70% complete

# Progress bar background
draw.rectangle([(progress_x1, progress_y1), (progress_x2, progress_y2)], 
               fill=(220, 220, 220, 200), 
               outline=(0, 0, 0, 255))

# Progress bar fill
draw.rectangle([(progress_x1, progress_y1), (progress_fill_x2, progress_y2)], 
               fill=(100, 200, 100, 200), 
               outline=None)

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")