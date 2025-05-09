from PIL import Image, ImageDraw
import os
import random

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a representation of a CLSM (Confocal Laser Scanning Microscopy) image
# Create a grid for the microscopy image
grid_size = 24
cell_size = 2
grid_x, grid_y = 20, 20

# Draw a microscopy image as a grid of colored pixels
# Use a color gradient to simulate a fluorescence image
for i in range(grid_size):
    for j in range(grid_size):
        # Calculate position
        x = grid_x + i * cell_size
        y = grid_y + j * cell_size
        
        # Skip if outside the main rectangle
        if x < 10 or x > 54 or y < 10 or y > 54:
            continue
            
        # Create a pattern that looks like a cell or tissue
        # Distance from center
        center_i, center_j = grid_size // 2, grid_size // 2
        dist = ((i - center_i) ** 2 + (j - center_j) ** 2) ** 0.5
        
        # Create a cell-like pattern with brighter center
        if dist < 8:
            # Cell interior - brighter
            intensity = max(0, 255 - int(dist * 20))
            color = (0, intensity, 0, 200)  # Green fluorescence
        elif dist < 10:
            # Cell membrane - different color
            intensity = max(0, 200 - int((dist-8) * 50))
            color = (intensity, 0, intensity, 200)  # Purple membrane
        else:
            # Background - darker
            intensity = max(0, 100 - int(dist * 5))
            color = (0, 0, intensity, 100)  # Blue background
            
        # Add some random noise
        noise = random.randint(-20, 20)
        r, g, b, a = color
        r = max(0, min(255, r + noise))
        g = max(0, min(255, g + noise))
        b = max(0, min(255, b + noise))
        
        draw.rectangle([(x, y), (x + cell_size, y + cell_size)], 
                      fill=(r, g, b, a))

# Draw a selection tool overlay
# Draw a rectangular selection
selection_x, selection_y = 32, 32
selection_size = 16
draw.rectangle(
    [(selection_x - selection_size//2, selection_y - selection_size//2), 
     (selection_x + selection_size//2, selection_y + selection_size//2)], 
    outline=(255, 0, 0, 255),  # Red outline
    width=1
)

# Draw selection handles
handle_size = 2
for dx in [-1, 1]:
    for dy in [-1, 1]:
        handle_x = selection_x + dx * selection_size//2
        handle_y = selection_y + dy * selection_size//2
        draw.rectangle(
            [(handle_x - handle_size, handle_y - handle_size),
             (handle_x + handle_size, handle_y + handle_size)],
            fill=(255, 0, 0, 255),  # Red fill
            outline=(0, 0, 0, 255)   # Black outline
        )

# Draw a small microscope icon in the corner
scope_x, scope_y = 14, 14
# Draw microscope base
draw.rectangle([(scope_x - 4, scope_y + 4), (scope_x + 4, scope_y + 6)], 
              fill=(100, 100, 100, 200), outline=(0, 0, 0, 255))
# Draw microscope body
draw.rectangle([(scope_x - 2, scope_y - 2), (scope_x + 2, scope_y + 4)], 
              fill=(150, 150, 150, 200), outline=(0, 0, 0, 255))
# Draw microscope objective
draw.ellipse([(scope_x - 3, scope_y - 6), (scope_x + 3, scope_y)], 
            fill=(200, 200, 200, 200), outline=(0, 0, 0, 255))

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")