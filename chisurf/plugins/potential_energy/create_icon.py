from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a potential energy calculator icon
# Background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a simple molecular structure (a few atoms and bonds)
# Center atom
center_x, center_y = 32, 32
atom_radius = 6
draw.ellipse([(center_x - atom_radius, center_y - atom_radius), 
              (center_x + atom_radius, center_y + atom_radius)], 
             fill=(200, 0, 0, 200), outline=(0, 0, 0, 255))

# Surrounding atoms
atoms = [
    (center_x - 15, center_y - 15),  # top-left
    (center_x + 15, center_y - 15),  # top-right
    (center_x + 15, center_y + 15),  # bottom-right
    (center_x - 15, center_y + 15),  # bottom-left
]

# Draw bonds (lines) from center to each atom
for x, y in atoms:
    draw.line([(center_x, center_y), (x, y)], fill=(0, 0, 0, 200), width=2)

    # Draw the atom
    small_radius = 4
    draw.ellipse([(x - small_radius, y - small_radius), 
                  (x + small_radius, y + small_radius)], 
                 fill=(0, 100, 200, 200), outline=(0, 0, 0, 255))

# Draw energy values near some atoms
energy_points = [(center_x - 15, center_y - 15), (center_x + 15, center_y + 15)]
for i, (x, y) in enumerate(energy_points):
    # Draw energy indicator (small lightning bolt symbol)
    draw.polygon([(x+5, y-5), (x+8, y-2), (x+5, y), (x+8, y+3)], 
                fill=(255, 215, 0, 200), outline=(0, 0, 0, 255))

    # Draw small "E" letter to indicate energy
    draw.text((x+10, y-2), "E", fill=(0, 0, 0, 255))

# Draw a potential energy curve in the corner
curve_x = 12
curve_y = 15
curve_width = 15
curve_height = 10
draw.rectangle([(curve_x, curve_y), (curve_x + curve_width, curve_y + curve_height)], 
               fill=(255, 255, 255, 200), outline=(0, 0, 0, 255))

# Draw curve axes
draw.line([(curve_x, curve_y + curve_height), (curve_x + curve_width, curve_y + curve_height)], 
          fill=(0, 0, 0, 255), width=1)  # x-axis
draw.line([(curve_x, curve_y), (curve_x, curve_y + curve_height)], 
          fill=(0, 0, 0, 255), width=1)  # y-axis

# Draw potential energy curve (parabola-like)
curve_points = []
for i in range(8):
    x = curve_x + 2 + i * 1.5
    # Parabola: y = a(x-h)^2 + k
    y = curve_y + curve_height - 2 - 5 * ((i - 3.5) / 3.5) ** 2
    curve_points.append((x, y))

draw.line(curve_points, fill=(0, 200, 0, 255), width=1)

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")
