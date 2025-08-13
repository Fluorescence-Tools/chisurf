from PIL import Image, ImageDraw, ImageFont
import os

# Create a 64x64 image with a transparent background
size = 64
icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a soft rounded rectangle background
margin = 6
bg_rect = [margin, margin, size - margin, size - margin]
draw.rounded_rectangle(bg_rect, radius=10, fill=(245, 245, 245, 255), outline=(200, 200, 200, 255), width=1)

# Draw a 2x2 mosaic grid to represent image tiles
# Inner area
inner_margin = 12
inner = [inner_margin, inner_margin, size - inner_margin, size - inner_margin]
# Grid lines
x_mid = (inner[0] + inner[2]) // 2
y_mid = (inner[1] + inner[3]) // 2

# Tiles with subtle colors (like channels)
tiles = [
    ([inner[0], inner[1], x_mid - 1, y_mid - 1], (220, 240, 255, 255)),  # light blue
    ([x_mid + 1, inner[1], inner[2], y_mid - 1], (255, 230, 230, 255)),  # light red
    ([inner[0], y_mid + 1, x_mid - 1, inner[3]], (230, 255, 230, 255)),  # light green
    ([x_mid + 1, y_mid + 1, inner[2], inner[3]], (245, 245, 220, 255)),  # light yellow
]
for rect, color in tiles:
    draw.rectangle(rect, fill=color, outline=(180, 180, 180, 255))

# Draw grid divider lines
line_color = (160, 160, 160, 255)
draw.line([(x_mid, inner[1]), (x_mid, inner[3])], fill=line_color, width=1)
draw.line([(inner[0], y_mid), (inner[2], y_mid)], fill=line_color, width=1)

# Add a small camera/glyph in the corner to imply imaging
cam_w, cam_h = 20, 14
cam_x = size - cam_w - 10
cam_y = 10
# Body
draw.rounded_rectangle([cam_x, cam_y + 3, cam_x + cam_w, cam_y + 3 + cam_h], radius=3, fill=(80, 80, 80, 255))
# Top prism
draw.rectangle([cam_x + 4, cam_y, cam_x + 10, cam_y + 5], fill=(100, 100, 100, 255))
# Lens
draw.ellipse([cam_x + cam_w - 10, cam_y + 5, cam_x + cam_w - 2, cam_y + 13], fill=(200, 200, 200, 255))

# Add a small star in the bottom-left to hint “rating” feature
def draw_star(cx, cy, r_outer=6, r_inner=2.8, fill=(255, 200, 0, 255), outline=(150, 120, 0, 255)):
    import math
    points = []
    for i in range(10):
        angle = math.pi/2 + i * math.pi/5
        r = r_outer if i % 2 == 0 else r_inner
        x = cx + r * math.cos(angle)
        y = cy - r * math.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=fill, outline=outline)

draw_star(16, size - 16)

# Save the icon next to this script
out_path = os.path.join(os.path.dirname(__file__), 'icon.png')
icon.save(out_path)
print(f"Icon created at {out_path}")
