from PIL import Image, ImageDraw, ImageFont
import os
import math

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Colors
bg_color = (245, 245, 245, 230)
border_color = (0, 0, 0, 255)
db_color = (80, 130, 200, 240)
check_color = (60, 200, 100, 255)
text_color = (30, 30, 30, 255)
gear_color = (80, 80, 80, 255)

# Draw background
draw.rounded_rectangle([(6, 6), (58, 58)], radius=8, fill=bg_color, outline=border_color)

# Draw database: 3 stacked ellipses
db_cx, db_cy = 32, 22
db_w, db_h = 30, 10
for i in range(3):
    y = db_cy + i * 8
    draw.ellipse([(db_cx - db_w//2, y), (db_cx + db_w//2, y + db_h)], fill=db_color, outline=border_color)
    if i < 2:
        draw.rectangle([(db_cx - db_w//2, y + db_h//2), (db_cx + db_w//2, y + 8)], fill=db_color, outline=border_color)

# Draw checkmark on middle level
check_start = (db_cx - 7, db_cy + 12)
check_mid = (check_start[0] + 4, check_start[1] + 4)
check_end = (check_start[0] + 12, check_start[1] - 6)
draw.line([check_start, check_mid, check_end], fill=check_color, width=2)

# Draw gear in bottom right
gear_cx, gear_cy = 48, 48
gear_r = 6
tooth_r = gear_r + 3
teeth = 8
for i in range(teeth):
    angle = i * 2 * math.pi / teeth
    outer = (gear_cx + tooth_r * math.cos(angle), gear_cy + tooth_r * math.sin(angle))
    inner1 = (gear_cx + gear_r * math.cos(angle - math.pi / 16), gear_cy + gear_r * math.sin(angle - math.pi / 16))
    inner2 = (gear_cx + gear_r * math.cos(angle + math.pi / 16), gear_cy + gear_r * math.sin(angle + math.pi / 16))
    draw.polygon([inner1, outer, inner2], fill=gear_color, outline=border_color)

# Draw gear center
draw.ellipse([(gear_cx - 3, gear_cy - 3), (gear_cx + 3, gear_cy + 3)], fill=gear_color, outline=border_color)

# Draw "M" in top left
try:
    font = ImageFont.truetype("arial.ttf", 12)
except IOError:
    font = ImageFont.load_default()
draw.text((10, 8), "M", font=font, fill=text_color)

# Save icon
icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.png')
icon.save(icon_path)
print(f"Icon created at {icon_path}")
