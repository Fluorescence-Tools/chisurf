from PIL import Image, ImageDraw, ImageFont
import os

# Create a 64x64 transparent icon with a chat bubble and "Chi" label
size = 64
icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Bubble background
pad = 6
bubble_rect = [pad, pad, size - pad, size - pad - 10]
draw.rounded_rectangle(bubble_rect, radius=12, fill=(230, 240, 255, 255), outline=(60, 120, 200, 255), width=2)

# Bubble tail
tail = [(size//2 - 6, size - pad - 12), (size//2 + 2, size - pad - 12), (size//2 + 6, size - pad - 2)]
draw.polygon(tail, fill=(230, 240, 255, 255), outline=(60, 120, 200, 255))

# Try to put "Chi" text in the middle
text = "Chi"
# Use a basic font; PIL will fallback if default not found
try:
    font = ImageFont.truetype("arial.ttf", 20)
except Exception:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

if font is not None:
    # Pillow 10+ removed ImageDraw.textsize; prefer textbbox if available, then font.getsize
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        try:
            tw, th = font.getsize(text)  # older API
        except Exception:
            tw, th = 24, 12
    tx = (size - tw) // 2
    ty = (size - 10 - th) // 2 + 4
    draw.text((tx, ty), text, fill=(40, 80, 160, 255), font=font)

# Save next to this file as icon.png
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icon.png')
icon.save(out_path)
print(f"ChiChat icon created at {out_path}")
