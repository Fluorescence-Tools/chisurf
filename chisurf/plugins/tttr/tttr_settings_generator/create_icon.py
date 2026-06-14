from PIL import Image, ImageDraw
import os
import math

def draw_icon_content(draw, icon_size=64):
    """
    Draw the content of the icon for TTTR Settings Generator.
    """
    # Background
    draw.rectangle([(0, 0), (icon_size, icon_size)], fill=(70, 130, 180, 255))

    # Draw a histogram-like shape
    bar_width = icon_size // 10
    max_height = icon_size * 0.8
    for i in range(8):
        height = int(max_height * (0.3 + 0.7 * (i % 3 + 1) / 3))
        x1 = i * bar_width + bar_width//2
        y1 = icon_size - height
        x2 = (i+1) * bar_width + bar_width//2
        y2 = icon_size
        draw.rectangle([(x1, y1), (x2, y2)], fill=(255, 255, 255, 200))

    # Draw a gear or settings symbol
    center = icon_size // 2
    radius = icon_size * 0.15
    # Simple gear shape
    points = []
    for i in range(8):
        angle = i * math.pi / 4
        if i % 2 == 0:
            r = radius
        else:
            r = radius * 0.7
        x = center + r * math.cos(angle)
        y = center + r * math.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=(255, 215, 0, 255))

def create_icon(icon_size=64):
    """
    Create an icon with the specified size.
    """
    # Create a transparent image
    icon = Image.new('RGBA', (icon_size, icon_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)

    # Draw the icon content
    draw_icon_content(draw, icon_size)

    return icon

if __name__ == "__main__":
    # Create the icon
    icon = create_icon()

    # Save the icon
    icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
    icon.save(icon_path)
    print(f"Icon created at {icon_path}")
