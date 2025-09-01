from PIL import Image, ImageDraw
import os
import math
import random

# Template for creating plugin icons
# Instructions:
# 1. Modify the draw_icon_content function to create an icon that represents your plugin's functionality
# 2. Run the script to generate the icon.png file

def draw_icon_content(draw, icon_size=64):
    """
    Draw the content of the icon.
    
    Args:
        draw: ImageDraw object to draw on
        icon_size: Size of the icon (default: 64x64)
    """
    # Draw a background rectangle
    margin = int(icon_size * 0.125)  # 8 pixels for 64x64 icon
    draw.rectangle([(margin, margin), (icon_size - margin, icon_size - margin)], 
                   fill=(240, 240, 240, 200), 
                   outline=(0, 0, 0, 255))
    
    # Add your custom drawing code here
    # Example: Draw a simple placeholder
    center_x, center_y = icon_size // 2, icon_size // 2
    radius = int(icon_size * 0.25)  # 16 pixels for 64x64 icon
    draw.ellipse([(center_x - radius, center_y - radius), 
                  (center_x + radius, center_y + radius)], 
                 fill=(100, 100, 200, 200), 
                 outline=(0, 0, 0, 255))
    
    # Draw text to indicate this is a custom plugin
    draw.text((center_x - 5, center_y - 10), "P", fill=(255, 255, 255, 255))

def create_icon(icon_size=64):
    """
    Create an icon with the specified size.
    
    Args:
        icon_size: Size of the icon (default: 64x64)
    
    Returns:
        PIL.Image: The created icon
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
    icon.save('icon.png')
    print(f"Icon created at {os.path.abspath('icon.png')}")
    
    # Uncomment to display the icon (requires matplotlib)
    # import matplotlib.pyplot as plt
    # plt.imshow(icon)
    # plt.axis('off')
    # plt.show()