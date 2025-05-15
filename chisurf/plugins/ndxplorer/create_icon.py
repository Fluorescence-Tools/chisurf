import os
import sys
from PIL import Image, ImageDraw

def create_ndxplorer_icon(png_path, width=128, height=128):
    """
    Create a simple icon for the ndXplorer plugin.
    
    Parameters:
    -----------
    png_path : str
        Path where the PNG file will be saved
    width : int
        Width of the output PNG image
    height : int
        Height of the output PNG image
    """
    try:
        # Create a new image with a transparent background
        image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        
        # Define colors
        red = (231, 76, 60, 255)       # Red
        orange = (243, 156, 18, 255)   # Orange
        yellow = (241, 196, 15, 255)   # Yellow
        black = (0, 0, 0, 255)         # Black
        
        # Draw a sun-like shape (inspired by the original SVG)
        center_x, center_y = width // 2, height // 2
        outer_radius = min(width, height) // 2 - 10
        middle_radius = outer_radius * 0.75
        inner_radius = outer_radius * 0.5
        
        # Draw outer sun (red)
        draw.ellipse(
            (center_x - outer_radius, center_y - outer_radius,
             center_x + outer_radius, center_y + outer_radius),
            fill=red
        )
        
        # Draw middle sun (orange)
        draw.ellipse(
            (center_x - middle_radius, center_y - middle_radius,
             center_x + middle_radius, center_y + middle_radius),
            fill=orange
        )
        
        # Draw inner sun (yellow)
        draw.ellipse(
            (center_x - inner_radius, center_y - inner_radius,
             center_x + inner_radius, center_y + inner_radius),
            fill=yellow
        )
        
        # Draw eye (black)
        eye_radius = inner_radius * 0.4
        eye_x = center_x + inner_radius * 0.3
        eye_y = center_y - inner_radius * 0.2
        draw.ellipse(
            (eye_x - eye_radius, eye_y - eye_radius,
             eye_x + eye_radius, eye_y + eye_radius),
            fill=black
        )
        
        # Save the image
        image.save(png_path)
        print(f"Successfully created icon at {png_path}")
        return True
    except Exception as e:
        print(f"Error creating icon: {e}")
        return False

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define path for the PNG file
    png_path = os.path.join(script_dir, "icon.png")
    
    # Create the icon
    success = create_ndxplorer_icon(png_path)
    
    if success:
        print(f"Icon saved to {png_path}")
    else:
        print("Icon creation failed.")
        sys.exit(1)