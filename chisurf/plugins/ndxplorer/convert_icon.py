import os
import sys
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    from PIL import Image
    SVGLIB_AVAILABLE = True
except ImportError:
    print("Warning: svglib, reportlab, or PIL not available. SVG to PNG conversion will be disabled.")
    SVGLIB_AVAILABLE = False

def convert_svg_to_png(svg_path, png_path, width=128, height=128):
    """
    Convert SVG file to PNG using svglib and reportlab.

    Parameters:
    -----------
    svg_path : str
        Path to the SVG file
    png_path : str
        Path where the PNG file will be saved
    width : int
        Width of the output PNG image
    height : int
        Height of the output PNG image
    """
    if not SVGLIB_AVAILABLE:
        print(f"Cannot convert {svg_path} to {png_path}: svglib, reportlab, or PIL not available")
        return False

    try:
        # Convert SVG to ReportLab Graphics object
        drawing = svg2rlg(svg_path)

        # Create a temporary PNG file
        temp_png = png_path + ".temp.png"
        renderPM.drawToFile(drawing, temp_png, fmt="PNG")

        # Resize the image to the desired dimensions
        img = Image.open(temp_png)
        img = img.resize((width, height), Image.LANCZOS)
        img.save(png_path)

        # Remove the temporary file
        os.remove(temp_png)

        print(f"Successfully converted {svg_path} to {png_path}")
        return True
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return False

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths
    svg_path = os.path.join(script_dir, "icon.svg")
    png_path = os.path.join(script_dir, "icon.png")

    # Check if SVG file exists
    if not os.path.exists(svg_path):
        print(f"Error: SVG file not found at {svg_path}")
        sys.exit(1)

    # Convert SVG to PNG
    success = convert_svg_to_png(svg_path, png_path)

    if success:
        print(f"Icon saved to {png_path}")
    else:
        print("Conversion failed.")
        sys.exit(1)
