"""
Create an icon for the UCFRET plugin.

This script creates a Bayesian-themed icon for the UCFRET plugin,
featuring overlapping probability distributions to represent
Bayesian inference in FRET analysis.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create a new image with a transparent background
width, height = 128, 128
image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
draw = ImageDraw.Draw(image)

# Define colors with some transparency for overlapping
blue = (41, 128, 185, 180)  # Prior distribution
red = (231, 76, 60, 180)    # Likelihood distribution
purple = (155, 89, 182, 180)  # Posterior distribution

# Function to draw a Gaussian distribution curve
def draw_gaussian(x_center, y_center, sigma, amplitude, color, fill=False):
    points = []
    x_range = np.linspace(0, width, 100)

    for x in x_range:
        y = amplitude * np.exp(-0.5 * ((x - x_center) / sigma)**2)
        points.append((x, y_center - y))

    if fill:
        # Create a closed polygon by adding points at the bottom
        bottom_points = [(x, y_center) for x, _ in reversed(points)]
        all_points = points + bottom_points
        draw.polygon(all_points, fill=color)
    else:
        # Just draw the curve
        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=color, width=2)

# Draw the prior distribution (blue)
prior_center = width * 0.35
prior_sigma = width * 0.15
prior_amplitude = height * 0.3
draw_gaussian(prior_center, height * 0.6, prior_sigma, prior_amplitude, blue, fill=True)

# Draw the likelihood distribution (red)
likelihood_center = width * 0.65
likelihood_sigma = width * 0.12
likelihood_amplitude = height * 0.25
draw_gaussian(likelihood_center, height * 0.6, likelihood_sigma, likelihood_amplitude, red, fill=True)

# Draw the posterior distribution (purple, where they overlap)
posterior_center = width * 0.5
posterior_sigma = width * 0.1
posterior_amplitude = height * 0.35
draw_gaussian(posterior_center, height * 0.6, posterior_sigma, posterior_amplitude, purple, fill=False)

# Add "FRET" text
try:
    font = ImageFont.truetype("arial.ttf", 24)
except IOError:
    font = ImageFont.load_default()

text = "B-FRET"
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
text_position = ((width - text_width) // 2, height - text_height - 10)
draw.text(text_position, text, fill=(0, 0, 0, 255), font=font)

# Save the image
icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
image.save(icon_path)

print(f"Icon saved to {icon_path}")

if __name__ == "__main__":
    # If run directly, show the image
    image.show()
