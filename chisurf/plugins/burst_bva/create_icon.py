from PIL import Image, ImageDraw
import os

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a representation of burst variance analysis
# Central axis line (time axis)
draw.line([(16, 40), (48, 40)], fill=(0, 0, 0, 200), width=2)

# Draw bursts of different intensities
burst_centers = [20, 32, 44]
burst_heights = [15, 25, 18]
burst_widths = [6, 8, 7]
burst_colors = [(200, 0, 0, 200), (0, 200, 0, 200), (0, 0, 200, 200)]

for i, center in enumerate(burst_centers):
    # Draw burst (as a Gaussian-like shape)
    height = burst_heights[i]
    width = burst_widths[i]
    color = burst_colors[i]
    
    # Draw the burst as a series of vertical lines with varying heights
    for j in range(-width//2, width//2 + 1):
        x = center + j
        # Gaussian-like height calculation
        h = height * (1 - (j/width)**2)
        draw.line([(x, 40), (x, 40-h)], fill=color, width=1)

# Draw variance indicators (error bars) on top of bursts
for i, center in enumerate(burst_centers):
    # Draw variance bars
    variance = burst_heights[i] * 0.2  # 20% of height as variance
    draw.line([(center-3, 40-burst_heights[i]-variance), (center+3, 40-burst_heights[i]-variance)], 
              fill=(0, 0, 0, 200), width=1)
    draw.line([(center, 40-burst_heights[i]-variance*2), (center, 40-burst_heights[i])], 
              fill=(0, 0, 0, 200), width=1)

# Draw a small plot in the corner showing variance analysis
plot_x, plot_y = 14, 14
plot_width, plot_height = 16, 12
draw.rectangle([(plot_x, plot_y), (plot_x + plot_width, plot_y + plot_height)], 
               fill=(255, 255, 255, 200), outline=(0, 0, 0, 255))

# Draw axes for the variance plot
draw.line([(plot_x, plot_y + plot_height), (plot_x + plot_width, plot_y + plot_height)], 
          fill=(0, 0, 0, 255), width=1)  # x-axis
draw.line([(plot_x, plot_y), (plot_x, plot_y + plot_height)], 
          fill=(0, 0, 0, 255), width=1)  # y-axis

# Draw a variance curve (parabola)
curve_points = []
for i in range(8):
    x = plot_x + 2 + i * 1.5
    # Parabola: y = a(x-h)^2 + k
    y = plot_y + plot_height - 2 - 8 * ((i - 3.5) / 3.5) ** 2
    curve_points.append((x, y))

draw.line(curve_points, fill=(200, 0, 0, 255), width=1)

# Add a small "σ²" symbol to indicate variance
draw.text((plot_x + 2, plot_y + 2), "σ²", fill=(0, 0, 0, 255))

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")