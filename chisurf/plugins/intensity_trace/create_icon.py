from PIL import Image, ImageDraw
import os
import math
import random

# Create a 64x64 image with a transparent background
icon = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a background rectangle
draw.rectangle([(8, 8), (56, 56)], fill=(240, 240, 240, 200), outline=(0, 0, 0, 255))

# Draw a time trace with intensity fluctuations
# X and Y axes
draw.line([(12, 44), (52, 44)], fill=(0, 0, 0, 255), width=1)  # X-axis (time)
draw.line([(12, 12), (12, 44)], fill=(0, 0, 0, 255), width=1)  # Y-axis (intensity)

# Add small ticks to the axes
for i in range(4):
    x_tick = 12 + (i+1) * 10
    draw.line([(x_tick, 44), (x_tick, 46)], fill=(0, 0, 0, 255), width=1)  # X-axis ticks
    
    y_tick = 44 - (i+1) * 8
    draw.line([(10, y_tick), (12, y_tick)], fill=(0, 0, 0, 255), width=1)  # Y-axis ticks

# Generate a time trace with intensity fluctuations
# Define states for HMM-like representation
states = [
    {"level": 15, "start": 12, "end": 20, "color": (0, 150, 0, 255)},
    {"level": 30, "start": 20, "end": 28, "color": (150, 0, 0, 255)},
    {"level": 20, "start": 28, "end": 36, "color": (0, 0, 150, 255)},
    {"level": 35, "start": 36, "end": 52, "color": (150, 0, 150, 255)}
]

# Draw the time trace with state transitions
for state in states:
    level = 44 - state["level"]  # Adjust for coordinate system
    start = state["start"]
    end = state["end"]
    color = state["color"]
    
    # Draw the state level
    draw.line([(start, level), (end, level)], fill=color, width=2)
    
    # Add some noise to make it look like a real trace
    points = []
    for x in range(start, end + 1):
        y = level + random.uniform(-1.5, 1.5)
        points.append((x, y))
    
    # Draw the noisy trace
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=color, width=1)
    
    # If not the last state, draw a transition line to the next state
    if state != states[-1]:
        next_level = 44 - states[states.index(state) + 1]["level"]
        draw.line([(end, level), (end, next_level)], fill=(100, 100, 100, 200), width=1)

# Add a small histogram in the corner to represent intensity distribution
hist_x1, hist_y1 = 40, 15
hist_x2, hist_y2 = 52, 30

# Histogram background
draw.rectangle([(hist_x1, hist_y1), (hist_x2, hist_y2)], 
               fill=(220, 220, 220, 150), 
               outline=(0, 0, 0, 200))

# Histogram bars representing different intensity states
bar_widths = [2, 2, 2, 2]
bar_heights = [3, 6, 4, 7]
bar_colors = [(0, 150, 0, 200), (150, 0, 0, 200), (0, 0, 150, 200), (150, 0, 150, 200)]
bar_spacing = 1
start_x = hist_x1 + 1

for width, height, color in zip(bar_widths, bar_heights, bar_colors):
    x1 = start_x
    y1 = hist_y2 - height
    x2 = x1 + width
    y2 = hist_y2
    draw.rectangle([(x1, y1), (x2, y2)], 
                   fill=color, 
                   outline=(0, 0, 0, 200))
    start_x += width + bar_spacing

# Save the icon
icon.save('icon.png')
print(f"Icon created at {os.path.abspath('icon.png')}")