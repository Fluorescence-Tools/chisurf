"""
Create an icon for the ndXplorer plugin: 2D histogram with marginal distributions.
This follows the convention used by other plugins' create_icon.py files and writes icon.png
in the same directory.
"""
from PIL import Image, ImageDraw
import os

SIZE = 128
BG = (255, 255, 255, 0)
FRAME_STROKE = (27, 30, 35, 255)
FRAME_FILL = (255, 255, 255, 255)
# Distinct colors for marginal distributions
MARG_TOP_COL = (220, 20, 60, 200)   # crimson-ish red base
MARG_RIGHT_COL = (46, 204, 113, 200)  # emerald green base

# blue gradient palette
PALETTE = [
    (219, 233, 255, 255),
    (183, 209, 255, 255),
    (147, 186, 255, 255),
    (110, 161, 242, 255),
    (75, 134, 224, 255),
    (58, 120, 214, 255),
    (79, 135, 223, 255),
    (106, 158, 238, 255),
    (90, 168, 247, 255),
    (85, 143, 230, 255),
    (47, 135, 223, 255),
    (47, 109, 203, 255),
    (31, 109, 203, 255),
    (43, 129, 223, 255)
]


def create_icon(path: str):
    img = Image.new("RGBA", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(img)

    # Frame params similar to SVG
    left, top, width, height = 18, 20, 80, 80
    right, bottom = left + width, top + height

    # Frame fill
    d.rectangle([left, top, right, bottom], fill=FRAME_FILL, outline=FRAME_STROKE, width=2)

    # Heatmap 5x5 cells
    cs = 16
    fills = [
        [0, 1, 2, 3, 4],
        [1, 2, 8, 10, 9],
        [1, 7, 10, 10, 11],
        [2, 7, 10, 12, 13],
        [2, 3, 4, 11, 12],
    ]
    for r in range(5):
        for c in range(5):
            x1 = left + c * cs
            y1 = top + r * cs
            x2 = x1 + cs
            y2 = y1 + cs
            color = PALETTE[fills[r][c] % len(PALETTE)]
            d.rectangle([x1, y1, x2, y2], fill=color)

    # Marginal top bars (red)
    top_bars = [
        (18, 10, 16, 8, 115),
        (34, 8, 16, 10, 140),
        (50, 6, 16, 12, 178),
        (66, 8, 16, 10, 140),
        (82, 12, 16, 6, 102),
    ]
    for x, y, w, h, a in top_bars:
        d.rectangle([x, y, x + w, y + h], fill=(MARG_TOP_COL[0], MARG_TOP_COL[1], MARG_TOP_COL[2], a))

    # Marginal right bars (green)
    right_bars = [
        (100, 84, 20, 16, 115),
        (100, 68, 22, 16, 140),
        (100, 52, 26, 16, 178),
        (100, 36, 22, 16, 140),
        (100, 20, 18, 16, 102),
    ]
    for x, y, w, h, a in right_bars:
        d.rectangle([x, y, x + w, y + h], fill=(MARG_RIGHT_COL[0], MARG_RIGHT_COL[1], MARG_RIGHT_COL[2], a))

    # Axes
    d.line([(left, bottom), (right, bottom)], fill=FRAME_STROKE, width=2)
    d.line([(left, top), (left, bottom)], fill=FRAME_STROKE, width=2)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    out = os.path.join(here, "icon.png")
    create_icon(out)
    print(f"Icon written to {out}")
