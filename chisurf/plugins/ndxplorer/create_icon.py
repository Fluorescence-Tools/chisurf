"""
Create an icon for the ndXplorer plugin: 2D histogram with marginal distributions.
Generates an 8x8 heatmap from an anisotropic 2D Gaussian and draws larger marginal
projections that decay towards zero. Boxes and marginals have black outlines for clarity.
"""
from PIL import Image, ImageDraw
import os
import math

SIZE = 128
BG = (255, 255, 255, 0)
FRAME_STROKE = (27, 30, 35, 255)
FRAME_FILL = (255, 255, 255, 255)
# Distinct colors for marginal distributions
MARG_TOP_COL = (220, 20, 60, 220)     # red (top)
MARG_RIGHT_COL = (46, 204, 113, 220)  # green (right)
OUTLINE_COL = (0, 0, 0, 255)          # black outlines

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


def _palette_color(val: float) -> tuple:
    """Map a normalized value 0..1 to a palette color index."""
    idx = max(0, min(len(PALETTE) - 1, int(round(val * (len(PALETTE) - 1)))))
    return PALETTE[idx]


def create_icon(path: str):
    img = Image.new("RGBA", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(img)

    # Frame params similar to SVG
    left, top, width, height = 18, 20, 80, 80
    right, bottom = left + width, top + height

    # Frame fill with border
    d.rectangle([left, top, right, bottom], fill=FRAME_FILL, outline=FRAME_STROKE, width=2)

    # Heatmap 8x8 cells computed from anisotropic Gaussian
    # Broad along x (sigma_x larger), sharp along y (sigma_y smaller)
    grid = 8
    cs = width // grid
    # Gaussian center slightly off-center to emphasize projection
    mu_x, mu_y = 0.60, 0.50
    sigma_x, sigma_y = 0.22, 0.12  # broad x, sharp y

    values = []
    for r in range(grid):
        row = []
        for c in range(grid):
            # map cell center to [0,1] coordinates
            x = (c + 0.5) / grid
            y = (r + 0.5) / grid
            z = math.exp(-(((x - mu_x) ** 2) / (2 * sigma_x ** 2) + ((y - mu_y) ** 2) / (2 * sigma_y ** 2)))
            row.append(z)
        values.append(row)

    # normalize values to [0,1]
    vmin = min(min(row) for row in values)
    vmax = max(max(row) for row in values)
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    norm = [[(v - vmin) / denom for v in row] for row in values]

    # draw heatmap cells with black outlines
    for r in range(grid):
        for c in range(grid):
            x1 = left + c * cs
            y1 = top + r * cs
            x2 = x1 + cs
            y2 = y1 + cs
            color = _palette_color(norm[r][c])
            d.rectangle([x1, y1, x2, y2], fill=color, outline=OUTLINE_COL, width=1)

    # Compute marginals (projections)
    # Top bars (function of x): sum over rows
    col_sums = [sum(values[r][c] for r in range(grid)) for c in range(grid)]
    # Right bars (function of y): sum over cols
    row_sums = [sum(values[r][c] for c in range(grid)) for r in range(grid)]

    # normalize marginals to [0,1]
    def _norm_list(ls):
        m = max(ls) if ls else 1.0
        return [v / m if m > 0 else 0.0 for v in ls]

    col_n = _norm_list(col_sums)
    row_n = _norm_list(row_sums)

    # Make marginals larger and clearly visible, within available margins
    # Top area: ~20px available; use up to 16px height
    max_top_h = 16
    for c in range(grid):
        h = max(1, int(round(col_n[c] * max_top_h)))  # at least 1px, decays to near zero
        bar_x1 = left + c * cs
        bar_y2 = top - 1  # just above the frame border
        bar_y1 = max(0, bar_y2 - h)
        d.rectangle([bar_x1, bar_y1, bar_x1 + cs, bar_y2], fill=MARG_TOP_COL, outline=OUTLINE_COL, width=1)

    # Right area: ~30px available; use up to 26px width
    max_right_w = 26
    right_start = right + 2
    for r in range(grid):
        w = max(1, int(round(row_n[r] * max_right_w)))  # at least 1px
        bar_x1 = right_start
        bar_x2 = min(SIZE - 2, bar_x1 + w)
        bar_y1 = top + r * cs
        d.rectangle([bar_x1, bar_y1, bar_x2, bar_y1 + cs], fill=MARG_RIGHT_COL, outline=OUTLINE_COL, width=1)

    # Axes (retain)
    d.line([(left, bottom), (right, bottom)], fill=FRAME_STROKE, width=2)
    d.line([(left, top), (left, bottom)], fill=FRAME_STROKE, width=2)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    out = os.path.join(here, "icon.png")
    create_icon(out)
    print(f"Icon written to {out}")
