#!/usr/bin/env python3
"""
Simple burst‑trace icon generator
--------------------------------
Run this script as-is. It writes a single PNG called "icon.png" in the
current directory. The icon shows a single‑molecule style time trace with
one highlighted burst window and the label "MLE" above it.

No arguments. No parsing. Just run it.
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# ==== Tunables (edit here if you like) ====
SIZE = 512           # output size (pixels)
LABEL = "MLE"        # text shown above the burst box
SEED = 42            # random seed for the synthetic trace
TRACE_LEN = 240      # number of samples in trace (horizontal resolution)
BURSTS = 1           # number of Gaussian bursts to synthesize
BURST_AMP = 1.0      # relative amplitude of bursts
BURST_SIGMA = 14.0   # width of bursts in samples
SPIKE_PROB = 0.10    # probability of sharp spikes (detector blips)
NOISE_STD = 0.06     # baseline noise level

# Style
MARGIN_RATIO = 0.12
LINE_WIDTH_RATIO = 0.018
TEXT_STROKE_RATIO = 0.018
TRACE_COLOR = (255, 255, 255, 245)
BOX_OUTLINE = (255, 255, 255, 220)
SHADE_COLOR = (56, 189, 248, 90)  # cyan-ish translucent
TEXT_FILL = (255, 255, 255, 255)
TEXT_STROKE = (0, 0, 0, 220)
SCALE = 2  # render at higher res, then downsample for crisp lines

# ==========================================


def _find_font(size: int) -> ImageFont.ImageFont:
    """Try a few common fonts; fall back to PIL's default if none found."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def synth_trace(n: int, bursts: int, amp: float, sigma: float, spike: float, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    y = 0.15 + rng.normal(0.0, noise, size=n)

    # Gaussian bursts placed away from edges
    if bursts > 0:
        centers = rng.integers(int(0.2*n), int(0.8*n), size=bursts)
        amps = amp * (0.85 + 0.3 * rng.random(bursts))
        sigmas = sigma * (0.85 + 0.3 * rng.random(bursts))
        for c, a, s in zip(centers, amps, sigmas):
            y += a * np.exp(-0.5 * ((t - c) / s)**2)

    # Occasional sharp spikes
    if spike > 0:
        mask = rng.random(n) < spike
        y[mask] += 0.6 + 0.6 * rng.random(mask.sum())

    # Light smoothing for icon aesthetics
    k = 5
    kernel = np.ones(k) / k
    y = np.convolve(y, kernel, mode='same')
    return y


def _fit_trace_to_box(y: np.ndarray, box):
    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    ymin, ymax = float(y.min()), float(y.max())
    y_norm = (y - ymin) / (max(1e-9, ymax - ymin))
    xs = np.linspace(x0, x1, len(y))
    ys = y1 - y_norm * h
    return list(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))


def _main_burst_region(y: np.ndarray, frac: float = 0.38):
    """Return (start_idx, end_idx) of the largest region above frac * max(y)."""
    thr = float(y.max()) * frac
    above = y >= thr
    best_len, best = 0, (0, 0)
    i, n = 0, len(y)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i > best_len:
                best_len = j - i
                best = (i, j - 1)
            i = j
        else:
            i += 1
    return best


def main():
    S = SIZE * SCALE
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Plot area
    m = int(S * MARGIN_RATIO)
    box = (m, int(S * 0.30), S - m, int(S * 0.80))
    draw.rounded_rectangle(box, radius=int(S * 0.06), outline=BOX_OUTLINE, width=max(2, int(S * 0.006)))

    # Trace
    y = synth_trace(TRACE_LEN, BURSTS, BURST_AMP, BURST_SIGMA, SPIKE_PROB, NOISE_STD, SEED)
    pts = _fit_trace_to_box(y, box)

    # Highlight main burst window
    i0, i1 = _main_burst_region(y, frac=0.38)
    if i1 > i0:
        x0 = int(np.interp(i0, [0, len(y) - 1], [box[0], box[2]]))
        x1 = int(np.interp(i1, [0, len(y) - 1], [box[0], box[2]]))
        shade = Image.new('RGBA', (S, S), (0, 0, 0, 0))
        ImageDraw.Draw(shade).rounded_rectangle((x0, box[1], x1, box[3]), radius=int(S * 0.02), fill=SHADE_COLOR)
        img.alpha_composite(shade)

    # Trace line
    lw = max(2, int(S * LINE_WIDTH_RATIO))
    draw.line(pts, fill=TRACE_COLOR, width=lw, joint='curve')

    # Label
    font_size = int(S * 0.28)
    font = _find_font(font_size)
    tb = draw.textbbox((0, 0), LABEL, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = (S - tw) // 2
    ty = int(S * 0.10)
    stroke_w = max(1, int(S * TEXT_STROKE_RATIO))
    draw.text((tx, ty), LABEL, font=font, fill=TEXT_FILL, stroke_fill=TEXT_STROKE, stroke_width=stroke_w)

    # Downsample & save
    if SCALE != 1:
        img = img.resize((SIZE, SIZE), Image.LANCZOS)
    img.save('icon.png', 'PNG')


if __name__ == '__main__':
    main()
