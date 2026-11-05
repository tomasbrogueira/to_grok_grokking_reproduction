"""Digitise the paper's raster figures into anchors/anchors.json.

Reads only the PDF's own images (extracted to .cache/figs by pdfimages).
Pixel-to-value calibration comes from tick positions and labels read off each
panel. Every value is an approximation of a plotted point, not a data value.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

D = Path(".cache/figs")


def load(i):
    return np.asarray(Image.open(D / f"im-{i:03d}.png").convert("RGB")).astype(int)


def groups(v, gap=3):
    out = []
    for x in v:
        if out and x - out[-1][-1] <= gap:
            out[-1].append(int(x))
        else:
            out.append([int(x)])
    return out


def masks(a):
    blue = (a[..., 2] > 180) & (a[..., 0] < 110) & (a[..., 1] < 110)
    red = (a[..., 0] > 180) & (a[..., 1] < 110) & (a[..., 2] < 110)
    return blue, red


def clusters(mask, x, t, b, halfwidth=1, minrun=4):
    """Vertical runs of a colour around column x (markers are thicker than lines)."""
    col = mask[t:b, x - halfwidth : x + halfwidth + 1].any(axis=1)
    return [float(np.mean(g)) + t for g in groups(np.where(col)[0], 1) if len(g) >= minrun]


# Timing panels: (image index, marker columns, grid values, y->log2 t or linear)
PANELS = {
    "fig2a": dict(im=4, xs=[99, 215, 331, 446, 561], grid=[2e-6, 4e-6, 8e-6, 1.6e-5, 3.2e-5],
                  y=lambda p: 2 ** (19 - (p - 78) / 21.5)),
    "fig2b": dict(im=6, xs=[99, 215, 330, 446, 561], grid=[10, 20, 40, 80, 160],
                  y=lambda p: 2 ** (15 - (p - 41) / 25.5)),
    "fig2c": dict(im=8, xs=[99, 215, 331, 446, 561], grid=[1000, 1500, 2000, 2500, 3000],
                  y=lambda p: 2 ** (14 - (p - 73) / 40.0)),
    "fig2d": dict(im=10, xs=[85, 164, 244, 323, 402, 482, 561], grid=[1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
                  y=lambda p: (319 - p) / 57.0 * 2e4),
    "fig4a": dict(im=20, xs=[100, 192, 284, 377, 469, 561], grid=[.05, .1, .2, .4, .8, 1.6],
                  y=lambda p: 2 ** (16 - (p - 87) / 30.5)),
    "fig4b": dict(im=22, xs=[100, 215, 331, 446, 561], grid=[10, 20, 40, 80, 160],
                  y=lambda p: 2 ** (15 - (p - 96) / 30.0)),
    "fig4c": dict(im=24, xs=[100, 215, 331, 446, 561], grid=[2000, 2500, 3000, 3500, 4000],
                  y=lambda p: 2 ** (16 - (p - 62) / 35.3)),
    "fig4d": dict(im=26, xs=[102, 178, 254, 330, 406, 484, 560], grid=[.1, .2, .4, .8, 1.6, 3.2, 6.4],
                  y=lambda p: (319 - p) / 104.0 * 1e5),
}

out = {}
for name, p in PANELS.items():
    a = load(p["im"])
    blue, red = masks(a)
    rows = []
    for x, g in zip(p["xs"], p["grid"]):
        rows.append({"x": g,
                     "blue": sorted(round(float(p["y"](c)), 1) for c in clusters(blue, x, 41, 330)),
                     "red": sorted(round(float(p["y"](c)), 1) for c in clusters(red, x, 41, 330))})
    out[name] = rows
    print(name)
    for r in rows:
        print("  ", r)
Path("anchors").mkdir(exist_ok=True)
Path("anchors/timing_raw.json").write_text(json.dumps(out, indent=1))
