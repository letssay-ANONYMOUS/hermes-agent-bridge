#!/usr/bin/env python3
"""Generate the app icons.

The mark is a hub-and-spoke: several outer nodes (the agent backends) linked
into one centre (this bridge). Regenerate with:

    python3 scripts/make_icons.py

Requires Pillow. Writes static/icon-{1024,512,180}.png.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent.parent / "static"
SIZE = 1024
BG = (13, 15, 17, 255)        # matches theme_color in the manifest
ACCENT = (233, 238, 245, 255)  # hub + spokes
NODE = (122, 162, 247, 255)    # backend nodes
SPOKE = (86, 98, 116, 255)


def rounded_background(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * 0.22), fill=BG)
    return img


def draw_mark(img: Image.Image, size: int) -> None:
    draw = ImageDraw.Draw(img)
    cx = cy = size / 2
    ring = size * 0.29
    hub_r = size * 0.088
    node_r = size * 0.062
    spoke_w = max(2, int(size * 0.018))

    # Five backends around the hub, first one pointing straight up.
    points = []
    for i in range(5):
        angle = -math.pi / 2 + i * (2 * math.pi / 5)
        points.append((cx + ring * math.cos(angle), cy + ring * math.sin(angle)))

    for px, py in points:
        draw.line([(cx, cy), (px, py)], fill=SPOKE, width=spoke_w)

    for px, py in points:
        draw.ellipse([px - node_r, py - node_r, px + node_r, py + node_r], fill=NODE)

    draw.ellipse([cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r], fill=ACCENT)


def main() -> None:
    base = rounded_background(SIZE)
    draw_mark(base, SIZE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in (1024, 512, 180):
        out = base if target == SIZE else base.resize((target, target), Image.LANCZOS)
        path = OUT_DIR / f"icon-{target}.png"
        out.save(path, "PNG", optimize=True)
        print(f"wrote {path.relative_to(OUT_DIR.parent)} ({target}x{target})")


if __name__ == "__main__":
    main()
