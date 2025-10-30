#!/usr/bin/env python3
# det_occ_map.py
#
# Deterministic occupancy map generator (PNG, 1-bit).
# 0 = drivable (white), 1 = non-drivable (black).
# Origin (meters): bottom-left. X right, Y up.
#
# Outputs (only these two):
#   <prefix>_raw.png
#   <prefix>_legend_axes.png
#
# Standard edge:
#   A continuous obstacle ring along the outer boundary.
#   Thickness is given in meters (default 0.25 m). Set --edge-m 0 to disable.
#
# Shapes (all parameters in meters; repeat flags to add multiple shapes):
#   --rect   x y w h
#   --line   x1 y1 x2 y2        (use --line-w for width)
#   --circle cx cy r
#   --poly   x1 y1 x2 y2 ... xn yn   (>=3 points, even number of coords)
#
# Example:
#   python det_occ_map.py --out-prefix map --width-m 80 --height-m 60 --mpc 0.05 \
#     --rect 5 5 10 6 --rect 25 5 10 6 --rect 45 5 10 6 \
#     --rect 5 20 12 8 --rect 30 22 8 14 --rect 55 24 12 10
#
from __future__ import annotations

import argparse
import math
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------- helpers ----------------


def meters_to_cell(x_m: float, y_m: float, mpc: float, H: int) -> Tuple[int, int]:
    cx = int(round(x_m / mpc))
    cy_from_bottom = int(round(y_m / mpc))
    ry = H - 1 - cy_from_bottom
    return cx, ry


def clamp_rect(x0: int, y0: int, x1: int, y1: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0c = max(0, min(W, x0))
    x1c = max(0, min(W, x1))
    y0c = max(0, min(H, y0))
    y1c = max(0, min(H, y1))
    if x0c > x1c:
        x0c, x1c = x1c, x0c
    if y0c > y1c:
        y0c, y1c = y1c, y0c
    return x0c, y0c, x1c, y1c


# ------------- rasterization -------------


def add_rect(grid: np.ndarray, x_m: float, y_m: float, w_m: float, h_m: float, mpc: float) -> None:
    H, W = grid.shape
    x0c, y0r = meters_to_cell(x_m, y_m, mpc, H)
    x1c, y1r = meters_to_cell(x_m + w_m, y_m + h_m, mpc, H)
    x0, y0, x1, y1 = clamp_rect(x0c, y1r, x1c, y0r, W, H)
    grid[y0:y1, x0:x1] = 1


def add_line(grid: np.ndarray, x1_m: float, y1_m: float, x2_m: float, y2_m: float, width_m: float, mpc: float) -> None:
    H, W = grid.shape
    mask = Image.new("1", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    x1c, y1r = meters_to_cell(x1_m, y1_m, mpc, H)
    x2c, y2r = meters_to_cell(x2_m, y2_m, mpc, H)
    w_px = max(1, int(round(width_m / mpc)))
    draw.line([(x1c, y1r), (x2c, y2r)], fill=1, width=w_px, joint="curve")
    grid[:] = np.maximum(grid, np.array(mask, dtype=np.uint8))


def add_circle(grid: np.ndarray, cx_m: float, cy_m: float, r_m: float, mpc: float) -> None:
    H, W = grid.shape
    cx, cy = meters_to_cell(cx_m, cy_m, mpc, H)
    r_px = max(1, int(round(r_m / mpc)))
    mask = Image.new("1", (W, H), 0)
    ImageDraw.Draw(mask).ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], fill=1)
    grid[:] = np.maximum(grid, np.array(mask, dtype=np.uint8))


def add_poly(grid: np.ndarray, pts_m: List[Tuple[float, float]], mpc: float) -> None:
    if len(pts_m) < 3:
        raise ValueError("Polygon needs at least 3 points.")
    H, W = grid.shape
    pts_px = [meters_to_cell(x, y, mpc, H) for (x, y) in pts_m]
    mask = Image.new("1", (W, H), 0)
    ImageDraw.Draw(mask).polygon(pts_px, outline=1, fill=1)
    grid[:] = np.maximum(grid, np.array(mask, dtype=np.uint8))


# ---------------- rendering ---------------


def paste_grid(canvas: Image.Image, origin_xy: Tuple[int, int], grid01: np.ndarray) -> None:
    ox, oy = origin_xy
    L = ((1 - grid01) * 255).astype(np.uint8)  # 0->white, 1->black
    img = Image.fromarray(L, "L").point(lambda p: 255 if p >= 128 else 0, "1")
    canvas.paste(img, (ox, oy))


def nice_step(range_m: float, target_ticks: int = 8) -> float:
    if range_m <= 0:
        return 1.0
    raw = range_m / max(target_ticks, 1)
    e = math.floor(math.log10(raw))
    base = raw / (10**e)
    k = 1.0 if base <= 1 else 2.0 if base <= 2 else 5.0 if base <= 5 else 10.0
    return k * (10**e)


def draw_axes(canvas: Image.Image, plot_origin: Tuple[int, int], plot_size: Tuple[int, int], mpp: float, grid_step_m: float) -> None:
    d = ImageDraw.Draw(canvas)
    f = ImageFont.load_default()
    x0, y0 = plot_origin
    w, h = plot_size

    # axes
    d.line([(x0, y0 + h), (x0 + w, y0 + h)], fill=0, width=1)
    d.line([(x0, y0), (x0, y0 + h)], fill=0, width=1)

    # grid
    step_px = max(1, int(round(grid_step_m / mpp)))
    if step_px >= 10:
        x = x0
        while x <= x0 + w:
            d.line([(x, y0), (x, y0 + h)], fill=0, width=1)
            x += step_px
        y = y0 + h
        while y >= y0:
            d.line([(x0, y), (x0 + w, y)], fill=0, width=1)
            y -= step_px

    # ticks
    width_m, height_m = w * mpp, h * mpp
    sx, sy = nice_step(width_m), nice_step(height_m)

    xm = 0.0
    while xm <= width_m + 1e-9:
        xp = x0 + int(round(xm / mpp))
        d.line([(xp, y0 + h), (xp, y0 + h + 6)], fill=0)
        t = f"{xm:.0f} m" if sx >= 1 else f"{xm:.2f} m"
        tw = d.textlength(t, font=f)
        d.text((xp - int(tw // 2), y0 + h + 8), t, font=f, fill=0)
        xm += sx

    ym = 0.0
    while ym <= height_m + 1e-9:
        yp = y0 + h - int(round(ym / mpp))
        d.line([(x0 - 6, yp), (x0, yp)], fill=0)
        t = f"{ym:.0f} m" if sy >= 1 else f"{ym:.2f} m"
        tw = d.textlength(t, font=f)
        d.text((x0 - 8 - int(tw), yp - 6), t, font=f, fill=0)
        ym += sy

    # size label
    size_lbl = f"W={width_m:.2f} m, H={height_m:.2f} m"
    tw = d.textlength(size_lbl, font=f)
    d.text((canvas.width - 12 - int(tw), canvas.height - 22), size_lbl, font=f, fill=0)

    # legend (fixed, top-right)
    items = [("Drivable", 1), ("Non-drivable", 0)]
    sw, sp, to = 14, 8, 6
    mw = max(d.textlength(lbl, font=f) for lbl, _ in items)
    bx = canvas.width - 12 - int(2 * sp + sw + to + mw)
    by = 12
    bw = int(2 * sp + sw + to + mw)
    bh = int(sp + len(items) * (sw + sp))
    d.rectangle([bx, by, bx + bw, by + bh], outline=0, fill=1)
    cy = by + sp
    for lbl, white_fill in items:
        d.rectangle([bx + sp, cy, bx + sp + sw, cy + sw], outline=0, fill=white_fill)
        d.text((bx + sp + sw + to, cy), lbl, font=f, fill=0)
        cy += sw + sp


# ---------------- main -------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic occupancy map PNG generator (raw + legend_axes).")
    ap.add_argument("--out-prefix", required=True, help="Output file prefix (no extension).")
    ap.add_argument("--width-m", type=float, required=True, help="Map width in meters.")
    ap.add_argument("--height-m", type=float, required=True, help="Map height in meters.")
    ap.add_argument("--mpc", type=float, default=0.05, help="Meters per cell (resolution).")
    ap.add_argument("--edge-m", type=float, default=0.25, help="Standard edge thickness in meters (0 disables).")
    ap.add_argument("--frame-px", type=int, default=64, help="Frame size for legend+axes variant.")
    ap.add_argument("--grid-step-m", type=float, default=5.0, help="Grid spacing in meters on legend+axes.")
    # shapes (repeatable)
    ap.add_argument("--rect", nargs=4, type=float, action="append", metavar=("X", "Y", "W", "H"))
    ap.add_argument("--line", nargs=4, type=float, action="append", metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--line-w", type=float, default=0.5, help="Line width in meters for all --line entries.")
    ap.add_argument("--circle", nargs=3, type=float, action="append", metavar=("CX", "CY", "R"))
    ap.add_argument("--poly", nargs="+", type=float, action="append", metavar="XnYn")
    args = ap.parse_args()

    # grid size
    W = int(round(args.width_m / args.mpc))
    H = int(round(args.height_m / args.mpc))
    if W <= 0 or H <= 0:
        raise SystemExit("Width/height too small for selected resolution.")

    # base grid: all drivable
    g = np.zeros((H, W), dtype=np.uint8)

    # standard edge ring (outer border)
    if args.edge_m > 0:
        b = max(1, int(round(args.edge_m / args.mpc)))
        g[:b, :] = 1
        g[-b:, :] = 1
        g[:, :b] = 1
        g[:, -b:] = 1

    # shapes
    if args.rect:
        for x, y, w, h in args.rect:
            add_rect(g, x, y, w, h, args.mpc)
    if args.line:
        for x1, y1, x2, y2 in args.line:
            add_line(g, x1, y1, x2, y2, args.line_w, args.mpc)
    if args.circle:
        for cx, cy, r in args.circle:
            add_circle(g, cx, cy, r, args.mpc)
    if args.poly:
        for coords in args.poly:
            if len(coords) < 6 or len(coords) % 2 != 0:
                raise SystemExit("--poly requires an even number of coords and at least 3 points.")
            pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            add_poly(g, pts, args.mpc)

    # RAW
    img_raw = Image.new("1", (W, H), 1)
    paste_grid(img_raw, (0, 0), g)
    img_raw.save(f"{args.out_prefix}_raw.png", format="PNG", optimize=True)

    # LEGEND + AXES
    canvas = Image.new("1", (W + 2 * args.frame_px, H + 2 * args.frame_px), 1)
    paste_grid(canvas, (args.frame_px, args.frame_px), g)
    draw_axes(canvas, (args.frame_px, args.frame_px), (W, H), args.mpc, args.grid_step_m)
    canvas.save(f"{args.out_prefix}_legend_axes.png", format="PNG", optimize=True)


if __name__ == "__main__":
    main()
