#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PANEL_W = 600
PANEL_H = 400
OVERLAP_X = 250
OFFSET_Y = 200
MARGIN_X = 60
MARGIN_Y = 70
BOTTOM_PAD = 80

BG = (255, 255, 255)
SCREEN_GRAY = (120, 120, 120)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (56, 160, 72)
RED = (176, 62, 62)
ARROW = (70, 70, 70)


def make_panel_base():
    return np.full((PANEL_H, PANEL_W, 3), SCREEN_GRAY, dtype=np.uint8)


def add_fixation_cross(panel, size=28, thickness=4):
    cy = PANEL_H // 2
    cx = PANEL_W // 2
    panel[cy - thickness // 2 : cy + (thickness + 1) // 2, cx - size : cx + size + 1] = WHITE
    panel[cy - size : cy + size + 1, cx - thickness // 2 : cx + (thickness + 1) // 2] = WHITE
    return panel


def add_circular_sine_grating(panel, radius=95, spatial_freq=0.055, orientation_deg=35):
    yy, xx = np.mgrid[0:PANEL_H, 0:PANEL_W]
    cx = PANEL_W / 2.0
    cy = PANEL_H / 2.0
    x = xx - cx
    y = yy - cy

    theta = np.deg2rad(orientation_deg)
    xr = x * np.cos(theta) + y * np.sin(theta)
    wave = 127.5 + 127.5 * np.sin(2 * np.pi * spatial_freq * xr)
    wave = wave.astype(np.uint8)

    mask = (x * x + y * y) <= radius * radius
    panel[mask] = np.stack([wave, wave, wave], axis=-1)[mask]
    return panel


def paste_panel(canvas_img, panel_arr, left, top, border=2):
    panel_img = Image.fromarray(panel_arr, mode="RGB")
    canvas_img.paste(panel_img, (left, top))
    draw = ImageDraw.Draw(canvas_img)
    draw.rectangle(
        [left, top, left + PANEL_W - 1, top + PANEL_H - 1],
        outline=WHITE,
        width=border,
    )


def add_feedback_ring(canvas_img,
                      left,
                      top,
                      is_correct=True,
                      radius=105,
                      width=7):
    cx = left + PANEL_W // 2
    cy = top + PANEL_H // 2
    draw = ImageDraw.Draw(canvas_img)
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(bbox, outline=GREEN if is_correct else RED, width=width)


def add_time_arrow(canvas_img, panel_x, panel_y):
      draw = ImageDraw.Draw(canvas_img)

      # Arrow runs from bottom-left vertex of fixation panel
      # to bottom-left vertex of feedback panel.
      start_x = panel_x[0]
      start_y = panel_y[0] + PANEL_H
      end_x = panel_x[2]
      end_y = panel_y[2] + PANEL_H

      draw.line((start_x, start_y, end_x, end_y), fill=ARROW, width=5)

      # Arrowhead aligned with the line direction.
      head = 18
      vx = end_x - start_x
      vy = end_y - start_y
      norm = max((vx * vx + vy * vy) ** 0.5, 1.0)
      ux = vx / norm
      uy = vy / norm
      px = -uy
      py = ux

      tip = (end_x, end_y)
      left = (
          end_x - ux * head - px * (head * 0.55),
          end_y - uy * head - py * (head * 0.55),
      )
      right = (
          end_x - ux * head + px * (head * 0.55),
          end_y - uy * head + py * (head * 0.55),
      )
      draw.polygon([tip, left, right], fill=ARROW)


def _draw_keycap(draw, cx, cy, key_label="F"):
    key_w = 64
    key_h = 52
    x0 = cx - key_w // 2
    y0 = cy - key_h // 2
    x1 = x0 + key_w
    y1 = y0 + key_h

    draw.rounded_rectangle([x0 + 2, y0 + 3, x1 + 2, y1 + 3], radius=10, fill=(230, 230, 230), outline=None)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=WHITE, outline=ARROW, width=3)

    bbox = draw.textbbox((0, 0), key_label)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw / 2, cy - th / 2 - 1), key_label, fill=BLACK)


def add_stim_response_keys(canvas_img, panel_x, panel_y):
    draw = ImageDraw.Draw(canvas_img)

    stim_left = panel_x[1]
    stim_top = panel_y[1]

    # Top-left-ish and top-right-ish within the stimulus panel.
    f_cx = stim_left + 85
    f_cy = stim_top + 55
    j_cx = stim_left + PANEL_W - 85
    j_cy = stim_top + 55

    _draw_keycap(draw, f_cx, f_cy, "F")
    _draw_keycap(draw, j_cx, j_cy, "J")


def build_trial_figure(spatial_freq=0.055,
                       orientation_deg=35,
                       is_correct=True,
                       save_path=None):
    panel_x = [MARGIN_X + i * (PANEL_W - OVERLAP_X) for i in range(3)]
    panel_y = [MARGIN_Y + i * OFFSET_Y for i in range(3)]
    total_w = panel_x[-1] + PANEL_W + MARGIN_X
    total_h = panel_y[-1] + PANEL_H + BOTTOM_PAD
    canvas = Image.new("RGB", (total_w, total_h), BG)

    fixation = add_fixation_cross(make_panel_base())
    stim = add_circular_sine_grating(make_panel_base(),
                                     spatial_freq=spatial_freq,
                                     orientation_deg=orientation_deg)
    feedback = add_circular_sine_grating(make_panel_base(),
                                         spatial_freq=spatial_freq,
                                         orientation_deg=orientation_deg)

    paste_panel(canvas, fixation, panel_x[0], panel_y[0])
    paste_panel(canvas, stim, panel_x[1], panel_y[1])
    paste_panel(canvas, feedback, panel_x[2], panel_y[2])
    add_feedback_ring(canvas, panel_x[2], panel_y[2], is_correct=is_correct)
    add_time_arrow(canvas, panel_x, panel_y)
    if save_path is not None:
        canvas.save(save_path, format="PNG")

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-freq", type=float, default=0.055)
    parser.add_argument("--orientation-deg", type=float, default=35)
    parser.add_argument("--incorrect", action="store_true")
    parser.add_argument("--output", type=str, default="example_trial.png")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / args.output
    build_trial_figure(spatial_freq=args.spatial_freq,
                       orientation_deg=args.orientation_deg,
                       is_correct=not args.incorrect,
                       save_path=out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
