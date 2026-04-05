import os

import numpy as np
import matplotlib.pyplot as plt

from util_func_stimcat import create_grating_patch, make_stim_cats

BACKGROUND = (0.5, 0.5, 0.5)
PANEL_LIMIT = 0.6
STIM_EXTENT = (-0.5, 0.5, -0.5, 0.5)
RESPONSE_TEXT = "Press D for A\nPress K for B"


def _style_panel(ax):
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(-PANEL_LIMIT, PANEL_LIMIT)
    ax.set_ylim(-PANEL_LIMIT, PANEL_LIMIT)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def _draw_fixation(ax):
    _style_panel(ax)
    arm = 0.1
    ax.plot([0, 0], [-arm, arm], linewidth=3, color="white")
    ax.plot([-arm, arm], [0, 0], linewidth=3, color="white")


def _draw_grating(ax, grating, text=None):
    _style_panel(ax)
    ax.imshow(grating,
              cmap="gray",
              interpolation="nearest",
              extent=STIM_EXTENT)
    if text is not None:
        ax.text(0,
                -0.7,
                text,
                ha="center",
                va="top",
                color="white",
                fontsize=9)


def _draw_feedback(ax, grating, is_correct):
    _draw_grating(ax, grating)
    color = "lime" if is_correct else "red"
    label = "Correct" if is_correct else "Incorrect"
    ax.add_patch(
        plt.Circle((0, 0), 0.55, fill=False, linewidth=4, edgecolor=color))
    ax.text(0,
            -0.7,
            label,
            ha="center",
            va="top",
            color="white",
            fontsize=9)


def _make_grating(spatial_freq, orientation_deg, pixels_per_inch=227 / 2,
                  size_cm=3):
    px_per_cm = pixels_per_inch / 2.54
    size_px = int(size_cm * px_per_cm)
    freq = spatial_freq * (px_per_cm**-1)
    theta = np.deg2rad(orientation_deg)
    return create_grating_patch(size_px, freq, theta)


def _fig_to_rgb_array(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(height, width, 4)[..., :3].copy()


def build_trial_figure(spatial_freq, orientation_deg, is_correct):
    grating = _make_grating(spatial_freq, orientation_deg)
    fig = plt.figure(figsize=(3.1, 3.0), facecolor=BACKGROUND)
    axes = [
        fig.add_axes([0.04, 0.64, 0.42, 0.30]),
        fig.add_axes([0.29, 0.36, 0.42, 0.30]),
        fig.add_axes([0.54, 0.08, 0.42, 0.30]),
    ]
    _draw_fixation(axes[0])
    _draw_grating(axes[1], grating, text=RESPONSE_TEXT)
    _draw_feedback(axes[2], grating, is_correct)
    image = _fig_to_rgb_array(fig)
    plt.close(fig)
    return image


def _select_example_trials():
    np.random.seed(4)
    ds = make_stim_cats()
    return ds[(ds.x == ds.x.min())].iloc[0], ds[(ds.y == ds.y.max())].iloc[0]


def make_example_trials_figure(
        save_path="../figures/example_trials_figure.png", dpi=300):
    trial1, trial2 = _select_example_trials()
    panels = [
        build_trial_figure(trial1["xt"], np.rad2deg(trial1["yt"]), True),
        build_trial_figure(trial2["xt"], np.rad2deg(trial2["yt"]), False),
    ]
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    positions = [
        [0.05, 0.70, 0.24, 0.24],
        [0.40, 0.70, 0.24, 0.24],
    ]

    for image, (x0, y0, w, h) in zip(panels, positions):
        ax = fig.add_axes([x0, y0 - 0.30, w + 0.22, h + 0.18])
        ax.imshow(image)
        ax.axis("off")

    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved updated overlapped-flow figure as '{save_path}'")


if __name__ == "__main__":
    make_example_trials_figure()
