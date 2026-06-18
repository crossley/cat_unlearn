import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from util_func_glc_gcc_recovery import (
    GCC_NOISE_GRID,
    GCC_SIDE_GRID,
    GCC_XC_GRID,
    GCC_YC_GRID,
    GLC_DIAGONAL_GRID,
    GLC_NOISE_GRID,
    GLC_SIDE_GRID,
    GLC_SLOPE_GRID,
    make_cat_trials,
)


def _plot_stimuli(ax, x, y, cat):
    ax.scatter(x[cat == 0], y[cat == 0], s=8, alpha=0.35, color="C0")
    ax.scatter(x[cat == 1], y[cat == 1], s=8, alpha=0.35, color="C1")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", adjustable="box")


def plot_glc_grid(out_path, n_per_cat):
    np.random.seed(462)
    x, y, cat = make_cat_trials(n_per_cat)
    xline = np.array([0, 100])

    fig, ax = plt.subplots(
        len(GLC_NOISE_GRID),
        len(GLC_SIDE_GRID),
        squeeze=False,
        figsize=(8, 10),
    )

    for r, noise in enumerate(GLC_NOISE_GRID):
        for c, side in enumerate(GLC_SIDE_GRID):
            _plot_stimuli(ax[r, c], x, y, cat)
            for slope in GLC_SLOPE_GRID:
                for diag in GLC_DIAGONAL_GRID:
                    yline = slope * (xline - diag) + diag
                    ax[r, c].plot(xline, yline, color="black", alpha=0.12, linewidth=0.8)
            ax[r, c].set_title(f"GLC side {side}, noise {noise:g}")
            ax[r, c].set_xlabel("x")
            ax[r, c].set_ylabel("y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gcc_grid(out_path, n_per_cat):
    np.random.seed(462)
    x, y, cat = make_cat_trials(n_per_cat)

    fig, ax = plt.subplots(
        len(GCC_NOISE_GRID),
        len(GCC_SIDE_GRID),
        squeeze=False,
        figsize=(14, 10),
    )

    for r, noise in enumerate(GCC_NOISE_GRID):
        for c, side in enumerate(GCC_SIDE_GRID):
            _plot_stimuli(ax[r, c], x, y, cat)
            for xc in GCC_XC_GRID:
                ax[r, c].axvline(xc, color="black", alpha=0.14, linewidth=0.8)
            for yc in GCC_YC_GRID:
                ax[r, c].axhline(yc, color="black", alpha=0.14, linewidth=0.8)
            ax[r, c].set_title(f"GCC side {side}, noise {noise:g}")
            ax[r, c].set_xlabel("x")
            ax[r, c].set_ylabel("y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="../figures")
    parser.add_argument("--n-per-cat", type=int, default=450)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    glc_out = os.path.join(args.out_dir, "glc_recovery_param_grid.png")
    gcc_out = os.path.join(args.out_dir, "gcc_recovery_param_grid.png")

    plot_glc_grid(glc_out, args.n_per_cat)
    plot_gcc_grid(gcc_out, args.n_per_cat)

    print("Wrote:")
    print(glc_out)
    print(gcc_out)
