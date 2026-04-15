import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _plot_heatmap(df, title, out_path, figsize):
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, vmin=0, vmax=1, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=int, default=6)
    parser.add_argument("--in-dir", type=str, default="../dbm_fits")
    parser.add_argument("--out-dir", type=str, default="../figures")
    args = parser.parse_args()

    fam_path = os.path.join(
        args.in_dir, f"dbm_recovery_empirical_block_{args.block}_family_props.csv"
    )
    mod_path = os.path.join(
        args.in_dir, f"dbm_recovery_empirical_block_{args.block}_model_props.csv"
    )

    fam = pd.read_csv(fam_path, index_col=0)
    mod = pd.read_csv(mod_path, index_col=0)

    os.makedirs(args.out_dir, exist_ok=True)

    fam_out = os.path.join(
        args.out_dir, f"dbm_recovery_pilot_block_{args.block}_family_props.png"
    )
    mod_out = os.path.join(
        args.out_dir, f"dbm_recovery_pilot_block_{args.block}_model_props.png"
    )

    _plot_heatmap(
        fam,
        title=f"Pilot Recovery Block {args.block}: Family Confusion",
        out_path=fam_out,
        figsize=(6, 4),
    )
    _plot_heatmap(
        mod,
        title=f"Pilot Recovery Block {args.block}: Model Confusion",
        out_path=mod_out,
        figsize=(7, 4),
    )

    print("Wrote:")
    print(fam_out)
    print(mod_out)
