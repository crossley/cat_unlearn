import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=int, default=6)
    parser.add_argument("--in-dir", type=str, default="../dbm_fits")
    parser.add_argument("--out-dir", type=str, default="../figures")
    args = parser.parse_args()

    fam_path = os.path.join(
        args.in_dir, f"fit_dbm_recovery_block_{args.block}_family_props.csv"
    )
    mod_path = os.path.join(
        args.in_dir, f"fit_dbm_recovery_block_{args.block}_model_props.csv"
    )

    fam = pd.read_csv(fam_path, index_col=0)
    mod = pd.read_csv(mod_path, index_col=0)

    model_order = [
        "nll_rand_guess",
        "nll_bias_guess",
        "nll_glc_0",
        "nll_glc_1",
        "nll_gcc_eq_0",
        "nll_gcc_eq_1",
        "nll_gcc_eq_2",
        "nll_gcc_eq_3",
        "nll_unix_0",
        "nll_unix_1",
        "nll_uniy_0",
        "nll_uniy_1",
    ]
    model_abbrev = {
        "nll_rand_guess": "RG",
        "nll_bias_guess": "BG",
        "nll_glc_0": "LC0",
        "nll_glc_1": "LC1",
        "nll_gcc_eq_0": "CC0",
        "nll_gcc_eq_1": "CC1",
        "nll_gcc_eq_2": "CC2",
        "nll_gcc_eq_3": "CC3",
        "nll_unix_0": "UX0",
        "nll_unix_1": "UX1",
        "nll_uniy_0": "UY0",
        "nll_uniy_1": "UY1",
    }
    mod_order = [model for model in model_order if model in mod.index or model in mod.columns]
    mod = mod.reindex(index=mod_order, columns=mod_order, fill_value=0)
    mod = mod.rename(index=model_abbrev, columns=model_abbrev)

    os.makedirs(args.out_dir, exist_ok=True)

    fam_out = os.path.join(
        args.out_dir, f"dbm_recovery_pilot_block_{args.block}_family_props.png"
    )
    mod_out = os.path.join(
        args.out_dir, f"dbm_recovery_pilot_block_{args.block}_model_props.png"
    )

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        fam,
        annot=True,
        annot_kws={"fontsize": 5},
        vmin=0,
        vmax=max(1, fam.values.max() + 0.3),
        cmap="Greys",
        cbar=False,
    )
    plt.xticks(rotation=0, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.xlabel("Recovered Family", fontsize=7)
    plt.ylabel("True Family", fontsize=7)
    plt.title(f"Pilot Recovery Block {args.block}: Family Confusion", fontsize=7)
    plt.tight_layout()
    plt.savefig(fam_out, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        mod,
        annot=True,
        annot_kws={"fontsize": 8},
        vmin=0,
        vmax=max(1, mod.values.max() + 0.3),
        cmap="Greys",
        cbar=False,
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.xlabel("Recovered Model", fontsize=12)
    plt.ylabel("True Model", fontsize=12)
    plt.title(f"Pilot Recovery Block {args.block}: Model Confusion", fontsize=12)
    plt.tight_layout()
    plt.savefig(mod_out, dpi=200)
    plt.close()

    print("Wrote:")
    print(fam_out)
    print(mod_out)
