import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dbm_results import load_best_dbm_fits
from trial_data import load_trial_data


def make_accuracy_figure(
    out_path=None,
    include_models=None,
    exclude_models=None,
):
    if include_models is not None and exclude_models is not None:
        raise ValueError("Use include_models or exclude_models, not both")
    if out_path is None:
        if include_models is not None:
            label = "include_" + "_".join(include_models)
        elif exclude_models is not None:
            label = "exclude_" + "_".join(exclude_models)
        else:
            label = "include_all_models"
        out_path = f"../figures/subjects_accuracy_{label}_learning_nonguessers.png"

    d = load_trial_data()
    best = load_best_dbm_fits(
        include_models=include_models,
        exclude_models=exclude_models,
    )

    keep_cols = ["experiment", "condition", "subject"]
    d = d.merge(best[keep_cols].drop_duplicates(), on=keep_cols, how="inner")

    block_size = 25
    d["block"] = d["orig_trial_index"] // block_size
    d["condition"] = d["condition"].astype("category")
    d = (
        d.groupby(["experiment", "condition", "subject", "phase", "block"], observed=True)["acc"]
        .mean()
        .reset_index()
    )

    d1 = d[d["experiment"] == 1].copy()
    d2 = d[d["experiment"] == 2].copy()

    condition_colors = {
        "relearn": "#B35C44",
        "new_learn": "#4C7899",
    }
    sns.set_palette(
        sns.color_palette([condition_colors[c] for c in d["condition"].cat.categories])
    )

    d1["condition"] = d1["condition"].map({
        "relearn": "Relearn",
        "new_learn": "New Learn",
    })
    d2["condition"] = d2["condition"].map({
        "relearn": "Relearn",
        "new_learn": "New Learn",
    })

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 5))

    for i, phase in enumerate(["Learn", "Intervention", "Test"]):
        leg = i == 0
        d11 = d1[d1["phase"] == phase].copy()
        d22 = d2[d2["phase"] == phase].copy()

        sns.lineplot(
            data=d11,
            x="block",
            y="acc",
            hue="condition",
            err_style="bars",
            errorbar="se",
            marker="o",
            legend=leg,
            ax=ax[0, 0],
        )
        sns.lineplot(
            data=d22,
            x="block",
            y="acc",
            hue="condition",
            err_style="bars",
            errorbar="se",
            marker="o",
            legend=leg,
            ax=ax[0, 1],
        )

    ax[0, 0].set_title("Experiment 1", fontsize=14)
    ax[0, 1].set_title("Experiment 2", fontsize=14)

    for axx in ax.flatten():
        for boundary in [11.5, 23.5]:
            axx.axvline(boundary, color="black", linestyle="--", linewidth=1)
        for label, x in [("Learn", 5.5), ("Intervention", 17.5), ("Test", 29.5)]:
            axx.text(x, 0.42, label, ha="center", va="bottom", fontsize=12)
        axx.set_xlabel("Block", fontsize=14)
        axx.set_ylabel("Accuracy", fontsize=14)
        axx.set_ylim(.4, 0.9)
        axx.tick_params(axis="both", labelsize=12)
        sns.move_legend(axx, "upper right", title=None, frameon=False, fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", default=None)
    parser.add_argument(
        "--include-models",
        default=None,
        help="Comma-separated exact DBM model names to include before selecting winners.",
    )
    parser.add_argument(
        "--exclude-models",
        default=None,
        help="Comma-separated exact DBM model names to exclude before selecting winners.",
    )
    args = parser.parse_args()

    include_models = None
    exclude_models = None
    if args.include_models is not None and args.include_models != "":
        include_models = [
            model.strip() for model in args.include_models.split(",") if model.strip()
        ]
    if args.exclude_models is not None and args.exclude_models != "":
        exclude_models = [
            model.strip() for model in args.exclude_models.split(",") if model.strip()
        ]

    make_accuracy_figure(
        out_path=args.out_path,
        include_models=include_models,
        exclude_models=exclude_models,
    )
