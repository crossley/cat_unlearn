import argparse

import matplotlib.pyplot as plt
import pandas as pd

from dbm_results import load_best_dbm_fits


def block_cross_counts(d, experiment, condition, block_x, block_y, value_col, order):
    d_x = d[
        (d["experiment"] == experiment)
        & (d["condition"] == condition)
        & (d["block"] == block_x)
    ][["subject", value_col]].rename(columns={value_col: f"b{block_x}"})

    d_y = d[
        (d["experiment"] == experiment)
        & (d["condition"] == condition)
        & (d["block"] == block_y)
    ][["subject", value_col]].rename(columns={value_col: f"b{block_y}"})

    both = pd.merge(d_x, d_y, on="subject", how="inner")
    if both.empty:
        return pd.DataFrame(0, index=order, columns=order)

    both[f"b{block_y}"] = pd.Categorical(both[f"b{block_y}"], categories=order)
    both[f"b{block_x}"] = pd.Categorical(both[f"b{block_x}"], categories=order)
    return pd.crosstab(both[f"b{block_y}"], both[f"b{block_x}"]).reindex(
        index=order, columns=order, fill_value=0
    )


def make_dbm_heatmap(
    out_path="../figures/best_model_raw_heatmap.png",
    include_models=None,
    exclude_models=None,
    procedural_models=None,
    rule_based_models=None,
):
    d = load_best_dbm_fits(
        include_models=include_models,
        exclude_models=exclude_models,
        procedural_models=procedural_models,
        rule_based_models=rule_based_models,
    )

    panels = [
        (0, 0, 1, "relearn", "Exp 1 - Relearn"),
        (1, 0, 1, "new_learn", "Exp 1 - New Learn"),
        (0, 1, 2, "relearn", "Exp 2 - Relearn"),
        (1, 1, 2, "new_learn", "Exp 2 - New Learn"),
    ]

    if procedural_models is None and rule_based_models is None:
        d = d.copy()
        model_abbrev = {
            "nll_rand_guess": "GS",
            "nll_bias_guess": "GS",
            "nll_glc_0": "PR",
            "nll_glc_1": "PR",
            "nll_unix_0": "UX",
            "nll_unix_1": "UX",
            "nll_uniy_0": "UX",
            "nll_uniy_1": "UX",
            "nll_gcc_eq_0": "CC",
            "nll_gcc_eq_1": "CC",
            "nll_gcc_eq_2": "CC",
            "nll_gcc_eq_3": "CC",
        }
        d["best_model_raw"] = d["best_model"].map(model_abbrev)
        value_col = "best_model_raw"
        order = ["GS", "PR", "UX", "CC"]
        labels = order
        figsize = (10, 8)
    else:
        value_col = "best_model_class"
        order = ["procedural", "rule-based", "guessing"]
        labels = ["PR", "RB", "GS"]
        figsize = (10, 8)

    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=figsize)

    for r, c, exp, cond, title in panels:
        counts = block_cross_counts(
            d,
            experiment=exp,
            condition=cond,
            block_x=2,
            block_y=6,
            value_col=value_col,
            order=order,
        )
        ax[r, c].imshow(
            counts.values,
            aspect="equal",
            cmap="Greys",
            vmin=0,
            vmax=max(1, counts.values.max() + 3),
        )
        ax[r, c].set_xticks(range(len(counts.columns)))
        ax[r, c].set_yticks(range(len(counts.index)))
        label_rotation = 45 if len(labels) > 6 else 0
        label_ha = "right" if len(labels) > 6 else "center"
        ax[r, c].set_xticklabels(
            labels, rotation=label_rotation, ha=label_ha, fontsize=10
        )
        ax[r, c].set_yticklabels(labels, rotation=0, fontsize=10)
        ax[r, c].set_xlabel("Learn", fontsize=12)
        ax[r, c].set_ylabel("Test", fontsize=12)
        ax[r, c].set_title(title, fontsize=12)

        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                ax[r, c].text(
                    j,
                    i,
                    str(counts.iat[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", default="../figures/best_model_raw_heatmap.png")
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
    parser.add_argument(
        "--procedural-models",
        default=None,
        help="Comma-separated exact DBM model names to treat as procedural.",
    )
    parser.add_argument(
        "--rule-based-models",
        default=None,
        help="Comma-separated exact DBM model names to treat as rule-based.",
    )
    args = parser.parse_args()

    include_models = None
    exclude_models = None
    procedural_models = None
    rule_based_models = None
    if args.include_models is not None and args.include_models != "":
        include_models = [
            model.strip() for model in args.include_models.split(",") if model.strip()
        ]
    if args.exclude_models is not None and args.exclude_models != "":
        exclude_models = [
            model.strip() for model in args.exclude_models.split(",") if model.strip()
        ]
    if args.procedural_models is not None and args.procedural_models != "":
        procedural_models = [
            model.strip() for model in args.procedural_models.split(",") if model.strip()
        ]
    if args.rule_based_models is not None and args.rule_based_models != "":
        rule_based_models = [
            model.strip() for model in args.rule_based_models.split(",") if model.strip()
        ]

    make_dbm_heatmap(
        out_path=args.out_path,
        include_models=include_models,
        exclude_models=exclude_models,
        procedural_models=procedural_models,
        rule_based_models=rule_based_models,
    )
