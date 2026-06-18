import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dbm_results import load_best_dbm_fits


def get_reacquisition_counts(d, experiment, condition):
    learn = d[
        (d["experiment"] == experiment)
        & (d["condition"] == condition)
        & (d["block"] == 2)
    ][["subject", "is_procedural"]].rename(columns={"is_procedural": "b2"})

    test = d[
        (d["experiment"] == experiment)
        & (d["condition"] == condition)
        & (d["block"] == 6)
    ][["subject", "is_procedural"]].rename(columns={"is_procedural": "b6"})

    both = pd.merge(learn, test, on="subject", how="inner")
    total = int(both["b2"].sum())
    success = int((both["b2"] & both["b6"]).sum())
    return success, total


def plot_pair(theta1, theta2, delta, row, axs, label):
    ci = np.percentile(delta, [2.5, 97.5])
    axs[row, 0].hist(theta1, bins=100, color="gray", density=True)
    axs[row, 0].set_title(f"{label}: Experiment 1", fontsize=12)
    axs[row, 0].set_xlabel("P(procedural at test | procedural at learning)")
    axs[row, 0].set_ylabel("Density")

    axs[row, 1].hist(theta2, bins=100, color="gray", density=True)
    axs[row, 1].set_title(f"{label}: Experiment 2", fontsize=12)
    axs[row, 1].set_xlabel("P(procedural at test | procedural at learning)")
    axs[row, 1].set_ylabel("Density")

    axs[row, 2].hist(delta, bins=100, color="gray", density=True)
    axs[row, 2].axvline(0, color="black", linestyle="--")
    axs[row, 2].axvline(ci[0], color="red", linestyle=":")
    axs[row, 2].axvline(ci[1], color="red", linestyle=":")
    axs[row, 2].set_title(f"{label}: Difference (Exp 1 - Exp 2)", fontsize=12)
    axs[row, 2].set_xlabel("Difference in reacquisition probability")
    axs[row, 2].set_ylabel("Density")

    print(
        f"{label}: mean delta={delta.mean():.3f}, "
        f"95% CI={ci[0]:.3f} to {ci[1]:.3f}, P(delta > 0)={(delta > 0).mean():.3f}"
    )


def make_bayes_post_figure(
    out_path="../figures/bayesian_comparison.png",
    include_models=None,
    exclude_models=None,
    procedural_models=None,
    samples=100000,
    seed=462,
):
    np.random.seed(seed)
    d = load_best_dbm_fits(
        include_models=include_models,
        exclude_models=exclude_models,
        procedural_models=procedural_models,
    )

    counts = {
        (1, "relearn"): get_reacquisition_counts(d, 1, "relearn"),
        (2, "relearn"): get_reacquisition_counts(d, 2, "relearn"),
        (1, "new_learn"): get_reacquisition_counts(d, 1, "new_learn"),
        (2, "new_learn"): get_reacquisition_counts(d, 2, "new_learn"),
    }

    for (experiment, condition), (success, total) in counts.items():
        print(f"Exp {experiment} {condition}: procedural reacquisition = {success}/{total}")

    theta1_relearn = np.random.beta(
        counts[(1, "relearn")][0] + 1,
        counts[(1, "relearn")][1] - counts[(1, "relearn")][0] + 1,
        samples,
    )
    theta2_relearn = np.random.beta(
        counts[(2, "relearn")][0] + 1,
        counts[(2, "relearn")][1] - counts[(2, "relearn")][0] + 1,
        samples,
    )
    theta1_new = np.random.beta(
        counts[(1, "new_learn")][0] + 1,
        counts[(1, "new_learn")][1] - counts[(1, "new_learn")][0] + 1,
        samples,
    )
    theta2_new = np.random.beta(
        counts[(2, "new_learn")][0] + 1,
        counts[(2, "new_learn")][1] - counts[(2, "new_learn")][0] + 1,
        samples,
    )

    delta_relearn = theta1_relearn - theta2_relearn
    delta_new = theta1_new - theta2_new
    delta_exp1 = theta1_relearn - theta1_new
    delta_exp2 = theta2_relearn - theta2_new

    fig, axs = plt.subplots(3, 3, figsize=(15, 8))
    plt.subplots_adjust(hspace=0.75, wspace=0.4)

    plot_pair(theta1_relearn, theta2_relearn, delta_relearn, 0, axs, "Relearn")
    plot_pair(theta1_new, theta2_new, delta_new, 1, axs, "New Learn")

    for col, delta, title in [
        (0, delta_exp1, "Experiment 1"),
        (1, delta_exp2, "Experiment 2"),
    ]:
        ci = np.percentile(delta, [2.5, 97.5])
        axs[2, col].hist(delta, bins=100, color="gray", density=True)
        axs[2, col].axvline(0, color="black", linestyle="--")
        axs[2, col].axvline(ci[0], color="red", linestyle=":")
        axs[2, col].axvline(ci[1], color="red", linestyle=":")
        axs[2, col].set_title(title, fontsize=12)
        axs[2, col].set_xlabel("Relearn - New Learn")
        axs[2, col].set_ylabel("Density")
        print(
            f"{title} Relearn - New Learn: mean={delta.mean():.3f}, "
            f"95% CI={ci[0]:.3f} to {ci[1]:.3f}, P(delta > 0)={(delta > 0).mean():.3f}"
        )

    axs[2, 2].axis("off")
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", default="../figures/bayesian_comparison.png")
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
        help="Comma-separated exact DBM model names counted as procedural.",
    )
    parser.add_argument("--samples", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()

    include_models = None
    exclude_models = None
    procedural_models = None
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

    make_bayes_post_figure(
        out_path=args.out_path,
        include_models=include_models,
        exclude_models=exclude_models,
        procedural_models=procedural_models,
        samples=args.samples,
        seed=args.seed,
    )
