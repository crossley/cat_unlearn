import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from util_func_wrangle import get_cl_df, get_dbm_df


def make_fig_acc_all():

    d = get_cl_df()
    dp = d.copy()

    ddd = get_dbm_df()
    if ddd is None:
        return

    keep_cols = ["experiment", "condition", "subject"]
    d = d.merge(ddd[keep_cols].drop_duplicates(), on=keep_cols,
                how="inner").copy()

    block_size = 25
    d["block"] = d.groupby(["condition", "subject"]).cumcount() // block_size
    d["condition"] = d["condition"].astype("category")
    d = d.groupby(["experiment", "condition", "subject", "phase", "block"],
                  observed=True)["acc"].mean().reset_index()
    d1 = d[d["experiment"] == 1].copy()
    d2 = d[d["experiment"] == 2].copy()

    relearn_color = "#B35C44"
    new_learn_color = "#4C7899"
    condition_colors = {
        "relearn": relearn_color,
        "new_learn": new_learn_color,
    }
    sns.set_palette(
        sns.color_palette(
            [condition_colors[c] for c in d["condition"].cat.categories]))

    d1["condition"] = d1["condition"].map({
        "relearn": "Relearn",
        "new_learn": "New Learn"
    })

    d2["condition"] = d2["condition"].map({
        "relearn": "Relearn",
        "new_learn": "New Learn"
    })

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))

    for i, phase in enumerate(["Learn", "Intervention", "Test"]):

        if i == 0:
            leg = True
        else:
            leg = False

        d11 = d1[d1["phase"] == phase].copy()
        d22 = d2[d2["phase"] == phase].copy()

        start = len(ax[0, 0].collections)
        sns.lineplot(data=d11,
                     x="block",
                     y="acc",
                     hue="condition",
                     err_style="bars",
                     errorbar="se",
                     marker="o",
                     legend=leg,
                     ax=ax[0, 0])
        for coll in ax[0, 0].collections[start:]:
            coll.set_alpha(0.8)

        start = len(ax[0, 1].collections)
        sns.lineplot(data=d22,
                     x="block",
                     y="acc",
                     hue="condition",
                     err_style="bars",
                     errorbar="se",
                     marker="o",
                     legend=leg,
                     ax=ax[0, 1])
        for coll in ax[0, 1].collections[start:]:
            coll.set_alpha(0.8)

    ax[0, 0].set_title("Experiment 1")
    ax[0, 1].set_title("Experiment 2")

    for axx in ax.flatten():
        axx.set_xlabel("Block", fontsize=12)
        axx.set_ylabel("Accuracy", fontsize=12)
        axx.set_ylim(.4, .9)
        sns.move_legend(axx, "upper right", title=None, frameon=False)

    plt.tight_layout()
    plt.savefig("../figures/subjects_accuracy_all.png")
    plt.close()


def make_fig_acc_talk():

    d = get_cl_df()
    dp = d.copy()

    block_size = 25
    d["block"] = d.groupby(["condition", "subject"]).cumcount() // block_size
    d["condition"] = d["condition"].astype("category")
    d = d.groupby(["experiment", "condition", "subject", "phase", "block"],
                  observed=True)["acc"].mean().reset_index()
    d1 = d[d["experiment"] == 1].copy()
    d2 = d[d["experiment"] == 2].copy()

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 6))
    d1["condition"] = d1["condition"].map({
        "relearn": "Relearn",
        "new_learn": "New Learn"
    })
    d1["condition"] = pd.Categorical(d1["condition"],
                                     categories=["Relearn", "New Learn"],
                                     ordered=True)
    d1["block"] = d1["block"] + 1
    sns.lineplot(data=d1,
                 x="block",
                 y="acc",
                 hue="condition",
                 marker="o",
                 ax=ax[0, 0])
    ax[0, 0].axvline(x=12.5, color='gray', linestyle='--')
    ax[0, 0].axvline(x=24.5, color='gray', linestyle='--')
    ax[0, 0].set_title(
        r"Experiment 1: Random Feedback Intervention with $\bf{Verbal~Instruction}$",
        fontsize=14)
    ax[0, 0].set_xlabel("Block", fontsize=14)
    ax[0, 0].set_ylabel("Accuracy", fontsize=14)
    ax[0, 0].set_xlim(0, 37)
    ax[0, 0].set_ylim(0.3, 1.1)
    ax[0, 0].set_yticks(np.arange(0.2, 1.1, 0.2))
    ax[0, 0].get_legend().set_title("")
    ax[0, 0].legend(loc='upper left')
    ax_inset_1 = ax[0, 0].inset_axes([0.1, 0.05, 0.15, 0.2])
    ax_inset_2 = ax[0, 0].inset_axes([0.425, 0.05, 0.15, 0.2])
    ax_inset_3 = ax[0, 0].inset_axes([0.76, 0.74, 0.15, 0.2])
    ax_inset_4 = ax[0, 0].inset_axes([0.76, 0.05, 0.15, 0.2])
    dp["condition"] = dp["condition"].map({
        "relearn": "Relearn",
        "new_learn": "New Learn"
    })
    sns.scatterplot(data=dp[dp["condition"] == "Relearn"].iloc[0:300, :],
                    x="x",
                    y="y",
                    hue="cat",
                    ax=ax_inset_1,
                    legend=False)
    sns.scatterplot(data=dp[dp["condition"] == "Relearn"].iloc[300:600, :],
                    x="x",
                    y="y",
                    hue="cat",
                    ax=ax_inset_2,
                    legend=False)
    sns.scatterplot(data=dp[dp["condition"] == "Relearn"].iloc[600:899, :],
                    x="x",
                    y="y",
                    hue="cat",
                    ax=ax_inset_3,
                    legend=False)
    sns.scatterplot(data=dp[dp["condition"] == "New Learn"].iloc[600:899, :],
                    x="x",
                    y="y",
                    hue="cat",
                    ax=ax_inset_4,
                    legend=False)
    ax_inset_1.set_title("Learn", fontsize=14)
    ax_inset_2.set_title("Intervention", fontsize=14)
    ax_inset_3.set_title("Test: Relearn", fontsize=14)
    ax_inset_4.set_title("Test: New Learn", fontsize=14)
    [
        x.set_xticks([])
        for x in [ax_inset_1, ax_inset_2, ax_inset_3, ax_inset_4]
    ]
    [
        y.set_yticks([])
        for y in [ax_inset_1, ax_inset_2, ax_inset_3, ax_inset_4]
    ]
    [
        x.set_xlabel("")
        for x in [ax_inset_1, ax_inset_2, ax_inset_3, ax_inset_4]
    ]
    [
        y.set_ylabel("")
        for y in [ax_inset_1, ax_inset_2, ax_inset_3, ax_inset_4]
    ]
    ax_inset_3.spines['top'].set_color('C0')
    ax_inset_3.spines['bottom'].set_color('C0')
    ax_inset_3.spines['left'].set_color('C0')
    ax_inset_3.spines['right'].set_color('C0')
    ax_inset_3.spines['top'].set_linewidth(2)
    ax_inset_3.spines['bottom'].set_linewidth(2)
    ax_inset_3.spines['left'].set_linewidth(2)
    ax_inset_3.spines['right'].set_linewidth(2)
    ax_inset_4.spines['top'].set_color('C1')
    ax_inset_4.spines['bottom'].set_color('C1')
    ax_inset_4.spines['left'].set_color('C1')
    ax_inset_4.spines['right'].set_color('C1')
    ax_inset_4.spines['top'].set_linewidth(2)
    ax_inset_4.spines['bottom'].set_linewidth(2)
    ax_inset_4.spines['left'].set_linewidth(2)
    ax_inset_4.spines['right'].set_linewidth(2)
    plt.tight_layout()
    plt.savefig("../figures/subjects_accuracy_talk_exp_1.pdf")
    plt.close()

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 6))
    d2["condition"] = d2["condition"].map({
        "relearn": "Relearn",
        "new_learn": "New Learn"
    })
    d2["condition"] = pd.Categorical(d2["condition"],
                                     categories=["Relearn", "New Learn"],
                                     ordered=True)
    d2["block"] = d2["block"] + 1
    sns.lineplot(data=d2,
                 x="block",
                 y="acc",
                 hue="condition",
                 marker="o",
                 ax=ax[0, 0])
    ax[0, 0].axvline(x=12.5, color='gray', linestyle='--')
    ax[0, 0].axvline(x=24.5, color='gray', linestyle='--')
    ax[0, 0].set_title(
        r"Experiment 2: Mixed Feedback Intervention with $\bf{Verbal~Instruction}$",
        fontsize=14)
    ax[0, 0].set_xlabel("Block", fontsize=14)
    ax[0, 0].set_ylabel("Accuracy", fontsize=14)
    ax[0, 0].set_xlim(0, 37)
    ax[0, 0].set_ylim(0.3, 1.1)
    ax[0, 0].set_yticks(np.arange(0.2, 1.1, 0.2))
    ax[0, 0].get_legend().set_title("")
    ax[0, 0].legend(loc='upper left')
    ax_inset_1 = ax[0, 0].inset_axes([0.1, 0.05, 0.15, 0.2])
    ax_inset_2 = ax[0, 0].inset_axes([0.425, 0.05, 0.15, 0.2])
    ax_inset_3 = ax[0, 0].inset_axes([0.76, 0.74, 0.15, 0.2])
    ax_inset_4 = ax[0, 0].inset_axes([0.76, 0.05, 0.15, 0.2])
    sns.scatterplot(data=dp[dp["condition"] == "Relearn"].iloc[0:300, :],
                    x="x",
                    y="y",
                    hue="cat",
                    ax=ax_inset_1,
                    legend=False)
    sns.scatterplot(data=dp[dp["condition"] == "Relearn"].iloc[300:600, :],
                    x="x",
                    y="y",
                    hue="cat",
                    ax=ax_inset_2,
                    legend=False)
    sns.scatterplot(data=dp[dp["condition"] == "Relearn"].iloc[600:899, :],
                    x="x",
                    y="y",
                    hue="cat",
                    ax=ax_inset_3,
                    legend=False)
    sns.scatterplot(data=dp[dp["condition"] == "New Learn"].iloc[600:899, :],
                    x="x",
                    y="y",
                    hue="cat",
                    ax=ax_inset_4,
                    legend=False)
    ax_inset_1.set_title("Learn", fontsize=14)
    ax_inset_2.set_title("Intervention", fontsize=14)
    ax_inset_3.set_title("Test: Relearn", fontsize=14)
    ax_inset_4.set_title("Test: New Learn", fontsize=14)
    [
        x.set_xticks([])
        for x in [ax_inset_1, ax_inset_2, ax_inset_3, ax_inset_4]
    ]
    [
        y.set_yticks([])
        for y in [ax_inset_1, ax_inset_2, ax_inset_3, ax_inset_4]
    ]
    [
        x.set_xlabel("")
        for x in [ax_inset_1, ax_inset_2, ax_inset_3, ax_inset_4]
    ]
    [
        y.set_ylabel("")
        for y in [ax_inset_1, ax_inset_2, ax_inset_3, ax_inset_4]
    ]
    ax_inset_3.spines['top'].set_color('C0')
    ax_inset_3.spines['bottom'].set_color('C0')
    ax_inset_3.spines['left'].set_color('C0')
    ax_inset_3.spines['right'].set_color('C0')
    ax_inset_3.spines['top'].set_linewidth(2)
    ax_inset_3.spines['bottom'].set_linewidth(2)
    ax_inset_3.spines['left'].set_linewidth(2)
    ax_inset_3.spines['right'].set_linewidth(2)
    ax_inset_4.spines['top'].set_color('C1')
    ax_inset_4.spines['bottom'].set_color('C1')
    ax_inset_4.spines['left'].set_color('C1')
    ax_inset_4.spines['right'].set_color('C1')
    ax_inset_4.spines['top'].set_linewidth(2)
    ax_inset_4.spines['bottom'].set_linewidth(2)
    ax_inset_4.spines['left'].set_linewidth(2)
    ax_inset_4.spines['right'].set_linewidth(2)
    plt.tight_layout()
    plt.savefig("../figures/subjects_accuracy_talk_exp_2.pdf")
    plt.close()


def make_fig_dbm():

    if os.path.exists("../dbm_fits/dbm_results.csv"):
        dbm = pd.read_csv("../dbm_fits/dbm_results.csv")
        dbm = dbm[dbm["block"] != "block"].copy()
        dbm["block"] = dbm["block"].astype(int)
        dbm["bic"] = dbm["bic"].astype(float)
        dbm["experiment"] = dbm["experiment"].astype(int)
    else:
        print("DBM results file not found. Please run fit_dbm_top() first.")

    def assign_best_model(x):
        model = x["model"].to_numpy()
        bic = x["bic"].to_numpy()
        best_model = np.unique(model[bic == bic.min()])[0]
        x["best_model"] = best_model
        return x

    dbm = dbm.groupby(["experiment", "condition", "subject", "block"
                       ]).apply(assign_best_model).reset_index(drop=True)

    d = get_cl_df()

    dd = dbm.loc[dbm["model"] == dbm["best_model"]]
    ddd = dd[["experiment", "condition", "subject", "block",
              "best_model"]].drop_duplicates()
    ddd.loc[ddd["best_model"] == "nll_rand_guess",
            "best_model_class"] = "guessing"
    ddd.loc[ddd["best_model"] == "nll_bias_guess",
            "best_model_class"] = "guessing"
    ddd.loc[ddd["best_model"] == "nll_unix_0",
            "best_model_class"] = "rule-based"
    ddd.loc[ddd["best_model"] == "nll_unix_1",
            "best_model_class"] = "rule-based"
    ddd.loc[ddd["best_model"] == "nll_uniy_0",
            "best_model_class"] = "rule-based"
    ddd.loc[ddd["best_model"] == "nll_uniy_1",
            "best_model_class"] = "rule-based"
    ddd.loc[ddd["best_model"] == "nll_glc_0",
            "best_model_class"] = "procedural"
    ddd.loc[ddd["best_model"] == "nll_glc_1",
            "best_model_class"] = "procedural"
    ddd["best_model_class"] = ddd["best_model_class"].astype("category")
    ddd['block'] = ddd['block'].astype("category")
    ddd = ddd.reset_index(drop=True)

    # exclude subjects best fit by guessing in the last learning block
    exc_subs_learn = ddd[(ddd["block"] == 2)
                         & (ddd["best_model_class"] == "guessing")][[
                             "experiment", "condition", "subject"
                         ]].drop_duplicates()
    ddd = ddd.merge(exc_subs_learn.assign(exclude_subject=True),
                    on=["experiment", "condition", "subject"],
                    how="left")
    ddd = ddd[ddd["exclude_subject"] != True].drop(columns="exclude_subject")

    print(
        ddd.groupby(["experiment", "condition", "block",
                     "best_model_class"])["subject"].nunique())

    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 6))

    class_order = ["procedural", "rule-based", "guessing"]

    def block_cross_counts(ddd, experiment, condition, block_x, block_y):

        d_x = ddd[(ddd["experiment"] == experiment)
                  & (ddd["condition"] == condition)
                  & (ddd["block"] == block_x)][[
                      "subject", "best_model_class"
                  ]].rename(columns={"best_model_class": f"b{block_x}"})

        d_y = ddd[(ddd["experiment"] == experiment)
                  & (ddd["condition"] == condition)
                  & (ddd["block"] == block_y)][[
                      "subject", "best_model_class"
                  ]].rename(columns={"best_model_class": f"b{block_y}"})

        # only subjects present in both blocks
        both = pd.merge(d_x, d_y, on="subject", how="inner")
        if both.empty:
            return pd.DataFrame(0, index=class_order, columns=class_order)

        # enforce ordering
        both[f"b{block_y}"] = pd.Categorical(both[f"b{block_y}"],
                                             categories=class_order)
        both[f"b{block_x}"] = pd.Categorical(both[f"b{block_x}"],
                                             categories=class_order)

        ct = pd.crosstab(both[f"b{block_y}"],
                         both[f"b{block_x}"]).reindex(index=class_order,
                                                      columns=class_order,
                                                      fill_value=0)
        return ct

    def draw_heatmap(ax, counts, title, xlabel, ylabel):
        im = ax.imshow(counts.values,
                       aspect="equal",
                       cmap="Greys",
                       vmin=1,
                       vmax=counts.values.max() + 7)
        # ticks & labels
        ax.set_xticks(range(len(counts.columns)))
        ax.set_yticks(range(len(counts.index)))
        ax.set_xticklabels(['PR', 'RB', 'GS'], rotation=0, fontsize=12)
        ax.set_yticklabels(['PR', 'RB', 'GS'], rotation=0, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)

        # annotate cell counts
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                ax.text(j, i, str(counts.iat[i, j]), ha="center", va="center")

    # (row, col, experiment, condition, title, xlabel, ylabel)
    panels = [
        (0, 0, 1, "relearn", "Exp 1 — Relearn", "Learn", "Test"),
        (1, 0, 1, "new_learn", "Exp 1 — New Learn", "Learn", "Test"),
        (0, 1, 2, "relearn", "Exp 2 — Relearn", "Learn", "Test"),
        (1, 1, 2, "new_learn", "Exp 2 — New Learn", "Learn", "Test"),
    ]

    for r, c, exp, cond, title, xlab, ylab in panels:
        counts = block_cross_counts(ddd, exp, cond, block_x=2, block_y=6)
        draw_heatmap(ax[r, c], counts, title, xlab, ylab)

    fig.tight_layout()
    plt.savefig("../figures/best_model_class_heatmap.png", dpi=300)
    plt.close()


def make_fig_dbm_bayes():

    ddd = get_dbm_df()
    if ddd is None:
        return

    def get_procedural_reacquisition_counts(ddd, experiment, condition):
        learn = ddd[(ddd["experiment"] == experiment)
                    & (ddd["condition"] == condition)
                    & (ddd["block"] == 2)][[
                        "subject", "best_model_class"
                    ]].rename(columns={"best_model_class": "b2"})
        test = ddd[(ddd["experiment"] == experiment)
                   & (ddd["condition"] == condition)
                   & (ddd["block"] == 6)][[
                       "subject", "best_model_class"
                   ]].rename(columns={"best_model_class": "b6"})
        both = pd.merge(learn, test, on="subject", how="inner")
        total = ((both["b2"] == "procedural")).sum()
        success = (((both["b2"] == "procedural")
                    & (both["b6"] == "procedural"))).sum()
        return int(success), int(total)

    def get_posterior_diff(exp1_success,
                           exp1_total,
                           exp2_success,
                           exp2_total,
                           samples=100000):
        theta1 = np.random.beta(exp1_success + 1,
                                exp1_total - exp1_success + 1, samples)
        theta2 = np.random.beta(exp2_success + 1,
                                exp2_total - exp2_success + 1, samples)
        delta = theta1 - theta2
        return theta1, theta2, delta

    def plot_bayesian_comparison(theta1, theta2, delta, row, axs,
                                 condition_label):
        ci = np.percentile(delta, [2.5, 97.5])
        prob_exp1_greater = (delta > 0).mean()

        axs[row, 0].hist(theta1, bins=100, color='gray', density=True)
        axs[row, 0].set_title(f'{condition_label}: Experiment 1', fontsize=12)
        axs[row,
            0].set_xlabel('P(procedural at test | procedural at learning)',
                          fontsize=12)
        axs[row, 0].set_ylabel('Density', fontsize=12)

        axs[row, 1].hist(theta2, bins=100, color='gray', density=True)
        axs[row, 1].set_title(f'{condition_label}: Experiment 2', fontsize=12)
        axs[row,
            1].set_xlabel('P(procedural at test | procedural at learning)',
                          fontsize=12)
        axs[row, 1].set_ylabel('Density', fontsize=12)

        axs[row, 2].hist(delta, bins=100, color='gray', density=True)
        axs[row, 2].axvline(0, color='black', linestyle='--', label='Δ = 0')
        axs[row, 2].axvline(ci[0], color='red', linestyle=':', label='95% CI')
        axs[row, 2].axvline(ci[1], color='red', linestyle=':')
        axs[row, 2].set_title(f'{condition_label}: Difference (Exp 1 - Exp 2)',
                              fontsize=12)
        axs[row, 2].set_xlabel('Difference in reacquisition probability',
                               fontsize=12)
        axs[row, 2].set_ylabel('Density', fontsize=12)
        axs[row, 2].legend()

        print(
            f"{condition_label} — Mean Δ (Exp 1 − Exp 2): {delta.mean():.3f}")
        print(
            f"{condition_label} — 95% CI for Δ (Exp 1 − Exp 2): {ci[0]:.3f} to {ci[1]:.3f}"
        )
        print(f"{condition_label} — P(θ₁ > θ₂) = {prob_exp1_greater:.3f}")

    count_lookup = {
        (1, "relearn"): get_procedural_reacquisition_counts(ddd, 1, "relearn"),
        (2, "relearn"): get_procedural_reacquisition_counts(ddd, 2, "relearn"),
        (1, "new_learn"):
        get_procedural_reacquisition_counts(ddd, 1, "new_learn"),
        (2, "new_learn"):
        get_procedural_reacquisition_counts(ddd, 2, "new_learn"),
    }

    for (experiment, condition), (success, total) in count_lookup.items():
        print(
            f"Exp {experiment} {condition}: procedural reacquisition = {success}/{total}"
        )

    # Posterior samples from DBM-derived counts
    theta1_relearn, theta2_relearn, delta_relearn = get_posterior_diff(
        *count_lookup[(1, "relearn")], *count_lookup[(2, "relearn")])
    theta1_new, theta2_new, delta_new = get_posterior_diff(
        *count_lookup[(1, "new_learn")], *count_lookup[(2, "new_learn")])

    # Cross-condition comparisons within experiments
    delta_exp1 = theta1_relearn - theta1_new
    delta_exp2 = theta2_relearn - theta2_new

    # Credible intervals and probabilities
    ci_exp1_relearn = np.percentile(theta1_relearn, [2.5, 97.5])
    print("Exp 1 - Relearn: mean = {:.3f}".format(theta1_relearn.mean()))
    print(
        f"Exp 1 — Relearn: 95% CI = {ci_exp1_relearn[0]:.3f} to {ci_exp1_relearn[1]:.3f}"
    )

    ci_exp1_new = np.percentile(theta1_new, [2.5, 97.5])
    print("Exp 1 - New Learn: mean = {:.3f}".format(theta1_new.mean()))
    print(
        f"Exp 1 — New Learn: 95% CI = {ci_exp1_new[0]:.3f} to {ci_exp1_new[1]:.3f}"
    )

    ci_exp2_relearn = np.percentile(theta2_relearn, [2.5, 97.5])
    print("Exp 2 - Relearn: mean = {:.3f}".format(theta2_relearn.mean()))
    print(
        f"Exp 2 — Relearn: 95% CI = {ci_exp2_relearn[0]:.3f} to {ci_exp2_relearn[1]:.3f}"
    )

    ci_exp2_new = np.percentile(theta2_new, [2.5, 97.5])
    print("Exp 2 - New Learn: mean = {:.3f}".format(theta2_new.mean()))
    print(
        f"Exp 2 — New Learn: 95% CI = {ci_exp2_new[0]:.3f} to {ci_exp2_new[1]:.3f}"
    )

    ci_exp1 = np.percentile(delta_exp1, [2.5, 97.5])
    ci_exp2 = np.percentile(delta_exp2, [2.5, 97.5])
    prob_exp1 = (delta_exp1 > 0).mean()
    prob_exp2 = (delta_exp2 > 0).mean()

    # Create the figure grid
    fig, axs = plt.subplots(3, 3, figsize=(15, 8))
    plt.subplots_adjust(hspace=0.75, wspace=0.4)

    # Row 1: Relearn condition
    plot_bayesian_comparison(theta1_relearn,
                             theta2_relearn,
                             delta_relearn,
                             row=0,
                             axs=axs,
                             condition_label='Relearn')

    # Row 2: New Learn condition
    plot_bayesian_comparison(theta1_new,
                             theta2_new,
                             delta_new,
                             row=1,
                             axs=axs,
                             condition_label='New Learn')

    # Row 3: Relearn − New Learn within each experiment
    axs[2, 0].hist(delta_exp1, bins=100, color='gray', density=True)
    axs[2, 0].axvline(0, color='black', linestyle='--')
    axs[2, 0].axvline(ci_exp1[0], color='red', linestyle=':')
    axs[2, 0].axvline(ci_exp1[1], color='red', linestyle=':')
    axs[2, 0].set_title('Experiment 1', fontsize=12)
    axs[2, 0].set_xlabel('Difference in reacquisition probability',
                         fontsize=12)
    axs[2, 0].set_ylabel('Density', fontsize=12)

    axs[2, 1].hist(delta_exp2, bins=100, color='gray', density=True)
    axs[2, 1].axvline(0, color='black', linestyle='--')
    axs[2, 1].axvline(ci_exp2[0], color='red', linestyle=':')
    axs[2, 1].axvline(ci_exp2[1], color='red', linestyle=':')
    axs[2, 1].set_title('Experiment 2', fontsize=12)
    axs[2, 1].set_xlabel('Difference in reacquisition probability',
                         fontsize=12)
    axs[2, 1].set_ylabel('Density', fontsize=12)

    axs[2, 2].axis('off')  # Empty final cell

    print(
        f"Exp 1 — Relearn − New Learn: mean = {delta_exp1.mean():.3f}, 95% CI = {ci_exp1[0]:.3f} to {ci_exp1[1]:.3f}, P(Δ > 0) = {prob_exp1:.3f}"
    )
    print(
        f"Exp 2 — Relearn − New Learn: mean = {delta_exp2.mean():.3f}, 95% CI = {ci_exp2[0]:.3f} to {ci_exp2[1]:.3f}, P(Δ > 0) = {prob_exp2:.3f}"
    )

    plt.savefig("../figures/bayesian_comparison.png", dpi=300)
    plt.close()
