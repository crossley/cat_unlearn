import numpy as np
import pandas as pd
from util_func_dbm import *

# local stimulus generator (uniform within ellipse; no pygame dependency)
def make_cat_trials(n_per_cat):
    var = 100
    corr = 0.9
    sigma = np.sqrt(var)

    theta = np.arctan(1)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    std_major = sigma * np.sqrt(1 + corr)
    std_minor = sigma * np.sqrt(1 - corr)

    rA = np.sqrt(np.random.uniform(0, 9, n_per_cat))
    angA = np.random.uniform(0, 2 * np.pi, n_per_cat)
    xA = rA * np.cos(angA) * std_major
    yA = rA * np.sin(angA) * std_minor
    ptsA = np.dot(rotation_matrix, np.vstack([xA, yA]))
    ptsA[0, :] += 40
    ptsA[1, :] += 60

    rB = np.sqrt(np.random.uniform(0, 9, n_per_cat))
    angB = np.random.uniform(0, 2 * np.pi, n_per_cat)
    xB = rB * np.cos(angB) * std_major
    yB = rB * np.sin(angB) * std_minor
    ptsB = np.dot(rotation_matrix, np.vstack([xB, yB]))
    ptsB[0, :] += 60
    ptsB[1, :] += 40

    x = np.concatenate([ptsA[0, :], ptsB[0, :]])
    y = np.concatenate([ptsA[1, :], ptsB[1, :]])
    cat = np.concatenate([np.zeros(n_per_cat, dtype=int), np.ones(n_per_cat, dtype=int)])

    p = np.random.permutation(2 * n_per_cat)
    x = x[p]
    y = y[p]
    cat = cat[p]

    return x, y, cat

if __name__ == "__main__":
    np.random.seed(462)

    # settings
    n_trials = 100
    n_per_cat = 50
    n_reps = 2
    z_limit = 3

    # parameter grids
    glc_slope_grid = np.linspace(-3, 3, 3)
    glc_intercept_grid = np.linspace(0, 100, 3)
    glc_noise_grid = np.array([2.5, 5.0, 10.0])

    gcc_xc_grid = np.linspace(35, 65, 3)
    gcc_yc_grid = np.linspace(35, 65, 3)
    gcc_noise_grid = np.array([2.5, 5.0, 10.0])

    # use fixed generating sides; fit both side variants later
    gen_side_glc = 0
    gen_side_gcc = 0

    # fit setup
    models = [
        nll_glc, nll_glc,
        nll_gcc_eq, nll_gcc_eq, nll_gcc_eq, nll_gcc_eq
    ]
    fit_side = [0, 1, 0, 1, 2, 3]
    k = [3, 3, 3, 3, 3, 3]
    model_names = [
        "nll_glc_0", "nll_glc_1",
        "nll_gcc_eq_0", "nll_gcc_eq_1", "nll_gcc_eq_2", "nll_gcc_eq_3"
    ]
    model_family_map = {
        "nll_glc": "GLC",
        "nll_gcc_eq": "GCC_eq",
        "nll_unix": "UNIX",
        "nll_uniy": "UNIY",
        "nll_rand_guess": "RAND_GUESS",
        "nll_bias_guess": "BIAS_GUESS"
    }

    rec = []

    # generate from GLC
    for slope in glc_slope_grid:
        for intercept in glc_intercept_grid:
            for noise in glc_noise_grid:
                for rep in range(n_reps):
                    a1 = -slope / np.sqrt(1 + slope ** 2)
                    a2 = np.sqrt(1 - a1 ** 2)
                    b = -intercept * a2

                    x, y, cat = make_cat_trials(n_per_cat)
                    resp0 = np.zeros(n_trials, dtype=int)

                    _, _, _, resp = val_glc((a1, b, noise), z_limit, cat, x, y, resp0, gen_side_glc)
                    resp = np.asarray(resp).reshape(-1)

                    ds = pd.DataFrame({
                        "cat": cat,
                        "x": x,
                        "y": y,
                        "resp": resp,
                        "condition": "sim",
                        "subject": "sub"
                    })

                    fit = fit_dbm(ds, models, fit_side, k, n_trials, model_names)
                    fit_bic = fit.groupby("model", as_index=False)["bic"].min()
                    best_model = fit_bic.loc[fit_bic["bic"].idxmin(), "model"]
                    model_parts = best_model.split("_")
                    model_key = "_".join(model_parts[:-1])
                    if model_key not in model_family_map:
                        raise ValueError(f"Unknown model family for best_model={best_model}")
                    recovered_family = model_family_map[model_key]
                    recovered_side = int(model_parts[-1]) if model_parts[-1].isdigit() else np.nan
                    success_family = int(recovered_family == "GLC")
                    success_strict = int((recovered_family == "GLC") and (recovered_side == gen_side_glc))

                    rec.append({
                        "true_family": "GLC",
                        "true_side": gen_side_glc,
                        "true_slope": slope,
                        "true_intercept": intercept,
                        "true_a1": a1,
                        "true_b": b,
                        "true_xc": np.nan,
                        "true_yc": np.nan,
                        "true_noise": noise,
                        "rep": rep,
                        "best_model": best_model,
                        "recovered_family": recovered_family,
                        "recovered_side": recovered_side,
                        "success_family": success_family,
                        "success_strict": success_strict,
                        "success": success_family
                    })

    # generate from GCC_eq
    for xc in gcc_xc_grid:
        for yc in gcc_yc_grid:
            for noise in gcc_noise_grid:
                for rep in range(n_reps):
                    x, y, cat = make_cat_trials(n_per_cat)
                    resp0 = np.zeros(n_trials, dtype=int)

                    _, _, _, resp = val_gcc_eq((xc, yc, noise), z_limit, cat, x, y, resp0, gen_side_gcc)
                    resp = np.asarray(resp).reshape(-1)

                    ds = pd.DataFrame({
                        "cat": cat,
                        "x": x,
                        "y": y,
                        "resp": resp,
                        "condition": "sim",
                        "subject": "sub"
                    })

                    fit = fit_dbm(ds, models, fit_side, k, n_trials, model_names)
                    fit_bic = fit.groupby("model", as_index=False)["bic"].min()
                    best_model = fit_bic.loc[fit_bic["bic"].idxmin(), "model"]
                    model_parts = best_model.split("_")
                    model_key = "_".join(model_parts[:-1])
                    if model_key not in model_family_map:
                        raise ValueError(f"Unknown model family for best_model={best_model}")
                    recovered_family = model_family_map[model_key]
                    recovered_side = int(model_parts[-1]) if model_parts[-1].isdigit() else np.nan
                    success_family = int(recovered_family == "GCC_eq")
                    success_strict = int((recovered_family == "GCC_eq") and (recovered_side == gen_side_gcc))

                    rec.append({
                        "true_family": "GCC_eq",
                        "true_side": gen_side_gcc,
                        "true_slope": np.nan,
                        "true_intercept": np.nan,
                        "true_a1": np.nan,
                        "true_b": np.nan,
                        "true_xc": xc,
                        "true_yc": yc,
                        "true_noise": noise,
                        "rep": rep,
                        "best_model": best_model,
                        "recovered_family": recovered_family,
                        "recovered_side": recovered_side,
                        "success_family": success_family,
                        "success_strict": success_strict,
                        "success": success_family
                    })

    res = pd.DataFrame(rec)

    # confusion / recovery matrix
    cm_counts = pd.crosstab(res["true_family"], res["recovered_family"])
    cm_props = pd.crosstab(res["true_family"], res["recovered_family"], normalize="index")

    print("\nConfusion matrix (counts):")
    print(cm_counts)

    print("\nConfusion matrix (row proportions):")
    print(cm_props)

    print("\nRecovery rates by true family:")
    print(
        res.groupby("true_family")[["success_family", "success_strict"]]
           .mean()
    )

    # characterize parameter values for successful vs failed recovery
    print("\nGLC success/failure summary:")
    print(
        res.loc[res["true_family"] == "GLC"]
           .groupby("success_family")[["true_slope", "true_intercept", "true_noise"]]
           .agg(["mean", "std"])
    )

    print("\nGCC_eq success/failure summary:")
    print(
        res.loc[res["true_family"] == "GCC_eq"]
           .groupby("success_family")[["true_xc", "true_yc", "true_noise"]]
           .agg(["mean", "std"])
    )

    # save outputs
    res.to_csv("../dbm_fits/dbm_recovery_results.csv", index=False)
    cm_counts.to_csv("../dbm_fits/dbm_recovery_confusion_counts.csv")
    cm_props.to_csv("../dbm_fits/dbm_recovery_confusion_props.csv")
