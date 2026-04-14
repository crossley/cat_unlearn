import numpy as np
import pandas as pd

from util_func_wrangle import get_cl_df
from util_func_dbm import *


if __name__ == "__main__":

    np.random.seed(462)

    # settings
    block_size = 100
    n_trials = block_size
    n_reps = 2
    z_limit = 3

    # real trial-level data, same preprocessing pattern as fit_dbm_top
    d = get_cl_df()
    d["block"] = d.groupby(["condition", "subject"]).cumcount() // block_size
    d = d.loc[(d["block"] == 2) | (d["block"] == 6)]
    d = d.sort_values(["experiment", "condition", "subject", "block", "trial"]).copy()

    # fitted model table from real data
    dbm = pd.read_csv("../dbm_fits/dbm_results.csv")
    dbm["experiment"] = dbm["experiment"].astype(int)
    dbm["subject"] = dbm["subject"].astype(int)
    dbm["block"] = dbm["block"].astype(int)
    dbm["p"] = dbm["p"].astype(float)
    dbm["nll"] = dbm["nll"].astype(float)
    dbm["bic"] = dbm["bic"].astype(float)

    # best model per empirical group
    keys = ["experiment", "condition", "subject", "block"]
    bic_by_model = dbm.groupby(keys + ["model"], as_index=False)["bic"].min()
    idx = bic_by_model.groupby(keys)["bic"].idxmin()
    best = bic_by_model.loc[idx].rename(columns={"model": "best_model", "bic": "best_bic"})

    # attach winner to parameter rows, then collect parameter vectors
    best_params = (
        dbm.merge(best[keys + ["best_model"]], on=keys, how="inner")
           .loc[lambda x: x["model"] == x["best_model"], keys + ["best_model", "p"]]
           .groupby(keys + ["best_model"], as_index=False)
           .agg(best_params=("p", lambda s: tuple(s.to_numpy())))
    )
    best_fit = best.merge(best_params, on=keys + ["best_model"], how="left")

    # trial-level table aligned to groups we have best fits for
    sim_in = d.merge(best_fit[keys + ["best_model", "best_params"]], on=keys, how="inner").copy()

    # candidate fit set (match dbm_results.csv)
    models = [
        nll_rand_guess,
        nll_bias_guess,
        nll_unix,
        nll_unix,
        nll_uniy,
        nll_uniy,
        nll_glc,
        nll_glc,
        nll_gcc_eq,
        nll_gcc_eq,
        nll_gcc_eq,
        nll_gcc_eq,
    ]
    fit_side = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3]
    k = [0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    model_names = [
        "nll_rand_guess",
        "nll_bias_guess",
        "nll_unix_0",
        "nll_unix_1",
        "nll_uniy_0",
        "nll_uniy_1",
        "nll_glc_0",
        "nll_glc_1",
        "nll_gcc_eq_0",
        "nll_gcc_eq_1",
        "nll_gcc_eq_2",
        "nll_gcc_eq_3",
    ]
    model_family_map = {
        "nll_rand_guess": "guessing",
        "nll_bias_guess": "guessing",
        "nll_unix": "rule-based",
        "nll_uniy": "rule-based",
        "nll_glc": "procedural",
        "nll_gcc_eq": "procedural",
    }

    rec = []

    for rep in range(n_reps):
        for gk, gg in sim_in.groupby(keys + ["best_model", "best_params"], sort=False):

            expt, cnd, sub, blk, true_model, true_params = gk
            true_params = np.asarray(true_params, dtype=float)

            x = gg["x"].to_numpy()
            y = gg["y"].to_numpy()
            cat = gg["cat"].to_numpy()

            # same scaling used inside fit_dbm
            range_x = np.max(x) - np.min(x)
            range_y = np.max(y) - np.min(y)
            x = ((x - np.min(x)) / range_x) * 100
            y = ((y - np.min(y)) / range_y) * 100

            resp0 = np.zeros_like(cat, dtype=int)

            # generate responses using model-specific val_* functions
            if true_model == "nll_rand_guess":
                _, _, _, resp = val_rand_guess(tuple(true_params), z_limit, cat, x, y, resp0, 0)
                resp = np.asarray(resp).reshape(-1)

            elif true_model == "nll_bias_guess":
                _, _, _, resp = val_bias_guess(tuple(true_params), z_limit, cat, x, y, resp0, 0)
                resp = np.asarray(resp).reshape(-1)

            elif "nll_unix" in true_model:
                side = int(true_model.split("_")[-1])
                _, _, _, resp = val_unix(tuple(true_params), z_limit, cat, x, y, resp0, side)
                resp = np.asarray(resp).reshape(-1)

            elif "nll_uniy" in true_model:
                side = int(true_model.split("_")[-1])
                _, _, _, resp = val_uniy(tuple(true_params), z_limit, cat, x, y, resp0, side)
                resp = np.asarray(resp).reshape(-1)

            elif "nll_glc" in true_model:
                side = int(true_model.split("_")[-1])
                _, _, _, resp = val_glc(tuple(true_params), z_limit, cat, x, y, resp0, side)
                resp = np.asarray(resp).reshape(-1)

            elif "nll_gcc_eq" in true_model:
                side = int(true_model.split("_")[-1])
                _, _, _, resp = val_gcc_eq(tuple(true_params), z_limit, cat, x, y, resp0, side)
                resp = np.asarray(resp).reshape(-1)

            else:
                raise ValueError(f"Unknown true_model: {true_model}")

            ds = pd.DataFrame({
                "cat": cat,
                "x": x,
                "y": y,
                "resp": resp,
                "condition": cnd,
                "subject": sub,
            })

            fit = fit_dbm(ds, models, fit_side, k, n_trials, model_names)
            fit_bic = fit.groupby("model", as_index=False)["bic"].min()
            recovered_model = fit_bic.loc[fit_bic["bic"].idxmin(), "model"]

            true_parts = true_model.split("_")
            rec_parts = recovered_model.split("_")
            true_key = "_".join(true_parts[:-1]) if true_parts[-1].isdigit() else true_model
            rec_key = "_".join(rec_parts[:-1]) if rec_parts[-1].isdigit() else recovered_model

            rec.append({
                "experiment": expt,
                "condition": cnd,
                "subject": sub,
                "block": blk,
                "rep": rep,
                "true_model": true_model,
                "recovered_model": recovered_model,
                "true_family": model_family_map[true_key],
                "recovered_family": model_family_map[rec_key],
                "success_strict": int(recovered_model == true_model),
                "success_family": int(model_family_map[rec_key] == model_family_map[true_key]),
            })

    rec = pd.DataFrame(rec)

    cm_family_counts = pd.crosstab(rec["true_family"], rec["recovered_family"])
    cm_family_props = pd.crosstab(rec["true_family"], rec["recovered_family"], normalize="index")
    cm_model_counts = pd.crosstab(rec["true_model"], rec["recovered_model"])
    cm_model_props = pd.crosstab(rec["true_model"], rec["recovered_model"], normalize="index")

    rec.to_csv("../dbm_fits/dbm_recovery_empirical_results.csv", index=False)
    cm_family_counts.to_csv("../dbm_fits/dbm_recovery_empirical_family_counts.csv")
    cm_family_props.to_csv("../dbm_fits/dbm_recovery_empirical_family_props.csv")
    cm_model_counts.to_csv("../dbm_fits/dbm_recovery_empirical_model_counts.csv")
    cm_model_props.to_csv("../dbm_fits/dbm_recovery_empirical_model_props.csv")
