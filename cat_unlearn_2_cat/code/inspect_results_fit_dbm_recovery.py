import argparse
import os

import numpy as np
import pandas as pd

from util_func_wrangle import get_cl_df
from util_func_dbm import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=2)
    parser.add_argument("--block", type=int, default=None)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=462)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="../dbm_fits/recovery_chunks")
    args = parser.parse_args()

    if args.num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if args.chunk_index < 0 or args.chunk_index >= args.num_chunks:
        raise ValueError("chunk_index must be in [0, num_chunks)")

    # settings
    block_size = 100
    n_trials = block_size
    n_reps = args.n_reps
    z_limit = 3

    # real trial-level data, same preprocessing pattern as fit_dbm_top
    d = get_cl_df()
    d["block"] = d["orig_trial_index"] // block_size
    d = d.loc[d["block"].isin([0, 2, 6])]
    if args.block is not None:
        d = d.loc[d["block"] == args.block]
    d = d.sort_values(
        ["experiment", "condition", "subject", "block", "orig_trial_index"]
    ).copy()

    # fitted model table from real data
    dbm = pd.read_csv("../dbm_fits/dbm_results.csv")
    dbm["experiment"] = dbm["experiment"].astype(int)
    dbm["subject"] = dbm["subject"].astype(int)
    dbm["block"] = dbm["block"].astype(int)
    dbm["p"] = dbm["p"].astype(float)
    dbm["nll"] = dbm["nll"].astype(float)
    dbm["bic"] = dbm["bic"].astype(float)
    if args.block is not None:
        dbm = dbm.loc[dbm["block"] == args.block]

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

    groups = list(sim_in.groupby(keys + ["best_model", "best_params"], sort=False))
    if args.max_groups is not None:
        if args.max_groups < 1:
            raise ValueError("max_groups must be >= 1 when provided")
        groups = groups[:args.max_groups]
    chunk_groups = groups[args.chunk_index::args.num_chunks]

    print(
        f"Total groups={len(groups)}, chunk_index={args.chunk_index}, "
        f"num_chunks={args.num_chunks}, groups_in_chunk={len(chunk_groups)}, "
        f"max_groups={args.max_groups}"
    )

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
        rng = np.random.default_rng(args.seed + (args.chunk_index * 100000) + rep)
        np.random.seed(int(rng.integers(0, 2**31 - 1)))

        for gk, gg in chunk_groups:

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

            fit_seed = int(rng.integers(0, 2**31 - 1))
            fit = fit_dbm(ds, models, fit_side, k, n_trials, model_names, fit_seed)
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
                "chunk_index": args.chunk_index,
                "num_chunks": args.num_chunks,
                "true_model": true_model,
                "recovered_model": recovered_model,
                "true_family": model_family_map[true_key],
                "recovered_family": model_family_map[rec_key],
                "success_strict": int(recovered_model == true_model),
                "success_family": int(model_family_map[rec_key] == model_family_map[true_key]),
            })

    rec = pd.DataFrame(rec)

    os.makedirs(args.out_dir, exist_ok=True)
    block_tag = "all" if args.block is None else str(args.block)
    chunk_tag = f"chunk_{args.chunk_index:04d}_of_{args.num_chunks:04d}"
    out_path = os.path.join(
        args.out_dir,
        f"dbm_recovery_empirical_results_block_{block_tag}_{chunk_tag}.csv",
    )
    rec.to_csv(out_path, index=False)
    print(f"Wrote chunk results to: {out_path}")
