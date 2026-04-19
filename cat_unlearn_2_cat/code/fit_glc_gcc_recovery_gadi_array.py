import argparse
import os
import time

import numpy as np
import pandas as pd

from util_func_dbm import fit_dbm, nll_glc, nll_gcc_eq, val_glc, val_gcc_eq
from util_func_glc_gcc_recovery import (
    glc_slope_diag_to_params,
    make_cat_trials,
    make_glc_gcc_recovery_jobs,
)


def _model_key(model_name):
    parts = model_name.split("_")
    return "_".join(parts[:-1]) if parts[-1].isdigit() else model_name


def _model_family(model_name):
    key = _model_key(model_name)
    if key == "nll_glc":
        return "GLC"
    if key == "nll_gcc_eq":
        return "GCC_eq"
    raise ValueError(f"Unknown model family for {model_name}")


def _model_side(model_name):
    parts = model_name.split("_")
    return int(parts[-1]) if parts[-1].isdigit() else np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=462)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--optimizer-workers", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="../dbm_fits/glc_gcc_recovery_chunks")
    args = parser.parse_args()

    if args.n_reps < 1:
        raise ValueError("n_reps must be >= 1")
    if args.num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if args.chunk_index < 0 or args.chunk_index >= args.num_chunks:
        raise ValueError("chunk_index must be in [0, num_chunks)")

    n_per_cat = 50
    n_trials = n_per_cat * 2
    z_limit = 3

    models = [
        nll_glc,
        nll_glc,
        nll_gcc_eq,
        nll_gcc_eq,
        nll_gcc_eq,
        nll_gcc_eq,
    ]
    fit_side = [0, 1, 0, 1, 2, 3]
    k = [3, 3, 3, 3, 3, 3]
    model_names = [
        "nll_glc_0",
        "nll_glc_1",
        "nll_gcc_eq_0",
        "nll_gcc_eq_1",
        "nll_gcc_eq_2",
        "nll_gcc_eq_3",
    ]

    jobs = make_glc_gcc_recovery_jobs(args.n_reps)
    chunk_jobs = jobs[args.chunk_index::args.num_chunks]

    print(
        f"Total jobs={len(jobs)}, chunk_index={args.chunk_index}, "
        f"num_chunks={args.num_chunks}, jobs_in_chunk={len(chunk_jobs)}"
    )

    if len(chunk_jobs) == 0:
        raise ValueError("No jobs assigned to this chunk.")

    rec = []
    t0 = time.time()

    for job in chunk_jobs:
        np.random.seed(args.seed + job["job_id"])
        x, y, cat = make_cat_trials(n_per_cat)
        resp0 = np.zeros(n_trials, dtype=int)

        if job["true_family"] == "GLC":
            a1, b = glc_slope_diag_to_params(
                job["true_slope"], job["true_diag"]
            )
            _, _, _, resp = val_glc(
                (a1, b, job["true_noise"]),
                z_limit,
                cat,
                x,
                y,
                resp0,
                job["true_side"],
            )
        elif job["true_family"] == "GCC_eq":
            _, _, _, resp = val_gcc_eq(
                (job["true_xc"], job["true_yc"], job["true_noise"]),
                z_limit,
                cat,
                x,
                y,
                resp0,
                job["true_side"],
            )
        else:
            raise ValueError(f"Unknown true family: {job['true_family']}")

        ds = pd.DataFrame({
            "cat": cat,
            "x": x,
            "y": y,
            "resp": np.asarray(resp).reshape(-1),
            "condition": "sim",
            "subject": "sub",
        })

        fit = fit_dbm(
            ds,
            models,
            fit_side,
            k,
            n_trials,
            model_names,
            args.seed + job["job_id"],
            args.optimizer_workers,
        )
        fit_bic = fit.groupby("model", as_index=False)["bic"].min()
        recovered_model = fit_bic.loc[fit_bic["bic"].idxmin(), "model"]
        recovered_family = _model_family(recovered_model)
        recovered_side = _model_side(recovered_model)

        rec.append({
            "true_model": job["true_model"],
            "true_family": job["true_family"],
            "true_side": job["true_side"],
            "true_slope": job["true_slope"],
            "true_diag": job["true_diag"],
            "true_xc": job["true_xc"],
            "true_yc": job["true_yc"],
            "true_noise": job["true_noise"],
            "rep": job["rep"],
            "recovered_model": recovered_model,
            "recovered_family": recovered_family,
            "recovered_side": recovered_side,
            "success_family": int(recovered_family == job["true_family"]),
            "success_strict": int(recovered_model == job["true_model"]),
        })

    out = pd.DataFrame(rec)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"glc_gcc_recovery_chunk_{args.chunk_index:04d}_of_{args.num_chunks:04d}.csv",
    )
    out.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"Wrote GLC/GCC recovery chunk to: {out_path}")
    print(f"Elapsed seconds: {elapsed:.1f}")
