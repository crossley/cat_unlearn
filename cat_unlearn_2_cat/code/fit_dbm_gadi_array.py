import argparse
import os
import time

import pandas as pd

from util_func_dbm import fit_dbm, get_dbm_fit_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=462)
    parser.add_argument("--optimizer-workers", type=int, default=1)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="../dbm_fits/dbm_results_chunks")
    args = parser.parse_args()

    if args.num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if args.chunk_index < 0 or args.chunk_index >= args.num_chunks:
        raise ValueError("chunk_index must be in [0, num_chunks)")

    d, models, side, k, n, model_names = get_dbm_fit_inputs()
    group_cols = ["experiment", "condition", "subject", "block"]
    groups = list(d.groupby(group_cols, sort=False))
    chunk_groups = groups[args.chunk_index::args.num_chunks]

    print(
        f"Total groups={len(groups)}, chunk_index={args.chunk_index}, "
        f"num_chunks={args.num_chunks}, groups_in_chunk={len(chunk_groups)}"
    )

    if len(chunk_groups) == 0:
        raise ValueError("No groups assigned to this chunk.")

    t0 = time.time()
    rec = []
    for _, group_df in chunk_groups:
        fit = fit_dbm(
            group_df,
            models,
            side,
            k,
            n,
            model_names,
            args.seed,
            args.optimizer_workers,
        ).reset_index(drop=True)

        meta = group_df.iloc[0][group_cols].to_dict()
        for key, value in meta.items():
            fit[key] = value
        rec.append(fit[["experiment", "condition", "subject", "block", "p", "nll", "bic", "model"]])

    out = rec[0] if len(rec) == 1 else pd.concat(rec, ignore_index=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"dbm_results_chunk_{args.chunk_index:04d}_of_{args.num_chunks:04d}.csv",
    )
    out.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print(f"Wrote DBM chunk to: {out_path}")
    print(f"Elapsed seconds: {elapsed:.1f}")
