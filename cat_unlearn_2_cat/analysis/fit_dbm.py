import argparse
import os
import time

import pandas as pd

from dbm_models import fit_dbm, get_dbm_fit_inputs


GROUP_COLS = ["experiment", "condition", "subject", "block"]
OUT_COLS = ["experiment", "condition", "subject", "block", "p", "nll", "bic", "model"]


def fit_groups(groups, models, side, k, n, model_names, seed, optimizer_workers):
    rec = []

    for _, group_df in groups:
        fit = fit_dbm(
            group_df,
            models,
            side,
            k,
            n,
            model_names,
            seed,
            optimizer_workers,
        ).reset_index(drop=True)

        meta = group_df.iloc[0][GROUP_COLS].to_dict()
        for key, value in meta.items():
            fit[key] = value
        rec.append(fit[OUT_COLS])

    if len(rec) == 0:
        raise ValueError("No groups to fit.")
    return rec[0] if len(rec) == 1 else pd.concat(rec, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=462)
    parser.add_argument("--optimizer-workers", type=int, default=1)
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--out-path", type=str, default="../dbm_fits/dbm_results.csv")
    parser.add_argument("--out-dir", type=str, default="../dbm_fits/dbm_results_chunks")
    args = parser.parse_args()

    if args.num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if args.chunk_index is not None and (
        args.chunk_index < 0 or args.chunk_index >= args.num_chunks
    ):
        raise ValueError("chunk_index must be in [0, num_chunks)")

    d, models, side, k, n, model_names = get_dbm_fit_inputs()
    groups = list(d.groupby(GROUP_COLS, sort=False))

    if args.chunk_index is None:
        selected_groups = groups
        mode = "full"
        print(f"Fitting all groups={len(groups)}")
    else:
        selected_groups = groups[args.chunk_index::args.num_chunks]
        mode = "chunk"
        print(
            f"Total groups={len(groups)}, chunk_index={args.chunk_index}, "
            f"num_chunks={args.num_chunks}, groups_in_chunk={len(selected_groups)}"
        )

    t0 = time.time()
    out = fit_groups(
        selected_groups,
        models,
        side,
        k,
        n,
        model_names,
        args.seed,
        args.optimizer_workers,
    )

    if mode == "full":
        out_dir = os.path.dirname(args.out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out.to_csv(args.out_path, index=False)
        print(f"Wrote DBM fits to: {args.out_path}")
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(
            args.out_dir,
            f"dbm_results_chunk_{args.chunk_index:04d}_of_{args.num_chunks:04d}.csv",
        )
        out.to_csv(out_path, index=False)
        print(f"Wrote DBM chunk to: {out_path}")

    elapsed = time.time() - t0
    print(f"Elapsed seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
