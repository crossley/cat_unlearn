import argparse
import os
import time

from util_func_dbm import fit_dbm_top


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=462)
    parser.add_argument("--optimizer-workers", type=int, default=1)
    parser.add_argument("--out-path", type=str, default="../dbm_fits/dbm_results.csv")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    fit_dbm_top(
        seed=args.seed,
        optimizer_workers=args.optimizer_workers,
        out_path=args.out_path,
    )
    elapsed = time.time() - t0
    print(f"Wrote DBM fits to: {args.out_path}")
    print(f"Elapsed seconds: {elapsed:.1f}")
