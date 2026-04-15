import argparse
import glob
import os

import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default="../dbm_fits/recovery_chunks")
    parser.add_argument("--glob-pattern", type=str, default="dbm_recovery_empirical_results_block_*.csv")
    parser.add_argument("--out-prefix", type=str, default="../dbm_fits/dbm_recovery_empirical")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.in_dir, args.glob_pattern)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No chunk files found in {args.in_dir} matching {args.glob_pattern}")

    rec = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    rec = rec.drop_duplicates().reset_index(drop=True)

    cm_family_counts = pd.crosstab(rec["true_family"], rec["recovered_family"])
    cm_family_props = pd.crosstab(rec["true_family"], rec["recovered_family"], normalize="index")
    cm_model_counts = pd.crosstab(rec["true_model"], rec["recovered_model"])
    cm_model_props = pd.crosstab(rec["true_model"], rec["recovered_model"], normalize="index")

    rec_path = f"{args.out_prefix}_results.csv"
    fam_counts_path = f"{args.out_prefix}_family_counts.csv"
    fam_props_path = f"{args.out_prefix}_family_props.csv"
    mod_counts_path = f"{args.out_prefix}_model_counts.csv"
    mod_props_path = f"{args.out_prefix}_model_props.csv"

    rec.to_csv(rec_path, index=False)
    cm_family_counts.to_csv(fam_counts_path)
    cm_family_props.to_csv(fam_props_path)
    cm_model_counts.to_csv(mod_counts_path)
    cm_model_props.to_csv(mod_props_path)

    print(f"Merged {len(paths)} files into {rec_path}")
    print("Rows:", len(rec))
