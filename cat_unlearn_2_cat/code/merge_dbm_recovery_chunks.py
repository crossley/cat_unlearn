import argparse
import glob
import os
import re

import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default="../dbm_fits/recovery_chunks")
    parser.add_argument("--glob-pattern", type=str, default="dbm_recovery_empirical_results_block_*.csv")
    parser.add_argument("--out-prefix", type=str, default="../dbm_fits/dbm_recovery_empirical")
    parser.add_argument("--expected-num-chunks", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.in_dir, args.glob_pattern)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No chunk files found in {args.in_dir} matching {args.glob_pattern}")

    if args.expected_num_chunks is not None:
        chunk_pattern = re.compile(r"_chunk_(\d{4})_of_(\d{4})\.csv$")
        chunk_map = {}

        for path in paths:
            match = chunk_pattern.search(os.path.basename(path))
            if match is None:
                raise ValueError(f"Could not parse chunk id from filename: {path}")

            chunk_idx = int(match.group(1))
            chunk_total = int(match.group(2))
            if chunk_total != args.expected_num_chunks:
                raise ValueError(
                    f"Chunk file {path} reports total={chunk_total}, "
                    f"expected {args.expected_num_chunks}"
                )
            if chunk_idx in chunk_map:
                raise ValueError(f"Duplicate chunk index detected: {chunk_idx:04d}")
            chunk_map[chunk_idx] = path

        missing = sorted(set(range(args.expected_num_chunks)) - set(chunk_map))
        if missing:
            missing_preview = ", ".join([f"{idx:04d}" for idx in missing[:10]])
            suffix = "" if len(missing) <= 10 else ", ..."
            raise ValueError(
                f"Missing {len(missing)} recovery chunks out of "
                f"{args.expected_num_chunks}: {missing_preview}{suffix}"
            )

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
