import argparse
import glob
import os
import re

import pandas as pd


def merge_glc_gcc_recovery_chunks(in_dir, out_prefix, expected_num_chunks):
    paths = sorted(glob.glob(os.path.join(in_dir, "glc_gcc_recovery_chunk_*.csv")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No chunk files found in {in_dir}")

    chunk_pattern = re.compile(r"glc_gcc_recovery_chunk_(\d{4})_of_(\d{4})\.csv$")
    chunk_map = {}

    for path in paths:
        match = chunk_pattern.search(os.path.basename(path))
        if match is None:
            raise ValueError(f"Could not parse chunk id from filename: {path}")

        chunk_idx = int(match.group(1))
        chunk_total = int(match.group(2))
        if chunk_total != expected_num_chunks:
            raise ValueError(
                f"Chunk file {path} reports total={chunk_total}, "
                f"expected {expected_num_chunks}"
            )
        if chunk_idx in chunk_map:
            raise ValueError(f"Duplicate chunk index detected: {chunk_idx:04d}")
        chunk_map[chunk_idx] = path

    missing = sorted(set(range(expected_num_chunks)) - set(chunk_map))
    if missing:
        missing_preview = ", ".join([f"{idx:04d}" for idx in missing[:10]])
        suffix = "" if len(missing) <= 10 else ", ..."
        raise ValueError(
            f"Missing {len(missing)} GLC/GCC recovery chunks out of "
            f"{expected_num_chunks}: {missing_preview}{suffix}"
        )

    rec = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    rec = rec.drop_duplicates().reset_index(drop=True)

    cm_family_counts = pd.crosstab(rec["true_family"], rec["recovered_family"])
    cm_family_props = pd.crosstab(
        rec["true_family"], rec["recovered_family"], normalize="index"
    )
    cm_model_counts = pd.crosstab(rec["true_model"], rec["recovered_model"])
    cm_model_props = pd.crosstab(
        rec["true_model"], rec["recovered_model"], normalize="index"
    )

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rec.to_csv(f"{out_prefix}_results.csv", index=False)
    cm_family_counts.to_csv(f"{out_prefix}_family_counts.csv")
    cm_family_props.to_csv(f"{out_prefix}_family_props.csv")
    cm_model_counts.to_csv(f"{out_prefix}_model_counts.csv")
    cm_model_props.to_csv(f"{out_prefix}_model_props.csv")

    print(f"Merged {len(paths)} files into {out_prefix}_results.csv")
    print("Rows:", len(rec))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--in-dir", type=str, default="../dbm_fits/glc_gcc_recovery_chunks")
    parser.add_argument("--out-prefix", type=str, default="../dbm_fits/glc_gcc_recovery")
    args = parser.parse_args()

    merge_glc_gcc_recovery_chunks(
        in_dir=args.in_dir,
        out_prefix=args.out_prefix,
        expected_num_chunks=args.num_chunks,
    )
