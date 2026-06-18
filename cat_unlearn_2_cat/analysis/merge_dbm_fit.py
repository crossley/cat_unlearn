import argparse
import glob
import os
import re

import pandas as pd


def merge_dbm_fit_chunks(in_dir, out_path, expected_num_chunks):
    paths = sorted(glob.glob(os.path.join(in_dir, "dbm_results_chunk_*.csv")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No DBM fit chunks found in {in_dir}")

    chunk_pattern = re.compile(r"dbm_results_chunk_(\d{4})_of_(\d{4})\.csv$")
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
            f"Missing {len(missing)} DBM fit chunks out of "
            f"{expected_num_chunks}: {missing_preview}{suffix}"
        )

    dbm = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    dbm = dbm.drop_duplicates().reset_index(drop=True)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dbm.to_csv(out_path, index=False)

    print(f"Merged {len(paths)} files into {out_path}")
    print("Rows:", len(dbm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--in-dir", type=str, default="../dbm_fits/dbm_results_chunks")
    parser.add_argument("--out-path", type=str, default="../dbm_fits/dbm_results_gadi.csv")
    args = parser.parse_args()

    merge_dbm_fit_chunks(
        in_dir=args.in_dir,
        out_path=args.out_path,
        expected_num_chunks=args.num_chunks,
    )
