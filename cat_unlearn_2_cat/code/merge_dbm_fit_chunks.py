import argparse
import glob
import os
import re

import pandas as pd


def merge_dbm_fit_chunks(in_dir, glob_pattern, out_path, expected_num_chunks=None):
    paths = sorted(glob.glob(os.path.join(in_dir, glob_pattern)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No chunk files found in {in_dir} matching {glob_pattern}")

    if expected_num_chunks is not None:
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
    parser.add_argument("--in-dir", type=str, default="../dbm_fits/dbm_results_chunks")
    parser.add_argument("--glob-pattern", type=str, default="dbm_results_chunk_*.csv")
    parser.add_argument("--out-path", type=str, default="../dbm_fits/dbm_results_gadi.csv")
    parser.add_argument("--expected-num-chunks", type=int, default=None)
    args = parser.parse_args()

    merge_dbm_fit_chunks(
        in_dir=args.in_dir,
        glob_pattern=args.glob_pattern,
        out_path=args.out_path,
        expected_num_chunks=args.expected_num_chunks,
    )
