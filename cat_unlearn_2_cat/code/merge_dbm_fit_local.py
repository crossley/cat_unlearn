import argparse

from merge_dbm_fit_chunks import merge_dbm_fit_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--in-dir", type=str, default="../dbm_fits/dbm_results_chunks")
    parser.add_argument("--out-path", type=str, default="../dbm_fits/dbm_results_gadi.csv")
    args = parser.parse_args()

    merge_dbm_fit_chunks(
        in_dir=args.in_dir,
        glob_pattern="dbm_results_chunk_*.csv",
        out_path=args.out_path,
        expected_num_chunks=args.num_chunks,
    )
