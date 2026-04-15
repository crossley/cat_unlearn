import argparse

from merge_dbm_recovery_chunks import merge_recovery_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", type=int, default=6)
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--in-dir", type=str, default="../dbm_fits/recovery_chunks")
    parser.add_argument("--out-dir", type=str, default="../dbm_fits")
    args = parser.parse_args()

    merge_recovery_chunks(
        in_dir=args.in_dir,
        glob_pattern=f"dbm_recovery_empirical_results_block_{args.block}_chunk_*.csv",
        out_prefix=f"{args.out_dir}/dbm_recovery_empirical_block_{args.block}",
        expected_num_chunks=args.num_chunks,
    )
