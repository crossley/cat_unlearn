import os

import numpy as np
import pandas as pd


def load_trial_data(data_dir="../data"):
    d_rec = []

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            d = pd.read_csv(os.path.join(data_dir, file))
            d["orig_trial_index"] = np.arange(len(d), dtype=int)
            d["phase"] = ["Learn"] * 300 + ["Intervention"] * 300 + ["Test"] * 299
            d_rec.append(d)

    d = pd.concat(d_rec, ignore_index=True)

    cat_map = {"A": 0, "B": 1, "0": 0, "1": 1, 0: 0, 1: 1}
    resp_map = {"A": 0, "B": 1, "0": 0, "1": 1, 0: 0, 1: 1}

    d["cat"] = d["cat"].replace(cat_map)
    d["resp"] = d["resp"].replace(resp_map)

    valid = d["cat"].isin([0, 1]) & d["resp"].isin([0, 1])

    invalid_rate = (
        pd.DataFrame({
            "experiment": d["experiment"],
            "condition": d["condition"],
            "subject": d["subject"],
            "invalid": ~valid,
        })
        .groupby(["experiment", "condition", "subject"], as_index=False)
        .agg(pct_invalid=("invalid", "mean"))
    )
    bad_subs = invalid_rate.loc[
        invalid_rate["pct_invalid"] > 0.10,
        ["experiment", "condition", "subject"],
    ]

    if bad_subs.shape[0] > 0:
        print(f"Excluding {bad_subs.shape[0]} participants with >10% unmapped responses.")
        print(bad_subs.groupby(["experiment", "condition"])["subject"].nunique())
        d = d.merge(
            bad_subs.assign(exclude_subject=True),
            on=["experiment", "condition", "subject"],
            how="left",
        )
        d = d[d["exclude_subject"] != True].drop(columns="exclude_subject")
        valid = d["cat"].isin([0, 1]) & d["resp"].isin([0, 1])

    n_drop = (~valid).sum()
    if n_drop > 0:
        print(f"Dropping {n_drop} rows with unmapped cat/resp values.")

    d = d.loc[valid].copy()
    d["cat"] = d["cat"].astype(int)
    d["resp"] = d["resp"].astype(int)
    d["orig_trial_index"] = d["orig_trial_index"].astype(int)
    d["acc"] = d["cat"] == d["resp"]

    print(d.groupby(["experiment", "condition"])["subject"].unique())
    print(d.groupby(["experiment", "condition"])["subject"].nunique())

    return d
