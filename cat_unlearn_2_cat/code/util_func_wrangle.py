import os
import numpy as np
import pandas as pd


def get_cl_df():

    dir_data = "../data"

    d_rec = []

    for file in os.listdir(dir_data):

        if file.endswith(".csv"):
            d = pd.read_csv(os.path.join(dir_data, file))
            d["phase"] = ["Learn"] * 300 + ["Intervention"] * 300 + ["Test"
                                                                     ] * 299
            d_rec.append(d)

    d = pd.concat(d_rec, ignore_index=True)

    d.groupby(["experiment", "condition"])["subject"].unique()
    d.groupby(["experiment", "condition"])["subject"].nunique()

    cat_map = {"A": 0, "B": 1, "0": 0, "1": 1, 0: 0, 1: 1}
    resp_map = {
        "A": 0, "B": 1, "0": 0, "1": 1, 0: 0, 1: 1
    }

    d["cat"] = d["cat"].replace(cat_map)
    d["resp"] = d["resp"].replace(resp_map)

    valid = d["cat"].isin([0, 1]) & d["resp"].isin([0, 1])

    # exclude participants with >10% unmapped responses
    invalid_rate = (
        pd.DataFrame({
            "experiment": d["experiment"],
            "condition": d["condition"],
            "subject": d["subject"],
            "invalid": ~valid
        }).groupby(["experiment", "condition", "subject"], as_index=False)
          .agg(pct_invalid=("invalid", "mean"))
    )
    bad_subs = invalid_rate.loc[invalid_rate["pct_invalid"] > 0.10,
                                ["experiment", "condition", "subject"]]
    n_bad = bad_subs.shape[0]
    if n_bad > 0:
        print(f"Excluding {n_bad} participants with >10% unmapped responses.")
        print(
            bad_subs.groupby(["experiment", "condition"])["subject"].nunique())
        d = d.merge(bad_subs.assign(exclude_subject=True),
                    on=["experiment", "condition", "subject"],
                    how="left")
        d = d[d["exclude_subject"] != True].drop(columns="exclude_subject")
        valid = d["cat"].isin([0, 1]) & d["resp"].isin([0, 1])

    n_drop = (~valid).sum()
    if n_drop > 0:
        print(f"Dropping {n_drop} rows with unmapped cat/resp values.")
    d = d.loc[valid].copy()

    d["cat"] = d["cat"].astype(int)
    d["resp"] = d["resp"].astype(int)
    d["acc"] = d["cat"] == d["resp"]

    print(d.groupby(["experiment", "condition"])["subject"].unique())
    print(d.groupby(["experiment", "condition"])["subject"].nunique())

    return d


def get_dbm_df():

    if os.path.exists("../dbm_fits/dbm_results.csv"):
        dbm = pd.read_csv("../dbm_fits/dbm_results.csv")
        dbm = dbm[dbm["block"] != "block"].copy()
        dbm["block"] = dbm["block"].astype(int)
        dbm["bic"] = dbm["bic"].astype(float)
        dbm["experiment"] = dbm["experiment"].astype(int)
        dbm["subject"] = dbm["subject"].astype(int)
    else:
        print("DBM results file not found. Please run fit_dbm_top() first.")
        return

    def assign_best_model(x):
        model = x["model"].to_numpy()
        bic = x["bic"].to_numpy()
        best_model = np.unique(model[bic == bic.min()])[0]
        x["best_model"] = best_model
        return x

    def get_best_model_frame(dbm):
        ddd = (dbm.groupby(["experiment", "condition", "subject", "block"
                            ]).apply(assign_best_model).reset_index(drop=True))
        ddd = ddd.loc[
            ddd["model"] == ddd["best_model"],
            ["experiment", "condition", "subject", "block", "best_model"
             ]].drop_duplicates()
        ddd.loc[ddd["best_model"] == "nll_rand_guess",
                "best_model_class"] = "guessing"
        ddd.loc[ddd["best_model"] == "nll_bias_guess",
                "best_model_class"] = "guessing"
        ddd.loc[ddd["best_model"] == "nll_unix_0",
                "best_model_class"] = "rule-based"
        ddd.loc[ddd["best_model"] == "nll_unix_1",
                "best_model_class"] = "rule-based"
        ddd.loc[ddd["best_model"] == "nll_uniy_0",
                "best_model_class"] = "rule-based"
        ddd.loc[ddd["best_model"] == "nll_uniy_1",
                "best_model_class"] = "rule-based"
        ddd.loc[ddd["best_model"] == "nll_glc_0",
                "best_model_class"] = "procedural"
        ddd.loc[ddd["best_model"] == "nll_glc_1",
                "best_model_class"] = "procedural"
        ddd["best_model_class"] = ddd["best_model_class"].astype("category")
        ddd["block"] = ddd["block"].astype("category")
        ddd = ddd.reset_index(drop=True)

        return ddd

    ddd = get_best_model_frame(dbm)

    exc_subs_learn = ddd[(ddd["block"] == 2)
                         & (ddd["best_model_class"] == "guessing")][[
                             "experiment", "condition", "subject"
                         ]].drop_duplicates()

    print(
        "Excluding subjects best fit by guessing in the last learning block:")
    print(
        exc_subs_learn.groupby(["experiment",
                                "condition"])["subject"].nunique())

    ddd = ddd.merge(exc_subs_learn.assign(exclude_subject=True),
                    on=["experiment", "condition", "subject"],
                    how="left")
    ddd = ddd[ddd["exclude_subject"] != True].drop(columns="exclude_subject")

    return ddd
