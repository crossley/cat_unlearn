import os

import pandas as pd


GUESSING_MODELS = ["nll_rand_guess", "nll_bias_guess"]
GCC_MODELS = ["nll_gcc_eq_0", "nll_gcc_eq_1", "nll_gcc_eq_2", "nll_gcc_eq_3"]


def classify_model(
    model,
    procedural_definition="glc",
    procedural_models=None,
    rule_based_models=None,
):
    if model in GUESSING_MODELS:
        return "guessing"
    if procedural_models is not None and model in procedural_models:
        return "procedural"
    if rule_based_models is not None and model in rule_based_models:
        return "rule-based"
    if procedural_models is not None and rule_based_models is not None:
        if model in procedural_models:
            return "procedural"
        if model in rule_based_models:
            return "rule-based"
        raise ValueError(f"Model has no class assignment: {model}")
    if model.startswith("nll_unix") or model.startswith("nll_uniy"):
        return "rule-based"
    if model.startswith("nll_gcc_eq"):
        if procedural_definition == "glc_gcc":
            return "procedural"
        return "rule-based"
    if model.startswith("nll_glc"):
        return "procedural"
    raise ValueError(f"Unknown DBM model: {model}")


def load_best_dbm_fits(
    path="../dbm_fits/dbm_results.csv",
    model_set="with_gcc",
    procedural_definition="glc",
    exclude_learning_guessers=True,
    include_models=None,
    exclude_models=None,
    procedural_models=None,
    rule_based_models=None,
):
    if model_set not in ["with_gcc", "no_gcc"]:
        raise ValueError("model_set must be 'with_gcc' or 'no_gcc'")
    if procedural_definition not in ["glc", "glc_gcc"]:
        raise ValueError("procedural_definition must be 'glc' or 'glc_gcc'")
    if include_models is not None and exclude_models is not None:
        overlap = sorted(set(include_models) & set(exclude_models))
        if overlap:
            raise ValueError(f"Models cannot be both included and excluded: {overlap}")
    if procedural_models is not None and rule_based_models is not None:
        overlap = sorted(set(procedural_models) & set(rule_based_models))
        if overlap:
            raise ValueError(
                f"Models cannot be both procedural and rule-based: {overlap}"
            )
    if not os.path.exists(path):
        raise FileNotFoundError(f"DBM results file not found: {path}")

    dbm = pd.read_csv(path)
    dbm = dbm[dbm["block"] != "block"].copy()
    dbm["block"] = dbm["block"].astype(int)
    dbm["bic"] = dbm["bic"].astype(float)
    dbm["experiment"] = dbm["experiment"].astype(int)
    dbm["subject"] = dbm["subject"].astype(int)

    available_models = set(dbm["model"].unique())
    for label, models in [
        ("included", include_models),
        ("excluded", exclude_models),
        ("procedural", procedural_models),
        ("rule-based", rule_based_models),
    ]:
        if models is not None:
            missing = sorted(set(models) - available_models)
            if missing:
                raise ValueError(f"Unknown {label} models: {missing}")

    if include_models is not None:
        dbm = dbm[dbm["model"].isin(include_models)].copy()

    if exclude_models is not None:
        dbm = dbm[~dbm["model"].isin(exclude_models)].copy()

    if model_set == "no_gcc":
        dbm = dbm[~dbm["model"].isin(GCC_MODELS)].copy()

    keys = ["experiment", "condition", "subject", "block"]
    bic_by_model = dbm.groupby(keys + ["model"], as_index=False)["bic"].min()
    idx = bic_by_model.groupby(keys)["bic"].idxmin()
    best = bic_by_model.loc[idx, keys + ["model", "bic"]].rename(
        columns={"model": "best_model", "bic": "best_bic"}
    )

    best["best_model_class"] = best["best_model"].apply(
        classify_model,
        procedural_definition=procedural_definition,
        procedural_models=procedural_models,
        rule_based_models=rule_based_models,
    )
    best["is_guessing"] = best["best_model_class"] == "guessing"
    best["is_rule_based"] = best["best_model_class"] == "rule-based"
    best["is_procedural"] = best["best_model_class"] == "procedural"
    best["is_glc"] = best["best_model"].str.startswith("nll_glc")
    best["is_gcc"] = best["best_model"].str.startswith("nll_gcc_eq")

    if exclude_learning_guessers:
        exc_subs_learn = best[(best["block"] == 2) & best["is_guessing"]][
            ["experiment", "condition", "subject"]
        ].drop_duplicates()

        print("Excluding subjects best fit by guessing in the last learning block:")
        print(exc_subs_learn.groupby(["experiment", "condition"])["subject"].nunique())

        best = best.merge(
            exc_subs_learn.assign(exclude_subject=True),
            on=["experiment", "condition", "subject"],
            how="left",
        )
        best = best[best["exclude_subject"] != True].drop(columns="exclude_subject")

    best["best_model_class"] = best["best_model_class"].astype("category")
    best["block"] = best["block"].astype("category")
    return best.reset_index(drop=True)
