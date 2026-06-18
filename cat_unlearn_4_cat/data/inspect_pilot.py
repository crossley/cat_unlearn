# !/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

subjects = np.arange(1, 18, 1)

dg = []
for s in subjects:
    ds = pd.read_csv(f"sub_{s}_data.csv")
    ds["block"] = ds["trial"] // 50
    ds["phase"] = ds["trial"] // 300
    ds["acc"] = ds["cat"] == ds["resp"]
    ds["acc"] = ds["acc"].astype(float)
    dg.append(ds)

dg = pd.concat(dg, ignore_index=True)

dd = dg.groupby(["experiment", "condition", "subject", "phase", "block"]).agg({
    "rt":
    "mean",
    "acc":
    "mean"
}).reset_index()

PHASE_PALETTES = {
    "relearn": dict(zip([0, 1, 2], sns.color_palette("rocket", 5)[:4:2] + [sns.color_palette("rocket", 5)[3]])),
    "new_learn": dict(zip([0, 1, 2], sns.color_palette("mako", 5)[:4:2] + [sns.color_palette("mako", 5)[3]])),
}

fig, ax = plt.subplots(subjects.shape[0], 2, figsize=(12, 5))
for i, s in enumerate(subjects):
    ds = dd[dd["subject"] == s]
    palette = PHASE_PALETTES[ds["condition"].iloc[0]]
    sns.lineplot(data=ds,
                 x="block",
                 y="acc",
                 hue="phase",
                 hue_order=[0, 1, 2],
                 palette=palette,
                 legend=False,
                 ax=ax[i, 0])
    sns.lineplot(data=ds,
                 x="block",
                 y="rt",
                 hue="phase",
                 hue_order=[0, 1, 2],
                 palette=palette,
                 legend=False,
                 ax=ax[i, 1])
    ax[i, 0].set_title(f"Subject {s}: Condition {ds['condition'].iloc[0]}")
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.lineplot(data=dd,
             x="block",
             y="acc",
             style="phase",
             hue="condition",
             ax=ax)
plt.show()
