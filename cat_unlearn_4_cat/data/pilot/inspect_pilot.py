# !/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d = pd.read_csv("sub_2_data.csv")
d["block"] = d["trial"] // 50
d["phase"] = d["trial"] // 300
d["acc"] = d["cat"] == d["resp"]
d["acc"] = d["acc"].astype(float)
dd = d.groupby("block").agg({"rt": "mean", "acc": "mean"}).reset_index()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.lineplot(data=d, x="block", y="acc", hue="phase",  legend=False, ax=ax[0])
sns.lineplot(data=d, x="block", y="rt", hue="phase", legend=False, ax=ax[1])
plt.show()

