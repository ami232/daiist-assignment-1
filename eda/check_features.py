import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from data_prep import load_data, split
from features import build_features

df = load_data()
tr, va, te = split(df)
print("target std by year:")
print(df.groupby("year").target.std().round(1).to_string())

d = build_features(tr, va, te)
names = d["feature_names"]
print("shapes:", d["X_train"].shape, d["X_val"].shape, d["X_test"].shape)
print("NaNs:", sum(int(np.isnan(d[k]).sum()) for k in ["X_train", "X_val", "X_test"]))

print("skew after transform:")
X = pd.DataFrame(d["X_train"], columns=names)
print(X[["log_duration", "sqrt_speechiness", "sqrt_instrumentalness", "sqrt_liveness"]].skew().round(2))

i = names.index("artist_enc")
for s in ["train", "val", "test"]:
    print(f"corr(artist_enc, target) {s}:", round(np.corrcoef(d["X_" + s][:, i], d["y_" + s])[0, 1], 3))

keep = [j for j, n in enumerate(names) if not n.startswith("artist_")]
for a in [1, 100, 1000]:
    full = Ridge(alpha=a).fit(d["X_train"], d["y_train"]).score(d["X_val"], d["y_val"])
    no_art = Ridge(alpha=a).fit(d["X_train"][:, keep], d["y_train"]).score(d["X_val"][:, keep], d["y_val"])
    print(f"alpha={a}: val R2 with artist feats {full:.3f} | without {no_art:.3f}")