import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SMOOTH = 5  # shrinkage: artists with few songs are pulled towards the global mean


def parse_artists(s):
    return ast.literal_eval(s)


def base_features(df):
    out = pd.DataFrame(index=df.index)
    for c in ["acousticness", "danceability", "energy", "loudness", "tempo", "valence"]:
        out[c] = df[c]
    out["log_duration"] = np.log(df["duration_ms"])
    for c in ["speechiness", "instrumentalness", "liveness"]:
        out["sqrt_" + c] = np.sqrt(df[c])
    out["explicit"] = df["explicit"]
    out["mode"] = df["mode"]
    for k in range(1, 12):  # key_0 dropped as reference category
        out[f"key_{k}"] = (df["key"] == k).astype(float)

    out["n_artists"] = df["artists"].apply(lambda s: len(parse_artists(s))).clip(upper=5)
    title = df["name"].str.lower()
    out["title_feat"] = title.str.contains(r"\b(?:feat|ft)\b").astype(float)
    out["title_remix"] = title.str.contains("remix").astype(float)

    has_month = df["release_date"].str.len() == 10
    month = pd.to_datetime(df["release_date"].where(has_month)).dt.month
    angle = 2 * np.pi * month / 12
    out["month_sin"] = np.sin(angle).fillna(0)  # unknown month -> 0 (neutral)
    out["month_cos"] = np.cos(angle).fillna(0)
    return out


def artist_stats(df):
    """artist -> (sum of target, number of songs), from the given rows only."""
    stats = {}
    for s, t in zip(df["artists"], df["target"]):
        for a in parse_artists(s):
            tot, cnt = stats.get(a, (0.0, 0))
            stats[a] = (tot + t, cnt + 1)
    return stats


def artist_features(df, stats, global_mean):
    enc, counts = [], []
    for s in df["artists"]:
        vals, cnts = [], []
        for a in parse_artists(s):
            tot, cnt = stats.get(a, (0.0, 0))  # unseen artist -> global mean
            vals.append((tot + SMOOTH * global_mean) / (cnt + SMOOTH))
            cnts.append(cnt)
        enc.append(np.mean(vals))
        counts.append(np.log1p(max(cnts)))
    return np.array(enc), np.array(counts)


def oof_artist_features(train, global_mean, n_splits=5, seed=42):
    """Out-of-fold encoding: a song never sees its own target."""
    enc = np.zeros(len(train))
    cnt = np.zeros(len(train))
    for fit_idx, hold_idx in KFold(n_splits, shuffle=True, random_state=seed).split(train):
        stats = artist_stats(train.iloc[fit_idx])
        e, c = artist_features(train.iloc[hold_idx], stats, global_mean)
        enc[hold_idx] = e
        cnt[hold_idx] = c
    return enc, cnt


def build_features(train, val, test):
    gm = train["target"].mean()
    full_stats = artist_stats(train)
    dfs = {"train": train, "val": val, "test": test}

    raw = {}
    for name, df in dfs.items():
        f = base_features(df)
        if name == "train":
            e, c = oof_artist_features(train, gm)
        else:
            e, c = artist_features(df, full_stats, gm)
        f["artist_enc"] = e
        f["artist_log_count"] = c
        raw[name] = f

    mean = raw["train"].mean()
    std = raw["train"].std().replace(0, 1)  # scaler fitted on train only

    out = {"feature_names": list(raw["train"].columns), "mean": mean, "std": std,
           "artist_stats": full_stats, "global_mean": gm}
    for name, df in dfs.items():
        out["X_" + name] = ((raw[name] - mean) / std).to_numpy(dtype=np.float32)
        out["y_" + name] = df["target"].to_numpy(dtype=np.float32)
    return out
    