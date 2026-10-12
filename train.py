"""
Assignment 1 — training pipeline.

Business question (see REPORT.md): an M&A advisory boutique ranks venture-backed
startups by how likely they are to be acquired, relative to peers of the same age,
from their funding profile, and pitches the top 5% of the list.

Run with `uv run python main.py train`. The script is meant to be read top to bottom:

    Part 1  Data exploration      what the raw data looks like and how variables behave
    Part 2  Preprocessing         cleaning, target, cohorts and split, each step
                                  justified by a finding in Part 1
    Part 3  Feature engineering   features derived from the Part 1 findings
    Part 4  Feature checks        sanity checks and the date-feature comparison

Section numbers in comments (e.g. "see 1.6") point back to the finding a step rests on.
"""

# %% Setup
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 120)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "investments_VC.csv"
ARTIFACTS = ROOT / "artifacts"  # everything app.py loads; nothing there is retrained

# Business costs from REPORT.md: a wasted pitch, and an acquirer we did not pitch
# (EUR 150k fee x 25% assumed win rate).
COST_FP = 3_000
COST_FN = 37_500

SNAPSHOT_YEAR = 2014           # last funding dates end on 2015-01-01 (see 1.5)
COHORTS = range(2000, 2009)    # founded 2000-2008 (see 1.6)
TRAIN_YEARS = range(2000, 2006)
VAL_YEARS = [2006]
TEST_YEARS = [2007, 2008]
TOP_SHARE = 0.05               # the boutique pitches the top 5% of the ranked list
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)  # regularisation strengths tried (4.2 and 5.2)

FUNDING_TYPES = [
    "seed", "angel", "venture", "debt_financing", "convertible_note", "grant",
    "equity_crowdfunding", "product_crowdfunding", "private_equity", "undisclosed",
    "secondary_market",
]
LETTERED_ROUNDS = ["round_A", "round_B", "round_C", "round_D",
                   "round_E", "round_F", "round_G", "round_H"]
POST_IPO = ["post_ipo_equity", "post_ipo_debt"]
LINE = "=" * 78


# =================================================================================
# PART 1 — DATA EXPLORATION
# =================================================================================

# %% 1.1 Raw structure
# The export is latin-1, not UTF-8 (reading it as UTF-8 fails), and a few column
# headers carry stray spaces.
raw = pd.read_csv(DATA_PATH, encoding="latin-1")
raw.columns = raw.columns.str.strip()

print(f"\n{LINE}\n1.1 Raw structure\n{LINE}")
print(f"rows x columns: {raw.shape[0]:,} x {raw.shape[1]}")
print(f"completely empty rows: {raw.isna().all(axis=1).sum():,}")
print("\ncolumn types:")
print(raw.dtypes.value_counts().to_string())
print("\ncolumns:", ", ".join(raw.columns))

# Everything below looks at the non-empty rows only.
explore = raw.dropna(how="all")

# %% 1.2 The target: status
print(f"\n{LINE}\n1.2 The target variable: status\n{LINE}")
print(explore["status"].value_counts(dropna=False).to_string())
print("\ncolumns that could date an acquisition:",
      [c for c in explore.columns if "acqui" in c.lower()] or "none")
print("-> status is a single snapshot: we know THAT a company was acquired, not WHEN.")

# %% 1.3 Missing values
print(f"\n{LINE}\n1.3 Missing values (% of non-empty rows)\n{LINE}")
missing = (explore.isna().mean() * 100).round(1)
print(missing[missing > 0].sort_values(ascending=False).to_string())

# %% 1.4 How funding is stored
print(f"\n{LINE}\n1.4 How funding_total_usd is stored\n{LINE}")
print("sample values:", explore["funding_total_usd"].head(5).tolist())
print(f"values that are '-': {(explore['funding_total_usd'].str.strip() == '-').sum():,}")
print("-> text with Indian digit grouping (' 17,50,000 ' = 1,750,000) and '-' for missing.")
print("   The round columns (seed, venture, round_A, ...) are already numeric:")
print(explore[["seed", "venture", "round_A"]].describe().loc[["min", "50%", "max"]].to_string())

# A parsed copy, used only to look at the data in this part (Part 2 does it for real).
funding_peek = pd.to_numeric(
    explore["funding_total_usd"].str.strip().str.replace(",", "").replace("-", np.nan),
    errors="coerce",
)

# %% 1.5 Dates
print(f"\n{LINE}\n1.5 Dates\n{LINE}")
dates_peek = {c: pd.to_datetime(explore[c], errors="coerce")
              for c in ("founded_at", "first_funding_at", "last_funding_at")}
print("earliest first_funding_at (raw text):", explore["first_funding_at"].min())
print("latest last_funding_at:", explore["last_funding_at"].max(),
      f"-> the snapshot was taken around {SNAPSHOT_YEAR}")
print("founded_year range:", int(explore["founded_year"].min()), "-",
      int(explore["founded_year"].max()))

# %% 1.6 Acquisition rate by founding year
acquired_peek = (explore["status"] == "acquired").astype(int)

print(f"\n{LINE}\n1.6 Acquisition rate by founding year\n{LINE}")
by_year = (
    pd.DataFrame({"founded_year": explore["founded_year"], "acquired": acquired_peek,
                  "closed": (explore["status"] == "closed").astype(int)})
    .query("1995 <= founded_year <= @SNAPSHOT_YEAR")
    .groupby("founded_year")
    .agg(startups=("acquired", "size"), acquired_rate=("acquired", "mean"),
         closed_rate=("closed", "mean"))
)
by_year.index = by_year.index.astype(int)
print(by_year.round(3).to_string())
print("-> the rate collapses for young cohorts: they have simply had less time to be")
print("   acquired (right-censoring). A label that mixes all cohorts would mostly")
print("   teach the model to recognise young companies.")

# %% 1.7 From here on, look only at the labelled training cohorts (founded 2000-2005,
# status known), so no decision below is informed by the validation or test companies.
in_train = explore["founded_year"].isin(TRAIN_YEARS) & explore["status"].notna()
tr = explore[in_train]
tr_acquired = acquired_peek[in_train]
tr_funding = funding_peek[in_train]
base_rate = tr_acquired.mean()

print(f"\n{LINE}\n1.7 Funding amounts (training cohorts 2000-2005)\n{LINE}")
tr_missing = tr_funding.isna()
print(f"companies: {len(tr):,} | acquired rate: {base_rate:.1%}")
print(f"funding_total_usd missing: {tr_missing.sum():,} ({tr_missing.mean():.1%})")
print(f"acquired rate | missing: {tr_acquired[tr_missing].mean():.1%}"
      f" | present: {tr_acquired[~tr_missing].mean():.1%}")
print("missing totals with an amount in any round column:",
      int((tr.loc[tr_missing, FUNDING_TYPES + LETTERED_ROUNDS].sum(axis=1) > 0).sum()))
print("-> a missing total means 'nothing disclosed' and is informative, not random.\n")

amounts = pd.concat([tr_funding.rename("funding_total_usd"),
                     tr[["seed", "angel", "venture", "debt_financing"]]], axis=1)
print(pd.DataFrame({
    "share > 0": (amounts > 0).mean(),
    "median": amounts.median(),
    "mean": amounts.mean(),
    "max": amounts.max(),
    "skew": amounts.skew(),
}).to_string(float_format="{:,.2f}".format))
print("-> amounts span several orders of magnitude with heavy right skew and many zeros.")

# %% 1.8 Funding rounds
print(f"\n{LINE}\n1.8 Funding round types and stages (training cohorts)\n{LINE}")
present = tr[FUNDING_TYPES + LETTERED_ROUNDS].gt(0)
print(pd.DataFrame({
    "share of companies": present.mean(),
    "acquired rate if present": [tr_acquired[present[c]].mean() if present[c].any()
                                 else np.nan for c in present.columns],
}).round(3).to_string())
print(f"\noverall acquired rate: {base_rate:.1%}")
has_c = present["round_C"]
print(f"companies with a round C that also had a round B: "
      f"{present.loc[has_c, 'round_B'].mean():.0%}")
print("funding_rounds:", tr["funding_rounds"].describe()[["min", "50%", "max"]].to_dict())
print("-> lettered rounds are largely nested (C usually follows B): the furthest stage")
print("   reached says more than each round separately.")

# %% 1.9 Leakage checks
print(f"\n{LINE}\n1.9 Leakage checks (training cohorts)\n{LINE}")
post_ipo = tr[POST_IPO].gt(0).any(axis=1)
print(f"companies with post-IPO funding: {post_ipo.sum()}"
      f" (acquired rate {tr_acquired[post_ipo].mean():.1%})")
print("-> post-IPO money only exists after an IPO, which is itself an exit outcome.")
span_peek = (dates_peek["last_funding_at"] - dates_peek["first_funding_at"]).dt.days[in_train] / 365.25
print("\nfunding span (last - first funding, years) by acquired:")
print(span_peek.groupby(tr_acquired).describe()[["mean", "50%", "max"]].round(2).to_string())
print("-> if acquired companies stopped raising after the deal, their span would be")
print("   shorter. It is not, so the leak is plausible but unproven: Part 4 tests it.")

# %% 1.10 Founding-to-first-funding gap
print(f"\n{LINE}\n1.10 Years from founding to first funding (training cohorts)\n{LINE}")
gap_peek = (dates_peek["first_funding_at"] - dates_peek["founded_at"]).dt.days[in_train] / 365.25
print(gap_peek.describe()[["min", "25%", "50%", "75%", "max"]].round(2).to_string())
print(f"negative gaps (funded before founded): {(gap_peek < 0).sum()},"
      f" down to {gap_peek.min():.1f} years")
print("median gap by acquired:", gap_peek.clip(lower=0).groupby(tr_acquired).median().round(2).to_dict())
print("-> negatives are data-entry errors or pre-incorporation money; acquired companies")
print("   tend to raise sooner after founding.")

# %% 1.11 Market
print(f"\n{LINE}\n1.11 Market (training cohorts)\n{LINE}")
market_counts = tr["market"].str.strip().value_counts()
print(f"distinct markets: {len(market_counts)} | missing: {tr['market'].isna().sum()}")
print(f"top 10 cover {market_counts.head(10).sum() / len(tr):.0%},"
      f" top 60 cover {market_counts.head(60).sum() / len(tr):.0%},"
      f" markets with < 10 companies: {(market_counts < 10).sum()}")
print("\nthe 30 largest markets:")
print(market_counts.head(30).to_string())
print("-> far too many sparse categories to one-hot encode directly.")

# %% 1.12 Geography and breadth
print(f"\n{LINE}\n1.12 Geography and breadth (training cohorts)\n{LINE}")
by_country = (pd.DataFrame({"country": tr["country_code"], "acquired": tr_acquired})
              .groupby("country")["acquired"].agg(companies="size", acquired_rate="mean")
              .sort_values("companies", ascending=False))
print(f"countries: {len(by_country)}")
print(by_country.head(8).round(3).to_string())
print("\nlargest regions:")
by_region = (pd.DataFrame({"region": tr["region"], "acquired": tr_acquired})
             .groupby("region")["acquired"].agg(companies="size", acquired_rate="mean")
             .sort_values("companies", ascending=False))
print(by_region.head(8).round(3).to_string())
print(f"\nstate_code missing for {tr['state_code'].isna().mean():.0%} of rows")
tags_peek = tr["category_list"].fillna("").str.count(r"[^|]+")  # "|A|B|" -> 2 tags
print("category tags per company:", tags_peek.describe()[["mean", "50%", "max"]].round(2).to_dict())
print("-> the US dominates and a handful of startup hubs stand out; the long tail of")
print("   countries is too thin to model one by one.")

# %% 1.13 Duplicates
print(f"\n{LINE}\n1.13 Duplicates\n{LINE}")
print("duplicated permalinks:", explore["permalink"].duplicated().sum(),
      "| duplicated names:", explore["name"].duplicated().sum(),
      "(different companies can share a name; the permalink is the identifier)")


# =================================================================================
# PART 2 — PREPROCESSING
# =================================================================================
print(f"\n{LINE}\nPART 2 — Preprocessing\n{LINE}")

# %% 2.1 Drop the completely empty rows (1.1).
df = raw.dropna(how="all").copy()
print(f"non-empty rows: {len(df):,}")

# %% 2.2 Parse funding_total_usd from text (1.4). A '-' becomes NaN, kept for now.
df["funding_total_usd"] = pd.to_numeric(
    df["funding_total_usd"].str.strip().str.replace(",", "").replace("-", np.nan),
    errors="coerce",
)

# %% 2.3 Parse dates (1.5). errors="coerce" turns impossible dates (year 0001) into NaT.
for col in ("founded_at", "first_funding_at", "last_funding_at"):
    df[col] = pd.to_datetime(df[col], errors="coerce")
df["market"] = df["market"].str.strip()

# %% 2.4 Drop rows that cannot be labelled (no status, 1.2) or placed in a founding
# cohort (no founded_year, 1.3), and duplicate permalinks (1.13).
df = df.dropna(subset=["status", "founded_year"])
df["founded_year"] = df["founded_year"].astype(int)
df = df.drop_duplicates(subset="permalink")
print(f"after dropping missing status / founded_year and duplicates: {len(df):,}")

# %% 2.5 Target: acquired = 1, operating or closed = 0 (1.2).
df["acquired"] = (df["status"] == "acquired").astype(int)

# %% 2.6 Keep founding cohorts 2000-2008 only (1.6): every company has had at least
# six years to be acquired by the snapshot, so the label means roughly the same thing
# for all of them. company_age records how long each one has had (exposure control).
df = df[df["founded_year"].isin(COHORTS)].reset_index(drop=True)
df["company_age"] = SNAPSHOT_YEAR - df["founded_year"]
print(f"founding cohorts {COHORTS.start}-{COHORTS.stop - 1}: {len(df):,}")

# %% 2.7 Time-based split by founding cohort (1.6): score newer startups using what
# happened to older ones. A random split would mix cohorts and hide the drift.
is_train = df["founded_year"].isin(TRAIN_YEARS)
is_val = df["founded_year"].isin(VAL_YEARS)
is_test = df["founded_year"].isin(TEST_YEARS)

print(f"\n{'split':<7}{'founded':<11}{'rows':>7}{'acquired':>10}{'rate':>8}{'age':>9}")
for name, mask in (("train", is_train), ("val", is_val), ("test", is_test)):
    part = df[mask]
    print(f"{name:<7}{f'{part.founded_year.min()}-{part.founded_year.max()}':<11}"
          f"{len(part):>7,}{part.acquired.sum():>10,}{part.acquired.mean():>8.1%}"
          f"{part.company_age.min():>6}-{part.company_age.max()}")
print("the rate drifts down with less exposure. Base-rate drift alone would push")
print("predicted probabilities high on test, but company_age (learned on older")
print("cohorts) can pull them the other way — check calibration after fitting.")
print("Either way the top-k ranking rule is unaffected by the level shift.")


# =================================================================================
# PART 3 — FEATURE ENGINEERING
# =================================================================================
print(f"\n{LINE}\nPART 3 — Feature engineering\n{LINE}")

# Features are computed row by row, so building them on all cohorts at once is safe.
# The only thing learned from data (which countries get their own column, 3.6) is
# taken from the training rows.
features = pd.DataFrame(index=df.index)

# %% 3.1 Exposure control (1.6, 2.6).
features["company_age"] = df["company_age"]

# %% 3.2 Funding size and mix (1.7). A missing total always comes with no amount in
# any round column and lower acquisition rates: keep the rows, flag them, fill with 0.
# Amounts are heavily skewed with many zeros, so they enter as log1p(amount).
features["funding_disclosed"] = df["funding_total_usd"].notna().astype(int)
funding_total = df["funding_total_usd"].fillna(0.0)
features["log_funding_total"] = np.log1p(funding_total)
for col in ("seed", "angel", "venture", "debt_financing"):
    features[f"log_{col}"] = np.log1p(df[col])

# The mix: what share of the typed funding came from each source. post_ipo_* are left
# out of every funding feature (1.9).
typed_total = df[FUNDING_TYPES].sum(axis=1)
for col in ("venture", "debt_financing", "seed"):
    features[f"share_{col}"] = np.where(typed_total > 0, df[col] / typed_total.where(typed_total > 0, 1), 0.0)

# %% 3.3 Stage reached (1.8). Lettered rounds are nested, so keep the furthest one:
# A = 1 ... H = 8, none = 0. Round types become flags and a count of distinct types.
lettered = df[LETTERED_ROUNDS].gt(0).to_numpy()
features["max_round"] = np.where(
    lettered.any(axis=1), len(LETTERED_ROUNDS) - np.argmax(lettered[:, ::-1], axis=1), 0)
for col, name in (("seed", "has_seed"), ("angel", "has_angel"), ("venture", "has_venture"),
                  ("debt_financing", "has_debt"), ("grant", "has_grant")):
    features[name] = (df[col] > 0).astype(int)
features["n_round_types"] = df[FUNDING_TYPES].gt(0).sum(axis=1)
features["funding_rounds"] = df["funding_rounds"]

# %% 3.4 Early pace (1.10). Negative gaps are clipped to 0 ("funded at founding")
# rather than dropping the companies.
years_to_first = (df["first_funding_at"] - df["founded_at"]).dt.days / 365.25
features["years_to_first_funding"] = years_to_first.clip(lower=0)
features["log_funding_per_round"] = np.log1p(funding_total / df["funding_rounds"].clip(lower=1))

# %% 3.5 Sector (1.11). The most frequent markets are mapped by hand into nine broad
# sectors; anything else, and missing markets, is "other" (the reference level, so
# it gets no column of its own). The mapping is fixed in advance, not learned.
SECTORS = {
    "software_it": ["Software", "Enterprise Software", "SaaS", "Cloud Computing",
                    "Analytics", "Security", "Web Hosting", "Web Development",
                    "Information Technology", "Technology", "Internet", "Email"],
    "hardware_semis": ["Hardware + Software", "Hardware", "Semiconductors",
                       "Nanotechnology", "Networking", "Manufacturing"],
    "health_life_sciences": ["Biotechnology", "Health Care", "Health and Wellness",
                             "Medical", "Medical Devices", "Pharmaceuticals"],
    "mobile_telecom": ["Mobile", "Wireless", "Telecommunications", "Messaging"],
    "consumer_web_media": ["Curated Web", "Social Media", "News", "Search", "Games",
                           "Music", "Video", "Photography", "Media", "Digital Media",
                           "Entertainment"],
    "commerce_marketing": ["E-Commerce", "Advertising", "Fashion", "Public Relations",
                           "Sales and Marketing", "Customer Service"],
    "finance": ["Finance", "Financial Services", "Payments"],
    "cleantech_mobility": ["Clean Technology", "Automotive", "Transportation",
                           "Public Transportation"],
    "services_verticals": ["Consulting", "Education", "Hospitality", "Travel",
                           "Real Estate", "Sports", "Design", "Business Services"],
}
market_to_sector = {market: sector for sector, markets in SECTORS.items() for market in markets}
sector = df["market"].map(market_to_sector).fillna("other")
for name in SECTORS:
    features[f"sector_{name}"] = (sector == name).astype(int)

print("acquired rate by sector (train):")
print(df[is_train].groupby(sector[is_train])["acquired"]
      .agg(companies="size", acquired_rate="mean")
      .sort_values("companies", ascending=False).round(3).to_string())

# %% 3.6 Geography and breadth (1.12). One flag for the US, one per large non-US
# country (taken from the training rows), and one for the four largest startup hubs.
# state_code is not used: it is missing for about a third of companies.
top_countries = (df.loc[is_train & (df["country_code"] != "USA"), "country_code"]
                 .value_counts().head(5).index.tolist())
features["is_usa"] = (df["country_code"] == "USA").astype(int)
for country in top_countries:
    features[f"country_{country}"] = (df["country_code"] == country).astype(int)
hub_regions = ["SF Bay Area", "Boston", "New York City", "London"]
features["is_hub"] = df["region"].isin(hub_regions).astype(int)
features["n_categories"] = df["category_list"].fillna("").str.count(r"[^|]+")
print("\ncountries with their own column (from train):", top_countries)
print("hub regions:", hub_regions)

# %% 3.7 Date features (1.9). Built here, but only used in the variant that tests
# whether they leak the outcome (Part 4).
span = (df["last_funding_at"] - df["first_funding_at"]).dt.days / 365.25
features["funding_span_years"] = span
features["rounds_per_year"] = df["funding_rounds"] / span.clip(lower=1)
DATE_FEATURES = ["funding_span_years", "rounds_per_year"]

features = features.astype(float)
FEATURES_ALL = list(features.columns)
FEATURES_NO_DATES = [c for c in FEATURES_ALL if c not in DATE_FEATURES]
y = df["acquired"].to_numpy()


# =================================================================================
# PART 4 — FEATURE CHECKS
# =================================================================================

# %% 4.1 No missing values, sensible ranges, and how each feature differs by target
print(f"\n{LINE}\n4.1 Engineered features on train: mean by target\n{LINE}")
print(f"{len(FEATURES_ALL)} features | missing values train/val/test:",
      [int(features[m].isna().sum().sum()) for m in (is_train, is_val, is_test)])
train_features = features[is_train]
print(pd.DataFrame({
    "mean": train_features.mean(),
    "min": train_features.min(),
    "max": train_features.max(),
    "mean | acquired": train_features[df.loc[is_train, "acquired"] == 1].mean(),
    "mean | not acquired": train_features[df.loc[is_train, "acquired"] == 0].mean(),
}).round(2).to_string())

# %% 4.2 Do the date features help? The same logistic regression with and without
# them, scored on validation; the test set stays untouched until the final comparison.
# Each variant gets its own regularisation strength, tuned on validation over the same
# grid and with the same solver settings as the final models (5.2), so the comparison
# is between each feature set at its best, not at an arbitrary C.
print(f"\n{LINE}\n4.2 Date features: with vs without (validation)\n{LINE}")
k_val = int(round(TOP_SHARE * is_val.sum()))
variant_scores = {}
print(f"{'variant':<22}{'features':>9}{'best C':>8}{'val ROC-AUC':>13}{'val P@5%':>10}")
for label, columns in (("without date features", FEATURES_NO_DATES),
                       ("with date features", FEATURES_ALL)):
    scaler = StandardScaler().fit(features.loc[is_train, columns])
    X_probe_train = scaler.transform(features.loc[is_train, columns])
    X_probe_val = scaler.transform(features.loc[is_val, columns])
    variant_C, variant_auc = None, -np.inf
    for C in C_GRID:
        probe = LogisticRegression(C=C, max_iter=10_000, tol=1e-10).fit(X_probe_train, y[is_train])
        val_scores = probe.predict_proba(X_probe_val)[:, 1]
        if roc_auc_score(y[is_val], val_scores) > variant_auc:
            variant_C, variant_auc = C, roc_auc_score(y[is_val], val_scores)
            variant_scores[label] = val_scores
    top = np.argsort(-variant_scores[label])[:k_val]
    print(f"{label:<22}{len(columns):>9}{variant_C:>8}{variant_auc:>13.3f}"
          f"{y[is_val][top].mean():>10.3f}")
print(f"\nvalidation base rate {y[is_val].mean():.1%}; the top 5% is {k_val} companies,"
      f" so one extra hit moves P@5% by {1 / k_val:.3f}")

# %% 4.3 Pick the feature set for the three final models. A small gain on 1,770
# validation companies can be noise, so measure it: resample the validation set with
# replacement 1,000 times and recompute the ROC-AUC gain each time (a paired
# bootstrap: both variants are scored on the same resampled companies). The date
# features are kept only if the whole 95% interval of the gain is above zero;
# otherwise the safer set wins, because they also carry the leakage risk from 1.9.
print(f"\n{LINE}\n4.3 Choosing the feature set for the final models\n{LINE}")
rng = np.random.default_rng(0)
y_val = y[is_val]
gains = []
for _ in range(1000):
    sample = rng.integers(0, len(y_val), len(y_val))
    if y_val[sample].min() == y_val[sample].max():
        continue  # a resample with a single class has no ROC-AUC
    gains.append(roc_auc_score(y_val[sample], variant_scores["with date features"][sample])
                 - roc_auc_score(y_val[sample], variant_scores["without date features"][sample]))
gain_low, gain_high = np.percentile(gains, [2.5, 97.5])
observed_gain = (roc_auc_score(y_val, variant_scores["with date features"])
                 - roc_auc_score(y_val, variant_scores["without date features"]))
print(f"ROC-AUC gain from the date features: {observed_gain:+.3f}"
      f" (95% bootstrap interval {gain_low:+.3f} to {gain_high:+.3f})")

if gain_low > 0:
    FINAL_FEATURES = FEATURES_ALL
    print("-> the gain is clearly above noise: the final models use the date features.")
else:
    FINAL_FEATURES = FEATURES_NO_DATES
    print("-> the interval includes zero: the gain is indistinguishable from noise, and")
    print("   the date features carry a leakage risk (1.9). The final models use the")
    print("   set without them.")
print(f"final feature set: {len(FINAL_FEATURES)} features")


# =================================================================================
# PART 5 — ONE MODEL, THREE IMPLEMENTATIONS
# =================================================================================
# The target is binary, so the model is logistic regression. It is trained three ways
# on exactly the same standardised features and split: scikit-learn, a from-scratch
# PyTorch loop (Session 5: raw tensors, autograd, a manual update), and the standard
# torch.nn + torch.optim workflow. For the three to be comparable they must minimise
# the same objective:
#
#   scikit-learn:  0.5 * ||w||^2 + C * sum(log-loss)
#   divided by C*n: mean(log-loss) + ||w||^2 / (2*C*n)
#
# so both PyTorch versions minimise mean BCE + (lambda / 2) * ||w||^2 with
# lambda = 1 / (C * n_train), penalising the weights but not the bias, as sklearn does.
# Everything runs in float64 so that rounding does not blur the comparison.

# %% 5.1 Standardise. Gradient descent needs features on a common scale to converge
# (log funding is ~14, flags are 0/1); the scaler is fitted on the training rows only.
scaler = StandardScaler().fit(features.loc[is_train, FINAL_FEATURES])
X_train = scaler.transform(features.loc[is_train, FINAL_FEATURES])
X_val = scaler.transform(features.loc[is_val, FINAL_FEATURES])
X_test = scaler.transform(features.loc[is_test, FINAL_FEATURES])
y_train, y_val, y_test = y[is_train], y[is_val], y[is_test]
n_train, n_features = X_train.shape

# %% 5.2 Tune the regularisation strength C, the only hyperparameter worth touching
# for sklearn. Chosen by validation ROC-AUC, since the boutique ranks companies.
print(f"\n{LINE}\n5.2 Regularisation strength C (validation)\n{LINE}")
print(f"{'C':>8}{'val ROC-AUC':>13}{'val log-loss':>14}")
best_C, best_val_auc = None, -np.inf
for C in C_GRID:
    candidate = LogisticRegression(C=C, max_iter=10_000, tol=1e-10).fit(X_train, y_train)
    val_proba = candidate.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"{C:>8}{val_auc:>13.4f}{log_loss(y_val, val_proba):>14.4f}")
    if val_auc > best_val_auc:
        best_C, best_val_auc = C, val_auc
L2 = 1.0 / (best_C * n_train)
print(f"-> C = {best_C}, which is lambda = 1 / (C * n_train) = {L2:.2e} in the PyTorch versions")

# %% 5.3 Implementation 1: scikit-learn. A very tight tolerance so that it reaches the
# optimum, not just "close enough", which matters when we compare the three below.
started = time.perf_counter()
sk_model = LogisticRegression(C=best_C, max_iter=10_000, tol=1e-10).fit(X_train, y_train)
sk_seconds = time.perf_counter() - started
sk_w = sk_model.coef_.ravel()
sk_b = sk_model.intercept_[0]

# %% 5.4 Learning rate for the two PyTorch versions, the only other hyperparameter
# worth touching. Full-batch gradient descent: too small never arrives, too large
# overshoots. Each candidate runs briefly from zero; the lowest training loss wins.
X_train_t = torch.tensor(X_train, dtype=torch.float64)
y_train_t = torch.tensor(y_train, dtype=torch.float64)
EPOCHS = 100_000  # the problem is badly conditioned (see 5.7): 20,000 steps stop short

print(f"\n{LINE}\n5.4 Learning rate for gradient descent (training loss after 500 steps)\n{LINE}")
best_lr, best_lr_loss = None, np.inf
for lr in (0.01, 0.1, 0.5, 1.0):
    w_try = torch.zeros(n_features, dtype=torch.float64, requires_grad=True)
    b_try = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    for _ in range(500):
        p = torch.sigmoid(X_train_t @ w_try + b_try).clamp(1e-12, 1 - 1e-12)
        loss = (-(y_train_t * torch.log(p) + (1 - y_train_t) * torch.log(1 - p)).mean()
                + L2 / 2 * (w_try ** 2).sum())
        loss.backward()
        with torch.no_grad():
            w_try -= lr * w_try.grad
            b_try -= lr * b_try.grad
        w_try.grad.zero_()
        b_try.grad.zero_()
    final = loss.item()
    print(f"lr {lr:<6} loss {final:.6f}" + ("  (diverged)" if not np.isfinite(final) else ""))
    if np.isfinite(final) and final < best_lr_loss:
        best_lr, best_lr_loss = lr, final
LR = best_lr
print(f"-> lr = {LR}, then {EPOCHS:,} full-batch steps for both PyTorch versions")

# %% 5.5 Implementation 2: PyTorch by hand, exactly the Session 5 pattern. Raw weight
# and bias tensors, a hand-written forward pass and binary cross-entropy, .backward(),
# a manual update inside torch.no_grad(), and .grad.zero_() after every step.
torch.manual_seed(0)
w = torch.zeros(n_features, dtype=torch.float64, requires_grad=True)
b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
manual_loss_history = []

started = time.perf_counter()
for epoch in range(EPOCHS):
    y_pred = torch.sigmoid(X_train_t @ w + b).clamp(1e-12, 1 - 1e-12)  # forward pass
    bce = -(y_train_t * torch.log(y_pred) + (1 - y_train_t) * torch.log(1 - y_pred)).mean()
    loss = bce + L2 / 2 * (w ** 2).sum()                                 # same objective as sklearn
    loss.backward()                                                      # gradients via autograd
    with torch.no_grad():                                                # don't track the update
        w -= LR * w.grad
        b -= LR * b.grad
    w.grad.zero_()                                                       # or gradients accumulate
    b.grad.zero_()
    if epoch < 100 or epoch % 100 == 0 or epoch == EPOCHS - 1:  # dense early, where the drop is
        manual_loss_history.append((epoch, loss.item()))
manual_seconds = time.perf_counter() - started
manual_w = w.detach().numpy().copy()
manual_b = b.item()

# %% 5.6 Implementation 3: the standard workflow. nn.Linear holds the weights,
# BCEWithLogitsLoss fuses the sigmoid into the loss (numerically safer than taking
# log of a sigmoid), and torch.optim.SGD performs the update. weight_decay applies
# lambda * w to the gradient, which is exactly the gradient of (lambda/2) * ||w||^2;
# it is set on the weight only, so the bias stays unpenalised as in the other two.
torch.manual_seed(0)
std_model = torch.nn.Linear(n_features, 1, dtype=torch.float64)
torch.nn.init.zeros_(std_model.weight)  # same zero start as the manual loop
torch.nn.init.zeros_(std_model.bias)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(
    [{"params": [std_model.weight], "weight_decay": L2},
     {"params": [std_model.bias], "weight_decay": 0.0}],
    lr=LR,
)
std_loss_history = []

started = time.perf_counter()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    logits = std_model(X_train_t).squeeze(1)
    bce = loss_fn(logits, y_train_t)
    bce.backward()
    optimizer.step()
    if epoch < 100 or epoch % 100 == 0 or epoch == EPOCHS - 1:  # dense early, where the drop is
        # report the same penalised objective as the manual loop, for comparable curves
        with torch.no_grad():
            penalised = bce.item() + L2 / 2 * (std_model.weight ** 2).sum().item()
        std_loss_history.append((epoch, penalised))
std_seconds = time.perf_counter() - started
std_w = std_model.weight.detach().numpy().ravel().copy()
std_b = std_model.bias.item()

# %% 5.7 Do the three agree? Same objective, same data: they should land on the same
# weights up to optimisation tolerance. The test probabilities are compared too.
X_test_t = torch.tensor(X_test, dtype=torch.float64)
test_proba = {
    "scikit-learn": sk_model.predict_proba(X_test)[:, 1],
    "PyTorch (manual)": torch.sigmoid(X_test_t @ torch.tensor(manual_w) + manual_b).numpy(),
    "PyTorch (nn + optim)": torch.sigmoid(std_model(X_test_t).squeeze(1)).detach().numpy(),
}
sk_objective = (log_loss(y_train, sk_model.predict_proba(X_train)[:, 1])
                + L2 / 2 * (sk_w ** 2).sum())

print(f"\n{LINE}\n5.7 Do the three implementations agree?\n{LINE}")
print(f"{'implementation':<22}{'train objective':>17}{'max |w - w_sklearn|':>21}"
      f"{'|b - b_sklearn|':>17}{'seconds':>9}")
for name, weights, bias, objective, seconds in (
    ("scikit-learn", sk_w, sk_b, sk_objective, sk_seconds),
    ("PyTorch (manual)", manual_w, manual_b, manual_loss_history[-1][1], manual_seconds),
    ("PyTorch (nn + optim)", std_w, std_b, std_loss_history[-1][1], std_seconds),
):
    print(f"{name:<22}{objective:>17.8f}{np.abs(weights - sk_w).max():>21.2e}"
          f"{abs(bias - sk_b):>17.2e}{seconds:>9.1f}")
# Why gradient descent needs so many steps: correlated features make the loss surface
# a long, narrow valley. The condition number of X'X/n measures how stretched it is.
eigenvalues = np.linalg.eigvalsh(X_train.T @ X_train / n_train)
feature_corr = np.corrcoef(X_train, rowvar=False)
i, j = np.unravel_index(np.abs(np.triu(feature_corr, 1)).argmax(), feature_corr.shape)
print(f"\ncondition number of X'X/n: {eigenvalues.max() / eigenvalues.min():,.0f}"
      f" | most correlated pair: {FINAL_FEATURES[i]} / {FINAL_FEATURES[j]}"
      f" (r = {feature_corr[i, j]:.3f})")
print("-> gradient descent crawls along that valley, so the PyTorch versions need many")
print(f"   more steps than sklearn's solver; after {EPOCHS:,} they reach the same optimum.")

print("\nlargest difference in test probabilities vs scikit-learn:")
for name in ("PyTorch (manual)", "PyTorch (nn + optim)"):
    print(f"  {name:<22}{np.abs(test_proba[name] - test_proba['scikit-learn']).max():.2e}")
print(f"  manual vs nn + optim  "
      f"{np.abs(test_proba['PyTorch (manual)'] - test_proba['PyTorch (nn + optim)']).max():.2e}")

print("\ncoefficients (standardised features: change in log-odds per 1 std):")
coefficients = pd.DataFrame({"scikit-learn": sk_w, "PyTorch (manual)": manual_w,
                             "PyTorch (nn + optim)": std_w}, index=FINAL_FEATURES)
print(coefficients.reindex(coefficients["scikit-learn"].abs().sort_values(ascending=False).index)
      .round(4).to_string())

# %% 5.8 Evaluate on the untouched test set, against two naive baselines:
#   base rate:       every company gets the training acquisition rate (a random ranking)
#   funding only:    rank by total funding alone, the obvious banker heuristic
# The operating rule from REPORT.md is "pitch the top 5% of the ranked list".
test_scores = {
    "baseline: base rate": np.full(len(y_test), y_train.mean()),
    "baseline: funding only": features.loc[is_test, "log_funding_total"].to_numpy(),
    **test_proba,
}
k_test = int(round(TOP_SHARE * len(y_test)))

print(f"\n{LINE}\n5.8 Test set ({len(y_test):,} startups founded 2007-2008,"
      f" {y_test.sum()} acquired; top 5% = {k_test} pitches)\n{LINE}")
results = []
for name, scores in test_scores.items():
    if name == "baseline: base rate":
        auc = 0.5
        expected_hits = y_test.mean() * k_test  # a constant score is a random ranking
        precision_k = y_test.mean()
    else:
        auc = roc_auc_score(y_test, scores)
        pitched = np.argsort(-scores, kind="stable")[:k_test]
        expected_hits = y_test[pitched].sum()
        precision_k = y_test[pitched].mean()
    false_positives = k_test - expected_hits
    false_negatives = y_test.sum() - expected_hits
    results.append({
        "model": name,
        "ROC-AUC": round(auc, 4),
        "precision@5%": round(precision_k, 4),
        "recall@5%": round(expected_hits / y_test.sum(), 4),
        "acquirers pitched": round(expected_hits, 1),
        "cost (EUR)": round(false_positives * COST_FP + false_negatives * COST_FN),
    })
results = pd.DataFrame(results).set_index("model")
print(results.to_string())
# Calibration on test: base-rate drift alone would push means high (train rate >
# test rate), but a positive company_age coefficient extrapolated to younger test
# cohorts pulls the other way — so report the observed direction, not the guess.
mean_predicted = test_proba["scikit-learn"].mean()
print(f"\nmean predicted probability on test {mean_predicted:.3f} vs actual rate {y_test.mean():.3f}:"
      f" the probabilities run {'high' if mean_predicted > y_test.mean() else 'low'}"
      f" (company_age coefficient {coefficients.loc['company_age', 'scikit-learn']:+.3f}"
      f" on standardised age; see 5.7).")
print("The top-5% rule depends only on the ranking, so it is unaffected either way.")


# =================================================================================
# PART 6 — SAVE EVERYTHING THE DASHBOARD NEEDS
# =================================================================================
# app.py only loads these files; it never trains anything.
ARTIFACTS.mkdir(exist_ok=True)

joblib.dump(sk_model, ARTIFACTS / "model_sklearn.joblib")
torch.save({"w": torch.tensor(manual_w), "b": torch.tensor([manual_b])},
           ARTIFACTS / "model_torch_manual.pt")
torch.save(std_model.state_dict(), ARTIFACTS / "model_torch_nn.pt")
joblib.dump(scaler, ARTIFACTS / "scaler.joblib")

# Test predictions next to the truth, for prediction-vs-actual and the threshold slider.
predictions = df.loc[is_test, ["name", "founded_year", "market", "country_code", "acquired"]].copy()
predictions["sector"] = sector[is_test]
for name, scores in test_scores.items():
    predictions[name] = scores
predictions.to_csv(ARTIFACTS / "test_predictions.csv", index=False)

# Unscaled features with split and target, for the distribution views.
feature_table = features[FINAL_FEATURES].copy()
feature_table["split"] = np.select([is_train, is_val, is_test], ["train", "val", "test"], default="")
feature_table["acquired"] = y
feature_table.to_csv(ARTIFACTS / "features.csv", index=False)

results.to_csv(ARTIFACTS / "results.csv")
coefficients.to_csv(ARTIFACTS / "coefficients.csv")
with open(ARTIFACTS / "metadata.json", "w") as f:
    json.dump({
        "features": FINAL_FEATURES,
        "C": best_C,
        "lambda": L2,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "top_share": TOP_SHARE,
        "cost_fp": COST_FP,
        "cost_fn": COST_FN,
        "train_base_rate": float(y_train.mean()),
        "test_base_rate": float(y_test.mean()),
        "sklearn_objective": float(sk_objective),  # the optimum the PyTorch curves approach
        "loss_history": {"PyTorch (manual)": manual_loss_history,
                         "PyTorch (nn + optim)": std_loss_history},
    }, f, indent=2)

print(f"\n{LINE}\nSaved to {ARTIFACTS.relative_to(ROOT)}/: "
      + ", ".join(sorted(p.name for p in ARTIFACTS.iterdir())) + f"\n{LINE}")
