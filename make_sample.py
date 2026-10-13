"""
One-off helper: builds data/car_prices_sample.csv from the full Kaggle CSV
(88 MB, NOT committed). Not part of the graded pipeline -- the committed
sample is what train.py reads. Run only if you want to regenerate it:

    uv run python make_sample.py /path/to/car_prices.csv
"""
import sys
import pandas as pd

SEED = 42
N_TRAIN, N_VAL, N_TEST = 9000, 2500, 2500   # ~14k rows total (< 20k guide)

src = sys.argv[1] if len(sys.argv) > 1 else "car_prices.csv"
df = pd.read_csv(src)

# sale date: "Tue Dec 16 2014 12:30:00 GMT-0800 (PST)" -> date
d = df["saledate"].str.extract(r"^\w{3} (\w{3}) (\d{2}) (\d{4})")
df["sale_date"] = pd.to_datetime(d[0] + " " + d[1] + " " + d[2], format="%b %d %Y", errors="coerce")
df = df.dropna(subset=["sale_date", "mmr", "sellingprice"])

# data-quality filter on a variable known BEFORE the sale (mmr). We deliberately do NOT
# filter on sellingprice: that would remove bad outcomes from the test set (target leakage).
df = df[df["mmr"] >= 1000]

# temporal windows (framing: train Dec-Mar, validate May, test June)
def window(start, end, n):
    w = df[(df["sale_date"] >= start) & (df["sale_date"] <= end)]
    return w.sample(n=n, random_state=SEED)

parts = [
    window("2014-12-01", "2015-03-31", N_TRAIN),
    window("2015-05-01", "2015-05-31", N_VAL),
    window("2015-06-01", "2015-06-30", N_TEST),
]
out = pd.concat(parts).sort_values("sale_date").reset_index(drop=True)
out = out.drop(columns=["saledate"])   # VIN kept ONLY to audit train/test overlap; never a feature
out.to_csv("data/car_prices_sample.csv", index=False)
print(out.shape, out["sale_date"].min(), out["sale_date"].max())

# audit in the FULL (filtered) data: how many VINs are sold more than once?
full = df.dropna(subset=["vin"])
rep = full["vin"].duplicated(keep=False)
print(f"full data: {rep.mean():.1%} of sales belong to a VIN that appears more than once")
