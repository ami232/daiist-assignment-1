import numpy as np
import pandas as pd

# keep "NA" as text (means "feature absent"); only truly empty cells are missing
df = pd.read_csv("data/AmesHousing.csv", keep_default_na=False, na_values=[""])

print("Shape:", df.shape)
print("\n=== Column types ===")
print(df.dtypes.value_counts())

print("\n=== Truly missing (empty) cells ===")
miss = df.isna().sum()
print(miss[miss > 0].sort_values(ascending=False))

print("\n=== Columns where the literal 'NA' category appears ===")
na_counts = {c: int((df[c] == "NA").sum()) for c in df.select_dtypes("object")}
print({c: n for c, n in na_counts.items() if n > 0})

print("\n=== SalePrice ===")
print(df["SalePrice"].describe())
print("Skew (raw):", round(df["SalePrice"].skew(), 2))
print("Skew (log):", round(np.log(df["SalePrice"]).skew(), 2))

print("\n=== Sales per year ===")
print(df.groupby("Yr Sold")["SalePrice"].agg(["count", "median", "mean"]).round(0))

print("\n=== Sales per month in the last year ===")
last = df["Yr Sold"].max()
print(df[df["Yr Sold"] == last]["Mo Sold"].value_counts().sort_index())

print("\n=== 5 largest houses by Gr Liv Area ===")
print(df.nlargest(5, "Gr Liv Area")[["Gr Liv Area", "SalePrice", "Sale Condition"]])

print("\n=== Numeric columns most correlated with SalePrice ===")
corr = df.select_dtypes("number").corr()["SalePrice"].drop("SalePrice")
print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(10).round(2))
