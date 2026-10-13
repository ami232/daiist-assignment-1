import os
import pandas as pd

df = pd.read_csv("data_raw/data.csv")
df = df[(df.year >= 2000) & (df.year <= 2020)]
df = df.drop_duplicates(["artists", "name"])

# Same number of songs per year (max 850), so no year weighs more than another
parts = [
    g.sample(min(len(g), 850), random_state=42)
    for _, g in df.groupby("year")
]
sample = pd.concat(parts).sort_values("year").reset_index(drop=True)

os.makedirs("data", exist_ok=True)
sample.to_csv("data/songs_2000_2020.csv", index=False)
print(sample.shape, sample.groupby("year").size().min())