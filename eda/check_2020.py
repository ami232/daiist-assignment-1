from data_prep import load_data

df = load_data()
for y in [2018, 2019, 2020]:
    d = df[df.year == y]
    print(y, "popularity quantiles:", d.popularity.quantile([0, .05, .25, .5, .75, .95, 1]).round(0).tolist())
    print(y, "share with popularity <= 5:", round((d.popularity <= 5).mean(), 3))

d = df[df.year == 2020].copy()
d["month"] = d.release_date.str[5:7]
print(d.groupby("month").popularity.agg(["count", "mean", "std"]).round(1))