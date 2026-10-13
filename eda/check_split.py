import ast
from data_prep import load_data, split

df = load_data()
tr, va, te = split(df)
print("sizes:", len(tr), len(va), len(te))
print("target std:", round(tr.target.std(), 1), round(va.target.std(), 1), round(te.target.std(), 1))

yo = df.assign(year_only=df.release_date.str.len() == 4)
print("share of year-only dates by year:")
print(yo.groupby("year").year_only.mean().round(2).to_string())

arts = lambda s: ast.literal_eval(s)
seen = {a for s in tr.artists for a in arts(s)}
for name, d in [("val", va), ("test", te)]:
    share = d.artists.apply(lambda s: any(a in seen for a in arts(s))).mean()
    print(name, "songs with an artist seen in train:", round(share, 3))

print(tr.artists.head(3).tolist())
