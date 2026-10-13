from pathlib import Path
import pandas as pd

URL = "https://zenodo.org/records/11295916/files/LC_loans_granting_model_dataset.csv?download=1"
YEARS = range(2012, 2016)  # ventana de años (decisión tuya, justifícala en el REPORT)
N_ROWS = 20_000
SEED = 42

out = Path("data/loans_sample.csv")
out.parent.mkdir(exist_ok=True)

print("Descargando el dataset (~167 MB, tarda un poco)...")
df = pd.read_csv(URL, low_memory=False)
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
df = df[df["issue_d"].dt.year.isin(YEARS)]

sample = df.sample(n=N_ROWS, random_state=SEED).sort_values("issue_d")
sample.to_csv(out, index=False)

print(f"Guardado {out}: {len(sample)} filas, {out.stat().st_size / 1e6:.1f} MB")
print(sample.groupby(sample["issue_d"].dt.year)["Default"].agg(["size", "mean"]).round(3))