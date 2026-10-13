import pandas as pd

URL = "https://jse.amstat.org/v19n3/decock/AmesHousing.txt"

# keep_default_na=False keeps the literal "NA" strings (they mean "feature absent")
df = pd.read_csv(URL, sep="\t", keep_default_na=False, na_values=[""])
df.to_csv("data/AmesHousing.csv", index=False)
print(df.shape)
