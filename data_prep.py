import pandas as pd

DATA_PATH = "data/songs_2000_2020.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    # data-quality exclusion: June 2020 labels are on a different scale (mean 36 vs ~70)
    df = df[~df.release_date.str.startswith("2020-06")].reset_index(drop=True)
    # target = popularity relative to the release-year cohort
    df["target"] = df.popularity - df.groupby("year").popularity.transform("mean")
    return df


def split(df):
    train = df[df.year <= 2015].reset_index(drop=True)
    val = df[(df.year >= 2016) & (df.year <= 2017)].reset_index(drop=True)
    test = df[df.year >= 2018].reset_index(drop=True)
    return train, val, test
