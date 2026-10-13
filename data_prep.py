"""
data_prep.py - Step 3: load, filter, clean and remove leakage.

"""
import pandas as pd

# ---------------------------------------------------------------
# DECISIONS (each one is explained in REPORT.md)
# ---------------------------------------------------------------
DATA_PATH = "data/hotel_bookings.csv"
HOTEL = "City Hotel"        # our client
SAMPLE_SIZE = 20_000        # brief suggests ~20k rows
RANDOM_STATE = 42           # fixed seed -> same sample every run
MAX_ADR = 1000              # price per night above this = data error

# Columns the hotel would NOT know on the day of booking (leakage)
LEAKAGE_COLUMNS = [
    "reservation_status",       # the final outcome = the answer itself
    "reservation_status_date",  # date of cancellation / check-out (future)
    "assigned_room_type",       # room given at check-in (future)
    "booking_changes",          # changes made after booking (future)
]

# Columns removed for business-framing reasons (not leakage)
POLICY_COLUMNS = [
    "deposit_type",  # the model is meant to DECIDE who pays a deposit
]

# Columns that carry no information once we keep one hotel
CONSTANT_COLUMNS = ["hotel"]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the original CSV, untouched."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Filter to our hotel, fix data-quality problems, drop leaky columns."""
    log = print if verbose else (lambda *a, **k: None)
    log(f"Raw data: {len(df):,} rows")

    # 1. Keep only our client
    df = df[df["hotel"] == HOTEL].copy()
    log(f"After keeping {HOTEL}: {len(df):,} rows")

    # 2. Real arrival date (needed later for the time-based split)
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-"
        + df["arrival_date_month"] + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )

    # 3. Missing values
    df["children"] = df["children"].fillna(0)          # 4 rows: assume no children
    df["country"] = df["country"].fillna("Unknown")    # unknown origin is its own group
    # agent / company: NaN means "no agent / no company" -> handled in Step 4

    # 4. Impossible or broken bookings
    nights = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    guests = df["adults"] + df["children"] + df["babies"]
    bad = (nights == 0) | (guests == 0) | (df["adr"] < 0) | (df["adr"] > MAX_ADR)
    log(f"Removing {bad.sum():,} broken rows (0 nights, 0 guests or impossible price)")
    df = df[~bad]

    # 5. Remove leakage, policy and constant columns
    df = df.drop(columns=LEAKAGE_COLUMNS + POLICY_COLUMNS + CONSTANT_COLUMNS)

    # 6. Random sample to a manageable size (fixed seed = reproducible)
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    # 7. Put rows back in time order (matters for the split in Step 5)
    df = df.sort_values("arrival_date").reset_index(drop=True)

    log(f"Final clean sample: {len(df):,} rows, {df.shape[1]} columns")
    log(f"Cancellation rate in sample: {df['is_canceled'].mean():.1%}")
    return df


if __name__ == "__main__":
    # Lets you check this step on its own: uv run python data_prep.py
    data = clean(load_raw())
    print("\nColumns kept:")
    print(list(data.columns))
    print("\nArrival dates from", data["arrival_date"].min().date(),
          "to", data["arrival_date"].max().date())