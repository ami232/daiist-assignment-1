"""
features.py - Step 4: feature engineering.

Every new feature is built from ONE row at a time (no averages or counts
over the whole dataset). That means no information from the test period
can leak into training through these features.
"""
import numpy as np
import pandas as pd

TARGET = "is_canceled"

# Numeric columns that go into the model
NUMERIC_FEATURES = [
    "log_lead_time",                   # F
    "total_nights",                    # A
    "adults",
    "children",
    "babies",
    "adr",
    "revenue_at_risk",                 # J
    "has_agent",                       # D
    "has_company",                     # D
    "cancelled_before",                # E
    "is_repeated_guest",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
]

# Text columns that go into the model (turned into numbers in Step 5)
CATEGORICAL_FEATURES = [
    "arrival_weekday",                 # G
    "country",                         # C + I (grouped in Step 5)
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "customer_type",
]

# Columns dropped on purpose in this step (explained in REPORT.md)
DROPPED_COLUMNS = [
    "required_car_parking_spaces",  # 0% cancellations among 492 bookings -> likely recorded at arrival (leakage)
    "total_of_special_requests",    # can be added after booking -> possible leakage, dropped to be safe
    "arrival_date_year",            # test period is later -> model would meet a year it never saw
    "arrival_date_week_number",     # duplicates month / date info
    "arrival_date_day_of_month",    # no business meaning on its own
    "arrival_date_month",           # weak effect (36-46%); City demand is not very seasonal
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the new features. Works row by row, so it is leak-free."""
    df = df.copy()

    # A. Length of stay in one number
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

    # D. Agent / company: the ID number is just a label, so keep yes/no
    df["has_agent"] = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)

    # E. Has this guest ever cancelled before?
    df["cancelled_before"] = (df["previous_cancellations"] > 0).astype(int)

    # F. Lead time is very skewed (0 to 600+ days) -> log squashes big values
    #    log1p(x) = log(1 + x), so 0 days stays 0 instead of -infinity
    df["log_lead_time"] = np.log1p(df["lead_time"])

    # G. Day of the week the guest arrives (Monday, Tuesday, ...)
    df["arrival_weekday"] = df["arrival_date"].dt.day_name()

    # J. Money at stake if this booking cancels
    df["revenue_at_risk"] = df["adr"] * df["total_nights"]

    return df


def get_X_y(df: pd.DataFrame):
    """Split a feature table into model inputs (X) and the target (y)."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)
    return X, y


if __name__ == "__main__":
    # Check this step on its own: uv run python features.py
    from data_prep import clean, load_raw

    data = add_features(clean(load_raw(), verbose=False))
    X, y = get_X_y(data)

    print("X shape:", X.shape, "| y shape:", y.shape)
    print("\nNew features (first 5 rows):")
    print(data[["total_nights", "has_agent", "has_company", "cancelled_before",
                "log_lead_time", "arrival_weekday", "revenue_at_risk"]].head())

    print("\nCancellation rate by new yes/no feature:")
    for col in ["has_agent", "has_company", "cancelled_before"]:
        print(f"  {col}:", data.groupby(col)[TARGET].mean().round(3).to_dict())

    print("\nCancellation rate by arrival weekday:")
    print(data.groupby("arrival_weekday")[TARGET].mean().round(3))

    print("\nMissing values in X:", int(X.isna().sum().sum()))