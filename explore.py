import pandas as pd

# Load the data 
df = pd.read_csv("data/hotel_bookings.csv")

# 1. Size and column types
print("Shape (rows, columns):", df.shape)
print(df.dtypes)

# 2. The target: how many bookings were cancelled?
print("\nOverall cancellation rate:", round(df["is_canceled"].mean(), 3))
print("\nCancellation rate by hotel:")
print(df.groupby("hotel")["is_canceled"].mean().round(3))

# 3. Bookings per year and hotel
print("\nBookings per year and hotel:")
print(pd.crosstab(df["arrival_date_year"], df["hotel"]))

# 4. Missing values
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0])

# 5. LEAKAGE CHECK: compare reservation_status with the target
print("\nreservation_status vs is_canceled:")
print(pd.crosstab(df["reservation_status"], df["is_canceled"]))

# 6. Look at a few real rows
print(df.head(5).T)

# --- Evidence for the 14-day refund deadline (City Hotel) ---
city = df[df["hotel"] == "City Hotel"].copy()

# Build the real arrival date from the three date columns
city["arrival_date"] = pd.to_datetime(
    city["arrival_date_year"].astype(str) + "-"
    + city["arrival_date_month"] + "-"
    + city["arrival_date_day_of_month"].astype(str),
    format="%Y-%B-%d",
)

# For cancelled bookings, how many days before arrival did they cancel?
cancelled = city[city["reservation_status"] == "Canceled"].copy()
cancelled["days_before_arrival"] = (
    cancelled["arrival_date"] - pd.to_datetime(cancelled["reservation_status_date"])
).dt.days

print("\nShare of cancellations made within X days of arrival:")
for days in [7, 14, 30]:
    share = (cancelled["days_before_arrival"] <= days).mean()
    print(f"  within {days} days: {share:.1%}")

print("\nShare of bookings made less than 14 days ahead:",
      f"{(city['lead_time'] < 14).mean():.1%}")