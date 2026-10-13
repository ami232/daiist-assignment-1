"""
Feature engineering for the restaurant-inspection model.

Shared by train.py (to build the training table) and app.py (to score a
what-if restaurant with exactly the same transformations).

One rule runs through this whole file: a feature may only use information
that was already known the day before the inspection it describes. Every
history feature therefore looks at inspections dated strictly before the
current one, and every constant that is learned from data (medians, the
base fail rate, chain sizes, the scaler) is learned on the training period
only.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "chicago_restaurant_inspections.csv"

# --- Decisions that come from the business framing (see REPORT.md) -----------
MODEL_START = pd.Timestamp("2022-07-01")  # first inspection the model has to predict
VAL_START = pd.Timestamp("2024-07-01")    # train = 2 years before this date
TEST_START = pd.Timestamp("2025-07-01")   # validation = 1 year, test = the final year

LOOKBACK_DAYS = 3 * 365   # how far back a restaurant's own record counts
AREA_WINDOW_DAYS = 365    # window for the ZIP code's track record
OWN_PRIOR_WEIGHT = 2      # pseudo-inspections used to smooth a restaurant's fail rate
AREA_PRIOR_WEIGHT = 20    # pseudo-inspections used to smooth a ZIP code's fail rate

# An inspection only has an outcome if the inspector actually got in and the
# business was operating. "Out of Business", "No Entry", "Not Ready" and
# "Business Not Located" are not passes or fails, so they are never targets.
DEFINITE_RESULTS = ["Pass", "Pass w/ Conditions", "Fail"]
COMPLAINT_TYPES = ["Complaint", "Short Form Complaint", "Suspected Food Poisoning"]

# Violation code numbers from the City of Chicago inspection form (July 2018 on).
PEST_CODES = {38}                                   # insects, rodents & animals not present
TEMPERATURE_CODES = {18, 19, 20, 21, 22, 23, 24, 33}  # cooking, holding, cooling, date marking
MANAGEMENT_CODES = {1, 2, 3, 5}                     # person in charge, certificate, staff knowledge

# name, group, short label, what it means. The order here is the column order
# of the model matrix everywhere (training, saved weights, the dashboard).
FEATURES = [
    ("risk_high", "Static profile", "High-risk class",
     "1 if the city classes the restaurant as Risk 1 (High): it handles raw food and cooks from scratch."),
    ("log_chain_size", "Static profile", "Chain size (log)",
     "Log of how many licensed restaurants share this name. Big chains run standardised kitchens."),
    ("month_sin", "Static profile", "Season (sine)",
     "Month of the year as a point on a circle, so December sits next to January."),
    ("month_cos", "Static profile", "Season (cosine)",
     "Second coordinate of the same circle. Together they let the model learn one smooth yearly wave."),
    ("zip_fail_rate_12m", "Area track record", "ZIP fail rate, last 12 months",
     "Share of routine inspections in the same ZIP code that failed in the previous 12 months, "
     "shrunk toward the city average when the ZIP has few inspections."),
    ("fail_rate_3y", "Own record", "Own fail rate, last 3 years",
     "The restaurant's own share of failed inspections, smoothed so that one inspection cannot swing it to 0% or 100%."),
    ("log_n_prior", "Own record", "Past inspections (log)",
     "Log of the number of inspections with a result in the last 3 years."),
    ("no_history", "Own record", "No history",
     "1 if the restaurant has no inspection with an outcome in the previous 3 years, "
     "so the history features are filled with typical values."),
    ("prev_canvass_fail", "Own record", "Last routine inspection failed",
     "1 if its most recent routine inspection ended in a fail. Re-inspections are ignored because they nearly always pass."),
    ("log_days_since_last", "Own record", "Days since last inspection (log)",
     "Log of the days since anyone last inspected it."),
    ("log_complaints", "Own record", "Complaint visits (log)",
     "Log of the number of inspections triggered by public complaints in the last 3 years."),
    ("mean_viol_prior", "Violation history", "Avg violations per inspection",
     "Average number of violations cited per inspection over the last 3 years."),
    ("prev_n_viol", "Violation history", "Violations at last inspection",
     "Number of violations cited at the most recent inspection."),
    ("pest_any", "Violation history", "Pests cited before",
     "1 if insects or rodents were cited at any inspection in the last 3 years."),
    ("temp_any", "Violation history", "Temperature control cited before",
     "1 if a cooking, holding, cooling or date-marking violation was cited in the last 3 years."),
    ("mgmt_any", "Violation history", "Management issues cited before",
     "1 if a person-in-charge, certification or staff-knowledge violation was cited in the last 3 years."),
]
FEATURE_NAMES = [name for name, _, _, _ in FEATURES]
FEATURE_GROUPS = list(dict.fromkeys(group for _, group, _, _ in FEATURES))
FEATURE_LABELS = {name: label for name, _, label, _ in FEATURES}

# Columns in human units, before imputation and log/cyclical transforms.
RAW_COLUMNS = [
    "risk", "chain_size", "month", "zip_fail_rate_12m", "n_prior", "n_fail_prior",
    "prev_canvass_fail", "days_since_last", "n_complaints", "mean_viol_prior",
    "prev_n_viol", "pest_any", "temp_any", "mgmt_any",
]


# --- Loading and cleaning ------------------------------------------------------
def parse_codes(text: str) -> list[int]:
    return [int(code) for code in text.split("|")] if text else []


def load_inspections(path: Path = DATA_PATH) -> pd.DataFrame:
    """Every restaurant inspection in the file, with per-inspection summaries."""
    df = pd.read_csv(
        path,
        dtype={"license": "string", "zip": "string", "violation_codes": "string"},
        parse_dates=["inspection_date"],
    )
    # Data quality: a few dozen rows have no usable licence number, so they
    # cannot be linked to any restaurant's history. They are dropped.
    df = df[df["license"].notna() & (df["license"] != "0")].copy()

    codes = df["violation_codes"].fillna("").map(parse_codes)
    df["n_violations"] = codes.map(len)
    df["pest"] = codes.map(lambda cited: int(bool(PEST_CODES & set(cited))))
    df["temp"] = codes.map(lambda cited: int(bool(TEMPERATURE_CODES & set(cited))))
    df["mgmt"] = codes.map(lambda cited: int(bool(MANAGEMENT_CODES & set(cited))))

    df["has_result"] = df["results"].isin(DEFINITE_RESULTS)
    df["failed"] = (df["results"] == "Fail").astype(int)
    df["is_canvass"] = df["inspection_type"] == "Canvass"
    df["is_complaint"] = df["inspection_type"].isin(COMPLAINT_TYPES)
    return df.sort_values(["inspection_date", "inspection_id"]).reset_index(drop=True)


def select_model_rows(inspections: pd.DataFrame) -> pd.DataFrame:
    """The inspections the model predicts: routine (canvass) visits with a result.

    Canvass inspections are the ones the city schedules at its own
    discretion, which is exactly the decision the model supports. Complaint
    and licence inspections are triggered by someone else, and
    re-inspections are a consequence of an earlier fail.
    """
    rows = inspections[
        inspections["is_canvass"]
        & inspections["has_result"]
        & (inspections["inspection_date"] >= MODEL_START)
    ]
    # A handful of restaurants have two canvass rows on the same day; keep one.
    rows = rows.drop_duplicates(["license", "inspection_date"])
    return rows.reset_index(drop=True)


# --- History features ----------------------------------------------------------
def add_own_record(rows: pd.DataFrame, inspections: pd.DataFrame) -> pd.DataFrame:
    """Summaries of each restaurant's own inspections in the LOOKBACK_DAYS before."""
    past = inspections[[
        "license", "inspection_id", "inspection_date", "has_result", "failed",
        "is_canvass", "is_complaint", "n_violations", "pest", "temp", "mgmt",
    ]].rename(columns={"inspection_id": "past_id", "inspection_date": "past_date"})

    # Pair every inspection with all inspections of the same restaurant, then
    # keep only the ones strictly earlier and inside the lookback window.
    pairs = rows[["inspection_id", "license", "inspection_date"]].merge(past, on="license")
    pairs["days_ago"] = (pairs["inspection_date"] - pairs["past_date"]).dt.days
    pairs = pairs[(pairs["days_ago"] > 0) & (pairs["days_ago"] <= LOOKBACK_DAYS)]

    with_result = pairs[pairs["has_result"]].sort_values(["inspection_id", "past_date", "past_id"])
    grouped = with_result.groupby("inspection_id")
    latest = grouped.tail(1).set_index("inspection_id")
    latest_canvass = (
        with_result[with_result["is_canvass"]].groupby("inspection_id").tail(1).set_index("inspection_id")
    )

    record = pd.DataFrame(index=rows["inspection_id"])
    record["n_prior"] = grouped.size()
    record["n_fail_prior"] = grouped["failed"].sum()
    record["mean_viol_prior"] = grouped["n_violations"].mean()
    record["pest_any"] = grouped["pest"].max()
    record["temp_any"] = grouped["temp"].max()
    record["mgmt_any"] = grouped["mgmt"].max()
    record["prev_n_viol"] = latest["n_violations"]
    record["days_since_last"] = latest["days_ago"]
    record["prev_canvass_fail"] = latest_canvass["failed"]
    record["n_complaints"] = pairs[pairs["is_complaint"]].groupby("inspection_id").size()

    # Counts are genuinely zero when nothing matched. The other columns stay
    # missing for restaurants with no history and are imputed later, using
    # values learned on the training period.
    for count in ["n_prior", "n_fail_prior", "n_complaints"]:
        record[count] = record[count].fillna(0)
    return rows.join(record, on="inspection_id")


def _window_counts(past_dates, past_fails, query_dates, window_days):
    """Inspections and fails in the `window_days` strictly before each query date.

    `past_dates` must be sorted. A running total of fails turns each window
    into two lookups instead of a loop over rows.
    """
    running_fails = np.concatenate([[0], np.cumsum(past_fails)])
    end = np.searchsorted(past_dates, query_dates, side="left")
    start = np.searchsorted(past_dates, query_dates - np.timedelta64(window_days, "D"), side="left")
    return end - start, running_fails[end] - running_fails[start]


def add_area_track_record(rows: pd.DataFrame, inspections: pd.DataFrame) -> pd.DataFrame:
    """Fail rate of routine inspections in the same ZIP over the previous year."""
    canvass = inspections[inspections["is_canvass"] & inspections["has_result"]].sort_values("inspection_date")
    as_days = lambda dates: dates.to_numpy().astype("datetime64[D]")

    rows = rows.copy()
    city_n, city_fails = _window_counts(
        as_days(canvass["inspection_date"]), canvass["failed"].to_numpy(),
        as_days(rows["inspection_date"]), AREA_WINDOW_DAYS,
    )
    city_rate = pd.Series(city_fails / np.maximum(city_n, 1), index=rows.index)

    # Start from the city rate (used as-is for a ZIP with no inspections),
    # then overwrite ZIP by ZIP.
    zip_rate = city_rate.copy()
    for zip_code, zip_canvass in canvass.groupby("zip"):
        in_zip = rows.index[rows["zip"] == zip_code]
        if len(in_zip) == 0:
            continue
        n, fails = _window_counts(
            as_days(zip_canvass["inspection_date"]), zip_canvass["failed"].to_numpy(),
            as_days(rows.loc[in_zip, "inspection_date"]), AREA_WINDOW_DAYS,
        )
        # Shrinkage: act as if every ZIP had AREA_PRIOR_WEIGHT extra
        # inspections that failed at the city rate. A ZIP with 3 inspections
        # stays near the city average; a ZIP with 300 barely moves.
        zip_rate.loc[in_zip] = (fails + AREA_PRIOR_WEIGHT * city_rate.loc[in_zip]) / (n + AREA_PRIOR_WEIGHT)

    rows["zip_fail_rate_12m"] = zip_rate
    return rows


# --- Static features -------------------------------------------------------------
LEGAL_SUFFIXES = {"LLC", "INC", "CORP", "CO", "LTD", "RESTAURANT", "THE"}


def chain_key(name: str) -> str:
    """Normalise a business name so branches of one chain share a key.

    "McDonald's #4521" and "MCDONALDS" both become "MCDONALDS". Store numbers,
    punctuation and legal suffixes are removed.
    """
    letters = re.sub(r"[^A-Z ]", "", str(name).upper().replace("&", " AND "))
    words = [word for word in letters.split() if word not in LEGAL_SUFFIXES]
    return " ".join(words)


def count_chain_sizes(inspections: pd.DataFrame, before: pd.Timestamp) -> dict[str, int]:
    """Distinct licences per normalised name, using only inspections before `before`."""
    known = inspections[inspections["inspection_date"] < before]
    sizes = known.assign(chain=known["dba_name"].map(chain_key)).groupby("chain")["license"].nunique()
    return sizes.to_dict()


def build_raw_table(inspections: pd.DataFrame, chain_sizes: dict[str, int]) -> pd.DataFrame:
    """One row per predicted inspection, with features still in human units."""
    rows = select_model_rows(inspections)
    rows = add_own_record(rows, inspections)
    rows = add_area_track_record(rows, inspections)
    rows["chain_size"] = rows["dba_name"].map(chain_key).map(chain_sizes).fillna(1).astype(int)
    rows["month"] = rows["inspection_date"].dt.month
    rows["split"] = np.select(
        [rows["inspection_date"] < VAL_START, rows["inspection_date"] < TEST_START],
        ["train", "validation"],
        default="test",
    )
    return rows


# --- From human units to the model matrix --------------------------------------
IMPUTED_WITH_MEDIAN = ["days_since_last", "mean_viol_prior", "prev_n_viol"]


def fit_constants(train_rows: pd.DataFrame) -> dict:
    """Everything the transformations need to learn, from training rows only."""
    return {
        "base_fail_rate": float(train_rows["failed"].mean()),
        "medians": {column: float(train_rows[column].median()) for column in IMPUTED_WITH_MEDIAN},
    }


def to_model_features(raw: pd.DataFrame, constants: dict) -> pd.DataFrame:
    """Impute and transform the raw columns into the 16 model features."""
    base_rate = constants["base_fail_rate"]
    out = pd.DataFrame(index=raw.index)

    out["risk_high"] = (raw["risk"] == "Risk 1 (High)").astype(float)
    # Counts and durations are right-skewed (a few restaurants have 15 past
    # inspections, most have 3). The log makes "1 to 2" matter as much as
    # "5 to 10", which is closer to how a linear model should treat them.
    out["log_chain_size"] = np.log(raw["chain_size"].astype(float))
    angle = 2 * np.pi * raw["month"].astype(float) / 12
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)

    out["zip_fail_rate_12m"] = raw["zip_fail_rate_12m"].astype(float)

    n_prior = raw["n_prior"].astype(float)
    # With no history this equals the base rate, i.e. "assume an average restaurant".
    out["fail_rate_3y"] = (raw["n_fail_prior"] + OWN_PRIOR_WEIGHT * base_rate) / (n_prior + OWN_PRIOR_WEIGHT)
    out["log_n_prior"] = np.log1p(n_prior)
    out["no_history"] = (n_prior == 0).astype(float)
    out["prev_canvass_fail"] = raw["prev_canvass_fail"].fillna(0).astype(float)
    out["log_days_since_last"] = np.log1p(
        raw["days_since_last"].astype(float).fillna(constants["medians"]["days_since_last"])
    )
    out["log_complaints"] = np.log1p(raw["n_complaints"].astype(float))

    out["mean_viol_prior"] = raw["mean_viol_prior"].astype(float).fillna(constants["medians"]["mean_viol_prior"])
    out["prev_n_viol"] = raw["prev_n_viol"].astype(float).fillna(constants["medians"]["prev_n_viol"])
    for flag in ["pest_any", "temp_any", "mgmt_any"]:
        out[flag] = raw[flag].fillna(0).astype(float)

    return out[FEATURE_NAMES]
