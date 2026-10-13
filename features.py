"""
Feature engineering for the credit-default model (version 1).

Every choice below was tested on the VALIDATION split only (never on test);
the experiment is reproducible with `uv run python experiments/feature_ablation.py`.
Rows are clients, columns are the 6-month history up to September 2005, and the
target is "defaults in October". All features are computed row by row from that
history, so none of them can see another client's data or the future.

Adding a feature never requires touching train.py or app.py: put it in
`engineer_features` and it flows through the scaler, the three models, the
saved artifacts and the dashboard's distribution view automatically.
"""

import numpy as np
import pandas as pd

RAW_TARGET = "default payment next month"
TARGET = "default"  # 1 = client defaults next month

# --- Decision 1: which raw columns never reach the model ---------------------
# * ID is a row index.
# * SEX and MARRIAGE are excluded: they are protected characteristics (or close
#   proxies) that do not describe repayment capacity, and using sex in credit
#   decisions is restricted in the EU. The performance price is tiny: on
#   validation, putting SEX back adds 0.0013 AUC and MARRIAGE 0.0008 AUC.
# * AGE is kept: it is a standard scorecard variable, but it is sensitive too,
#   so it is worth re-testing with it removed if the framing requires it.
EXCLUDED = ["ID", "SEX", "MARRIAGE"]

# --- Decision 2: which columns are categories rather than quantities ---------
# Only EDUCATION is a true category (one-hot). The PAY_* repayment-status codes
# (-2..8) stay numeric: default rate rises steadily with months of delay.
CATEGORICAL_COLS = ["EDUCATION"]

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]  # PAY_0 = most recent month


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model's raw (unscaled) features plus TARGET.

    Must be a pure function of the input rows (no fitting on the whole
    dataset), otherwise statistics from the test set leak into training.
    Anything that must be *learned* from data (means, scalers, encoders) goes
    in the sklearn preprocessor in train.py, which is fitted on train only.
    """
    out = df.rename(columns={RAW_TARGET: TARGET}).copy()

    # --- Decision 3: data quality -------------------------------------------
    # EDUCATION is documented as 1..4 but also contains 0, 5 and 6 (345 rows).
    # They are undocumented, tiny groups (EDUCATION=0: 14 clients, none
    # defaulted), which makes the logistic coefficient for them unstable.
    # Pooled into 4 ("others"), the documented catch-all category.
    out["EDUCATION"] = out["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    # Negative BILL_AMT (1,930 rows) are credit balances after overpaying. They
    # are real information, so the raw columns are left untouched.

    # --- Decision 4: engineered features ------------------------------------
    # Repayment-delay history carries almost all the signal: removing these
    # three features drops validation AUC from 0.767 to 0.742, while
    # utilisation and payment-ratio features added nothing measurable (so they
    # were tried and dropped).
    status = out[PAY_STATUS]
    # Worst delay in 6 months, in months late. Codes -2/-1/0 mean "no delay".
    out["max_delay"] = status.max(axis=1).clip(lower=0)
    # In how many of the 6 months was the client late at all.
    out["n_months_delayed"] = (status >= 1).sum(axis=1)
    # Default rate jumps from 17% to 70% when the last month is 2+ months late;
    # a linear PAY_0 term cannot represent that step.
    out["severe_delay_last"] = (out["PAY_0"] >= 2).astype(int)
    # Credit limit is right-skewed (median 140k, 99th percentile 500k).
    out["log_limit"] = np.log(out["LIMIT_BAL"])
    # Payments are extremely heavy-tailed (a few clients pay 1M+ against a
    # typical few thousand), so a handful of rows dominate a linear fit and the
    # three optimisers disagree most on exactly those rows. log1p replaces the
    # raw values (payments are never negative). Validation: AUC +0.005 and log
    # loss 0.4384 -> 0.4367; business cost per client unchanged. The same
    # transform on BILL_AMT* gave +0.001 AUC (noise), so bills stay raw.
    for col in [f"PAY_AMT{i}" for i in range(1, 7)]:
        out[col] = np.log1p(out[col])

    return out.drop(columns=[c for c in EXCLUDED if c in out.columns])


def split_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(numeric_cols, categorical_cols) of an engineered frame."""
    cats = [c for c in CATEGORICAL_COLS if c in frame.columns]
    nums = [c for c in frame.columns if c not in cats and c != TARGET]
    return nums, cats
