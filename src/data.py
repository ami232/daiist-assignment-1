"""Single source of truth for loading the data and making THE split.

Every model trains on exactly the split this module returns, so the comparison is fair.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "inventario_julio_2025_limpio.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns that must never become features: the label is built from Stock and Punto reorden,
# and the other four are functions of the same columns (see REPORT.md / data/README.md).
# Price columns are excluded separately as a design choice (markup x cost, nothing beyond Costo).
LEAKY_COLUMNS = [
    "Stock", "Estado stock", "Valor inventario", "Punto reorden", "Cant. reorden", "Alertas",
]

# Price columns: never features (they are markup x cost and add nothing beyond Costo), but
# they ARE read by fp_cost_per_sku below to price a false positive. tests/test_leakage.py
# checks that build_features ignores them.
PRICE_COLUMNS = [
    "Precio detalle", "Precio mayor 1", "Precio mayor 2", "Precio mayor 3", "Precio mínimo",
    "Margen %",
]

# Business cost of each kind of mistake, in DOP. ASSUMPTIONS, not measured figures.
# False positive = discontinuing a live item: we lose its margin (Precio detalle - Costo),
# assumed to be lost twice over (the sale that was in the pipeline and the next one),
# hence the multiplier. Per SKU, floored at 0 for the few items priced below cost.
# False negative = keeping a dead item listed: a flat catalogue/admin/storage drag per SKU.
# The FP cost is CAPPED: a wrongly discontinued high-value item gets noticed and re-listed
# within weeks, so the realistic loss is bounded, however large its margin on paper. Without
# the cap a handful of big-margin SKUs dominate the threshold choice.
FP_MARGIN_MULTIPLIER = 2.0
FP_COST_CAP = 10_000.0
# A false positive is a ONE-OFF loss; a false negative RECURS every year the dead listing
# survives. To compare them, the FN drag is expressed over the same planning horizon.
FN_COST_PER_YEAR = 500.0
HORIZON_YEARS = 3
FN_COST = FN_COST_PER_YEAR * HORIZON_YEARS


def load_raw() -> pd.DataFrame:
    """Load the cleaned snapshot and keep only physical products with a usable cost."""
    df = pd.read_csv(DATA_PATH)
    df = df[(df["Tipo"] == "Producto") & (df["Costo"] > 0)].copy()
    return df.reset_index(drop=True)


def make_target(df: pd.DataFrame) -> pd.Series:
    """Dead stock = nothing on hand AND no reorder point, i.e. the system does not intend
    to restock it. Returned as int (1 = dead, 0 = live)."""
    return ((df["Stock"] <= 0) & (df["Punto reorden"] == 0)).astype(int)


def fp_cost_per_sku(df: pd.DataFrame) -> pd.Series:
    """Cost of wrongly flagging each SKU as dead: FP_MARGIN_MULTIPLIER x its unit margin,
    floored at 0 and capped at FP_COST_CAP. Works on any frame that has Precio detalle and
    Costo (raw rows or test_predictions.csv); it is the ONE place this cost is defined."""
    margin = df["Precio detalle"] - df["Costo"]
    return (FP_MARGIN_MULTIPLIER * margin).clip(lower=0.0, upper=FP_COST_CAP)


def split(df: pd.DataFrame):
    """One split for the whole project. Returns (train_df, test_df).

    Stratified on the label so train and test have the same dead rate. NOTE: stratifying
    changes which rows land in the test set compared with the earlier regression run
    (same random_state, different split), so numbers are not comparable across the two."""
    return train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=make_target(df)
    )


if __name__ == "__main__":
    df = load_raw()
    tr, te = split(df)
    print(f"{len(df)} rows -> train {len(tr)} / test {len(te)}")
    y = make_target(df)
    print(f"dead: {int(y.sum())} of {len(y)} ({y.mean():.1%})")
