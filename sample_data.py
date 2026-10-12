"""Filter to delivered orders and draw a stratified sample of ~N orders.

Strata are purchase year-month x late flag, so the sample keeps both the
seasonal volume profile and the real late rate within each month. Rows from
the other Olist tables are kept only for the sampled orders.
"""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/sampled")
TARGET_ROWS = 20_000
SEED = 42


def load_delivered_orders() -> pd.DataFrame:
    orders = pd.read_csv(
        RAW / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    # Late is only observable for delivered orders with a delivery date.
    orders = orders[orders.order_purchase_timestamp.dt.year != 2016]
    delivered = orders[orders.order_status == "delivered"].dropna(
        subset=["order_delivered_customer_date"]
    )
    delivered = delivered.assign(
        late=delivered.order_delivered_customer_date.dt.normalize()
        > delivered.order_estimated_delivery_date
    )
    return delivered


def stratified_sample(orders: pd.DataFrame) -> pd.DataFrame:
    strata = [orders.order_purchase_timestamp.dt.to_period("M"), orders.late]
    frac = TARGET_ROWS / len(orders)
    return orders.groupby(strata, group_keys=False).sample(frac=frac, random_state=SEED)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    delivered = load_delivered_orders()
    sample = stratified_sample(delivered)
    print(f"delivered orders: {len(delivered)}  late rate: {delivered.late.mean():.4f}")
    print(f"sampled orders:   {len(sample)}  late rate: {sample.late.mean():.4f}")

    order_ids = set(sample.order_id)
    sample.to_csv(OUT / "orders.csv", index=False)

    items = pd.read_csv(RAW / "olist_order_items_dataset.csv")
    items = items[items.order_id.isin(order_ids)]
    items.to_csv(OUT / "order_items.csv", index=False)

    for name in ["order_payments", "order_reviews"]:
        df = pd.read_csv(RAW / f"olist_{name}_dataset.csv")
        df[df.order_id.isin(order_ids)].to_csv(OUT / f"{name}.csv", index=False)

    customers = pd.read_csv(RAW / "olist_customers_dataset.csv")
    customers = customers[customers.customer_id.isin(sample.customer_id)]
    customers.to_csv(OUT / "customers.csv", index=False)
    products = pd.read_csv(RAW / "olist_products_dataset.csv")
    products[products.product_id.isin(items.product_id)].to_csv(
        OUT / "products.csv", index=False
    )
    sellers = pd.read_csv(RAW / "olist_sellers_dataset.csv")
    sellers = sellers[sellers.seller_id.isin(items.seller_id)]
    sellers.to_csv(OUT / "sellers.csv", index=False)

    # Raw geolocation has ~1M rows (many per zip prefix). Collapse to one
    # centroid per zip prefix, keeping only prefixes the sample refers to.
    geo = pd.read_csv(RAW / "olist_geolocation_dataset.csv")
    needed = set(customers.customer_zip_code_prefix) | set(sellers.seller_zip_code_prefix)
    geo = geo[geo.geolocation_zip_code_prefix.isin(needed)]
    geo = geo.groupby("geolocation_zip_code_prefix", as_index=False)[
        ["geolocation_lat", "geolocation_lng"]
    ].mean()
    geo.to_csv(OUT / "geolocation.csv", index=False)
    print(f"wrote tables to {OUT}/")


if __name__ == "__main__":
    main()
