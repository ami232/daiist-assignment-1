"""Create an auditable cleaned CSV, preserving the original advertisements.

Run: uv run python clean_data.py
Rows missing engine displacement or drive type are excluded by student choice.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def clean_data():
    source = ROOT / 'data' / 'car_ad.csv'
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    raw = pd.read_csv(source, encoding='cp1251')
    data = raw.copy()
    # The original CSV line is an audit identifier, never a model input.
    data.insert(0, 'source_row', np.arange(2, len(data) + 2))
    reasons = pd.Series('', index=data.index, dtype='object')

    def flag(mask, reason):
        # Record every applicable rule; counts may overlap.
        for i in data.index[mask.fillna(False)]:
            reasons.loc[i] += ('; ' if reasons.loc[i] else '') + reason

    flag(raw.duplicated(), 'exact_duplicate')
    flag(data.engV.isna(), 'missing_engine_displacement')
    flag(data.drive.isna() | data.drive.astype('string').str.strip().eq(''), 'missing_drive_type')
    flag(data.price.isna() | ~np.isfinite(data.price) | data.price.le(0),
         'missing_nonfinite_or_nonpositive_price')
    flag(data.mileage.notna() & (~np.isfinite(data.mileage) | data.mileage.lt(0)),
         'negative_or_nonfinite_mileage')
    flag(data.year.notna() & (~np.isfinite(data.year) | data.year.gt(2016)
                              | data.year.le(0) | data.year.mod(1).ne(0)),
         'invalid_manufacturing_year')
    # A project scope heuristic, not proof that every excluded value is an error.
    flag(data.engV.notna() & (~np.isfinite(data.engV) | ~data.engV.between(0.5, 10)),
         'engine_outside_0.5_to_10_litres')

    kept = data.loc[reasons.eq('')].copy()
    removed = data.loc[reasons.ne('')].copy()
    removed['removal_reasons'] = reasons.loc[removed.index]
    destination = ROOT / 'data' / 'processed'
    destination.mkdir(parents=True, exist_ok=True)
    kept.to_csv(destination / 'car_ad_clean.csv', index=False)
    removed.to_csv(destination / 'removed_rows.csv', index=False)
    counts = reasons[reasons.ne('')].str.split('; ').explode().value_counts().to_dict()
    summary = {
        'raw_rows': len(raw), 'clean_rows': len(kept), 'removed_rows': len(removed),
        'counts_by_rule_overlaps_possible': {k:int(v) for k,v in counts.items()},
        'raw_sha256': original_hash,
        'output_encoding': 'utf-8',
        'missing_engine_or_drive_rows_removed': True,
        'engine_range_note': 'Scope heuristic; 0.5 through 10 litres, inclusive.',
    }
    (destination / 'cleaning_summary.json').write_text(json.dumps(summary, indent=2))
    assert len(kept) + len(removed) == len(raw)
    assert not kept.drop(columns='source_row').duplicated().any()
    assert kept.price.gt(0).all()
    assert kept.engV.dropna().between(0.5, 10).all()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    print(json.dumps(summary, indent=2))
    return kept


if __name__ == '__main__':
    clean_data()
