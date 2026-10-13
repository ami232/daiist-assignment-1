"""Steps 3–4: split and prepare data, without training a prediction model.

Run: uv run python prepare_data.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from clean_data import clean_data

ROOT = Path(__file__).resolve().parent
RAW_FEATURES = ['car', 'body', 'mileage', 'engV', 'engType', 'registration', 'year', 'model', 'drive']
NUMERIC = ['age', 'mileage', 'engV', 'mileage_per_year']
CATEGORICAL = ['car', 'brand_model', 'body', 'engType', 'registration', 'drive']


def make_features(data):
    """Deterministic row transformations; no price or population statistics."""
    x = data.copy()
    x['age'] = 2016 - x['year']
    # Dataset mileage is thousands of km. This is only an approximate usage rate.
    x['mileage_per_year'] = x['mileage'] / x['age'].clip(lower=1)
    x['brand_model'] = x['car'].fillna('Unknown') + ' / ' + x['model'].fillna('Unknown')
    # Source documentation treats Other fuel as an unknown value.
    x['engType'] = x['engType'].replace('Other', np.nan)
    for col in CATEGORICAL:
        x[col] = x[col].astype(object).where(x[col].notna(), np.nan)
    return x[NUMERIC + CATEGORICAL]


def main():
    data = clean_data().reset_index(drop=True)
    out = ROOT / 'data' / 'prepared'
    out.mkdir(parents=True, exist_ok=True)
    # Same observed characteristics stay together even if asking prices differ.
    groups = pd.util.hash_pandas_object(data[RAW_FEATURES].fillna('Unknown'), index=False)
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=42)
                     .split(data, groups=groups))
    train_local, validation_local = next(GroupShuffleSplit(
        n_splits=1, test_size=.15/.85, random_state=43)
        .split(data.iloc[dev], groups=groups.iloc[dev]))
    splits = {'train': dev[train_local], 'validation': dev[validation_local], 'test': test}

    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), NUMERIC),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                     min_frequency=20, sparse_output=False)),
        ]), CATEGORICAL),
    ], sparse_threshold=0)
    # Only this training partition determines medians, scales and categories.
    preprocessor.fit(make_features(data.iloc[splits['train']]))
    arrays = {}
    assignment = data[['source_row']].copy()
    assignment['split'] = ''
    for name, idx in splits.items():
        part = data.iloc[idx]
        x = preprocessor.transform(make_features(part))
        y = part.price.to_numpy(dtype=float)
        assert np.isfinite(x).all() and np.isfinite(y).all() and (y > 0).all()
        arrays[f'X_{name}'] = x
        arrays[f'y_{name}'] = y
        arrays[f'source_rows_{name}'] = part.source_row.to_numpy()
        assignment.loc[idx, 'split'] = name
        part.to_csv(out / f'{name}.csv', index=False)
    for a, b in [('train','validation'), ('train','test'), ('validation','test')]:
        assert set(groups.iloc[splits[a]]).isdisjoint(set(groups.iloc[splits[b]]))
    assert sum(len(idx) for idx in splits.values()) == len(data)
    assert assignment.source_row.is_unique and assignment.split.ne('').all()
    assert np.allclose(arrays['X_train'][:, :len(NUMERIC)].mean(axis=0), 0, atol=1e-10)
    np.savez_compressed(out / 'arrays.npz', **arrays)
    joblib.dump(preprocessor, out / 'preprocessor.joblib')
    reloaded = joblib.load(out / 'preprocessor.joblib')
    np.testing.assert_allclose(reloaded.transform(make_features(data.iloc[splits['train'][:5]])),
                               arrays['X_train'][:5])
    assignment.to_csv(out / 'split_assignment.csv', index=False)
    summary = {
        'primary_metric': 'MAPE (%) = 100 * mean(abs(actual - predicted) / actual)',
        'secondary_metric': 'MAE (USD)',
        'metric_note': 'All targets are positive, but unusually low prices can dominate MAPE.',
        'target': 'price (USD), untransformed',
        'split_strategy': 'GroupShuffleSplit on observed raw predictor attributes',
        'seed_test': 42, 'seed_validation': 43,
        'rows': {name:len(idx) for name,idx in splits.items()},
        'percent_rows': {name:round(100*len(idx)/len(data),2) for name,idx in splits.items()},
        'numeric_features': NUMERIC, 'categorical_features': CATEGORICAL,
        'encoded_feature_count': len(preprocessor.get_feature_names_out()),
        'encoded_feature_names': preprocessor.get_feature_names_out().tolist(),
        'rare_category_min_training_count': 20,
        'training_only_preprocessing': True,
        'prediction_model_trained': False,
    }
    (out / 'preparation_summary.json').write_text(json.dumps(summary, indent=2))
    print('\nPREPARATION COMPLETE')
    print('Rows per split:', summary['rows'])
    print('Percent per split:', summary['percent_rows'])
    print('Encoded features:', summary['encoded_feature_count'])
    print('All integrity checks passed. No prediction model trained.')


if __name__ == '__main__':
    main()
