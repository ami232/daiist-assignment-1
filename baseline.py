"""Step 5: evaluate a training-median baseline on validation only.

Run: uv run python baseline.py
This deliberately simple reference ignores all vehicle characteristics.
The training median minimizes constant-prediction absolute error on training
prices, but is not necessarily the constant that minimizes percentage error.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

ROOT = Path(__file__).resolve().parent


def main():
    # Do not load the test set: it is reserved for final evaluation.
    training = pd.read_csv(ROOT / 'data/prepared/train.csv')
    validation = pd.read_csv(ROOT / 'data/prepared/validation.csv')
    if not (training.price.gt(0).all() and validation.price.gt(0).all()):
        raise ValueError('Positive prices are required to calculate MAPE.')

    # Learn a single reference price using training data only.
    reference_price = float(training['price'].median())
    predictions = np.full(len(validation), reference_price)
    actual = validation['price'].to_numpy()

    # sklearn returns MAPE as a fraction; multiply by 100 for a percentage.
    mape = float(mean_absolute_percentage_error(actual, predictions) * 100)
    mae = float(mean_absolute_error(actual, predictions))
    result = {
        'model': 'Training-median baseline',
        'constant_price_USD': reference_price,
        'training_rows': len(training),
        'evaluation_split': 'validation',
        'evaluation_rows': len(validation),
        'MAPE_percent': mape,
        'MAE_USD': mae,
        'test_evaluated': False,
        'note': 'Median baseline is a simple MAE-oriented reference, not a MAPE-optimal constant.',
    }
    out = ROOT / 'results/baseline'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'baseline.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    details = validation[['source_row', 'price']].copy()
    details['predicted_price'] = predictions
    details['absolute_error_USD'] = np.abs(actual - predictions)
    details['absolute_percentage_error'] = details['absolute_error_USD'] / actual * 100
    details.to_csv(out / 'validation_predictions.csv', index=False)

    # Check metric calculation against the row-by-row errors.
    assert np.isfinite(predictions).all()
    assert np.isclose(details.absolute_error_USD.mean(), mae)
    assert np.isclose(details.absolute_percentage_error.mean(), mape)
    print(json.dumps(result, indent=2))
    print('\nFirst five validation examples:')
    print(details.head().round(2).to_string(index=False))


if __name__ == '__main__':
    main()
