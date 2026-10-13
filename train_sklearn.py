"""Step 6: train regularized linear regression and select it on validation MAPE.

Run after prepare_data.py: uv run python train_sklearn.py
This stage never reads the test arrays or evaluates the test partition.
"""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

ROOT = Path(__file__).resolve().parent


def main():
    prepared = ROOT / 'data/prepared'
    # np.load reads individual compressed arrays only when accessed by name.
    with np.load(prepared / 'arrays.npz') as arrays:
        x_train = arrays['X_train']
        y_train = arrays['y_train']
        x_val = arrays['X_validation']
        y_val = arrays['y_validation']
        row_ids = arrays['source_rows_validation']
    metadata = json.loads((prepared / 'preparation_summary.json').read_text())
    assert x_train.shape[1] == len(metadata['encoded_feature_names'])
    assert np.isfinite(x_train).all() and np.isfinite(x_val).all()
    assert (y_train > 0).all() and (y_val > 0).all()

    # One model type, three regularization settings. No feature changes here.
    # Ridge minimizes SUM squared error + alpha * sum(weight^2).
    # alpha = n * lambda matches MEAN squared error + lambda * sum(weight^2),
    # the objective we will implement later with both PyTorch methods.
    candidates = []
    models = []
    for strength in [0.01, 0.1, 1.0]:
        alpha = len(y_train) * strength
        model = Ridge(alpha=alpha, fit_intercept=True, solver='svd')
        model.fit(x_train, y_train)
        prediction = model.predict(x_val)
        candidates.append({
            'lambda': strength,
            'sklearn_alpha': alpha,
            'validation_MAPE_percent': float(mean_absolute_percentage_error(y_val, prediction) * 100),
            'validation_MAE_USD': float(mean_absolute_error(y_val, prediction)),
            'negative_validation_predictions': int((prediction < 0).sum()),
        })
        models.append(model)
    selected = min(range(len(candidates)), key=lambda i: candidates[i]['validation_MAPE_percent'])
    model = models[selected]
    prediction = model.predict(x_val)
    reference = float(np.median(y_train))
    comparison = pd.DataFrame([
        {'model': 'Training-median baseline', 'split': 'validation',
         'MAPE_percent': float(mean_absolute_percentage_error(y_val, np.full(len(y_val), reference))*100),
         'MAE_USD': float(mean_absolute_error(y_val, np.full(len(y_val), reference)))},
        {'model': 'scikit-learn Ridge', 'split': 'validation',
         'MAPE_percent': candidates[selected]['validation_MAPE_percent'],
         'MAE_USD': candidates[selected]['validation_MAE_USD']},
    ])

    out = ROOT / 'results/sklearn'
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / 'model.joblib')
    pd.DataFrame(candidates).to_csv(out / 'regularization_comparison.csv', index=False)
    comparison.to_csv(out / 'validation_comparison.csv', index=False)
    pd.DataFrame({'source_row': row_ids, 'actual_price_USD': y_val,
                  'predicted_price_USD': prediction,
                  'absolute_error_USD': np.abs(prediction-y_val),
                  'absolute_percentage_error': np.abs(prediction-y_val)/y_val*100,
                  }).to_csv(out / 'validation_predictions.csv', index=False)
    pd.DataFrame({'feature': metadata['encoded_feature_names'], 'coefficient': model.coef_})\
        .to_csv(out / 'coefficients.csv', index=False)
    summary = {
        **candidates[selected],
        'model': 'Ridge linear regression',
        'target': 'Untransformed advertised price in USD',
        'intercept': float(model.intercept_),
        'training_rows': len(y_train), 'validation_rows': len(y_val),
        'input_columns': x_train.shape[1],
        'training_objective': 'Mean squared error in USD + lambda * sum(weight^2); unpenalized intercept',
        'selection_metric': 'Lowest validation MAPE among three fixed candidates',
        'prediction_postprocessing': 'None; negative predictions are retained and reported',
        'test_evaluated': False,
        'prepared_arrays_sha256': hashlib.sha256((prepared/'arrays.npz').read_bytes()).hexdigest(),
        'preprocessor_sha256': hashlib.sha256((prepared/'preprocessor.joblib').read_bytes()).hexdigest(),
    }
    (out / 'model_summary.json').write_text(json.dumps(summary, indent=2))
    restored = joblib.load(out / 'model.joblib')
    np.testing.assert_allclose(restored.predict(x_val), prediction, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(x_val @ model.coef_ + model.intercept_, prediction)
    assert np.isfinite(prediction).all()
    print('REGULARIZATION CANDIDATES')
    print(pd.DataFrame(candidates).round(3).to_string(index=False))
    print('\nVALIDATION COMPARISON')
    print(comparison.round(3).to_string(index=False))
    print(f'\nSelected lambda: {candidates[selected]["lambda"]}')
    print(f'Negative predictions: {(prediction < 0).sum()} of {len(prediction)}')
    print('Saved model reload and linear prediction checks passed. Test set not evaluated.')


if __name__ == '__main__':
    main()
