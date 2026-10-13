"""Compare log-target Ridge with the saved raw-target model on validation only."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

ROOT = Path(__file__).resolve().parent


def metrics(actual, predicted):
    return {
        'MAPE_percent': float(mean_absolute_percentage_error(actual, predicted) * 100),
        'MAE_USD': float(mean_absolute_error(actual, predicted)),
        'median_absolute_percentage_error': float(np.median(np.abs(predicted-actual)/actual)*100),
        'negative_predictions': int((predicted < 0).sum()),
    }


def main():
    prepared = ROOT / 'data/prepared'
    arrays_path = prepared / 'arrays.npz'
    before = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    with np.load(arrays_path) as data:
        x_train, y_train = data['X_train'], data['y_train']
        x_val, y_val = data['X_validation'], data['y_validation']
        row_ids = data['source_rows_validation']
    assert (y_train > 0).all() and (y_val > 0).all()
    candidates, models = [], []
    for strength in [0.01, 0.1, 1.0]:
        model = TransformedTargetRegressor(
            regressor=Ridge(alpha=len(y_train)*strength, solver='svd'),
            func=np.log, inverse_func=np.exp,
        )
        model.fit(x_train, y_train)
        prediction = model.predict(x_val)
        assert np.isfinite(prediction).all() and (prediction > 0).all()
        candidates.append({'lambda': strength, 'sklearn_alpha': len(y_train)*strength,
                           **metrics(y_val, prediction)})
        models.append(model)
    best = min(range(len(candidates)), key=lambda i: candidates[i]['MAPE_percent'])
    selected = models[best]
    predictions = {
        'Training-median baseline': np.full(len(y_val), np.median(y_train)),
        'Raw-price Ridge': joblib.load(ROOT / 'results/sklearn/model.joblib').predict(x_val),
        'Log-price Ridge': selected.predict(x_val),
    }
    comparison = pd.DataFrame([{'model': name, **metrics(y_val, pred)}
                              for name, pred in predictions.items()])
    out = ROOT / 'results/sklearn_log'
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, out / 'model.joblib')
    np.testing.assert_allclose(joblib.load(out/'model.joblib').predict(x_val), predictions['Log-price Ridge'])
    np.testing.assert_allclose(np.exp(x_val @ selected.regressor_.coef_ + selected.regressor_.intercept_),
                               predictions['Log-price Ridge'])
    assert before == hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    pd.DataFrame(candidates).to_csv(out/'regularization_comparison.csv', index=False)
    comparison.to_csv(out/'validation_comparison.csv', index=False)
    details = pd.DataFrame({'source_row': row_ids, 'actual_price_USD': y_val})
    for name, pred in predictions.items():
        details[name] = pred
    details['log_absolute_percentage_error'] = np.abs(predictions['Log-price Ridge']-y_val)/y_val*100
    details.to_csv(out/'validation_predictions.csv', index=False)
    # Fixed, descriptive price bands; no filtering or model selection by band.
    bands = pd.cut(y_val, [0, 3000, 10000, 25000, np.inf], right=False,
                   labels=['Under 3,000', '3,000–9,999', '10,000–24,999', '25,000 or more'])
    groups = []
    for band in bands.categories:
        mask = bands == band
        if not mask.any():
            continue
        for name, pred in predictions.items():
            errors = np.abs(pred-y_val)/y_val
            groups.append({'actual_price_band_USD': band, 'model': name,
                           'rows': int(mask.sum()), **metrics(y_val[mask], pred[mask]),
                           'share_of_total_absolute_percentage_error_percent':
                               float(errors[mask].sum()/errors.sum()*100)})
    band_table = pd.DataFrame(groups)
    band_table.to_csv(out/'errors_by_price_band.csv', index=False)
    details.nlargest(20, 'log_absolute_percentage_error').to_csv(out/'largest_validation_errors.csv', index=False)
    summary = {
        **candidates[best], 'model': 'Ridge on natural log of advertised price in USD',
        'prediction': 'exp(intercept + X @ weights); no clipping or bias correction',
        'training_objective': 'Mean squared log-price error + lambda * sum(weight^2); unpenalized intercept',
        'selection': 'Minimum validation MAPE across three fixed lambda candidates',
        'training_rows': len(y_train), 'validation_rows': len(y_val), 'input_columns': x_train.shape[1],
        'prepared_arrays_sha256': before, 'test_evaluated': False,
        'checks': 'Positive finite predictions; reload and explicit formula agree; prepared arrays unchanged',
    }
    (out/'model_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(pd.DataFrame(candidates).round(3).to_string(index=False))
    print(comparison.round(3).to_string(index=False))
    print(band_table.round(3).to_string(index=False))
    print('Checks passed. Test partition not accessed or evaluated.')


if __name__ == '__main__':
    main()
