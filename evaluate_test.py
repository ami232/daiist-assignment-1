"""Evaluate frozen model choices on the held-out test set; never fit models."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from train_log_sklearn import metrics

ROOT = Path(__file__).resolve().parent


def main():
    prepared = ROOT/'data/prepared/arrays.npz'
    checksum = hashlib.sha256(prepared.read_bytes()).hexdigest()
    selected = json.loads((ROOT/'results/sklearn_log/model_summary.json').read_text())
    assert selected['prepared_arrays_sha256'] == checksum
    assert selected['lambda'] == 0.01, 'The pre-test model choice must remain fixed.'
    with np.load(prepared) as data:
        x, y = data['X_test'], data['y_test']
        row_ids = data['source_rows_test']
    baseline = json.loads((ROOT/'results/baseline/baseline.json').read_text())
    # Read the baseline's saved value, never estimate a constant from test data.
    median = float(baseline['constant_price_USD'])
    predictions = {
        'Training-median baseline': np.full(len(y), median),
        'scikit-learn log-price Ridge': joblib.load(ROOT/'results/sklearn_log/model.joblib').predict(x),
    }
    for name in ['manual', 'standard']:
        saved = torch.load(ROOT/f'results/pytorch/{name}.pt', weights_only=True)
        assert saved['prepared_arrays_sha256'] == checksum
        assert saved['lambda'] == selected['lambda']
        if name == 'manual':
            pred = np.exp(x @ saved['weights'].numpy()+saved['bias'].item())
        else:
            model = nn.Linear(x.shape[1], 1, dtype=torch.float64)
            model.load_state_dict(saved['state_dict'])
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(x, dtype=torch.float64)).exp().numpy().ravel()
        predictions[f'PyTorch {name}'] = pred
    out = ROOT/'results/test'
    out.mkdir(parents=True, exist_ok=True)
    rows, details = [], pd.DataFrame({'source_row': row_ids, 'actual_price_USD': y})
    reference = predictions['scikit-learn log-price Ridge']
    for name, pred in predictions.items():
        assert np.isfinite(pred).all() and (pred > 0).all()
        rows.append({'model': name, 'split': 'test', 'rows': len(y), **metrics(y, pred)})
        details[name] = pred
        if name.startswith('PyTorch'):
            np.testing.assert_allclose(pred, reference, rtol=1e-5, atol=0.01)
    table = pd.DataFrame(rows)
    table.to_csv(out/'comparison.csv', index=False)
    details.to_csv(out/'predictions.csv', index=False)
    summary = {
        'test_rows': len(y), 'training_rows': selected['training_rows'],
        'lambda': selected['lambda'], 'target_transform': 'natural_log',
        'selection': 'Target transform and lambda selected on validation before test evaluation',
        'refit_on_validation': False, 'test_used_for_tuning': False,
        'prepared_arrays_sha256': checksum,
        'max_pytorch_difference_from_sklearn_USD': {
            name: float(np.max(np.abs(pred-reference)))
            for name, pred in predictions.items() if name.startswith('PyTorch')},
        'metrics': rows,
    }
    (out/'evaluation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('\nFINAL HELD-OUT TEST EVALUATION')
    print(table.round(4).to_string(index=False))
    print('No fitting or hyperparameter selection used test data.')


if __name__ == '__main__':
    main()
