"""Fit the same log-price Ridge with manual autograd and nn.Module + SGD.

Run after train_log_sklearn.py. Only training and validation are accessed.
"""
from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from train_log_sklearn import metrics

ROOT = Path(__file__).resolve().parent
DTYPE = torch.float64
MAX_STEPS = 20000
GRADIENT_TOLERANCE = 1e-8


def objective(prediction, target, weights, strength):
    # Match sklearn alpha = n * lambda. Do not penalize the intercept.
    return ((prediction-target)**2).mean() + strength * weights.square().sum()


def fit(x, y, strength, learning_rate, standard):
    # Identical deterministic initialization; no sklearn coefficients are used.
    if standard:
        model = nn.Linear(x.shape[1], 1, dtype=DTYPE)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(y.mean())
        parameters = list(model.parameters())
        optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    else:
        weights = torch.zeros((x.shape[1], 1), dtype=DTYPE, requires_grad=True)
        bias = y.mean().detach().clone().requires_grad_(True)
        parameters = [weights, bias]
    history = []
    start = time.perf_counter()
    for step in range(MAX_STEPS+1):
        if standard:
            optimizer.zero_grad()
            prediction, penalty_weights = model(x), model.weight
        else:
            for parameter in parameters:
                parameter.grad = None
            prediction, penalty_weights = x @ weights + bias, weights
        loss = objective(prediction, y, penalty_weights, strength)
        loss.backward()
        gradient = max(p.grad.abs().max().item() for p in parameters)
        if step % 250 == 0 or gradient <= GRADIENT_TOLERANCE:
            history.append({'step': step, 'training_objective': loss.item(),
                            'max_absolute_gradient': gradient})
        if gradient <= GRADIENT_TOLERANCE:
            break
        if step == MAX_STEPS:
            raise RuntimeError('Optimizer did not reach the training convergence tolerance.')
        if standard:
            optimizer.step()
        else:
            with torch.no_grad():
                for parameter in parameters:
                    parameter -= learning_rate * parameter.grad
    if standard:
        weights_out = model.weight.detach().reshape(-1).clone()
        bias_out = model.bias.detach().reshape(()).clone()
        state = model.state_dict()
    else:
        weights_out = weights.detach().reshape(-1).clone()
        bias_out = bias.detach().clone()
        state = None
    return weights_out, bias_out, state, history, time.perf_counter()-start


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    path = ROOT/'data/prepared/arrays.npz'
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as arrays:
        x = torch.tensor(arrays['X_train'], dtype=DTYPE)
        y_usd = arrays['y_train']
        x_val = arrays['X_validation']
        y_val = arrays['y_validation']
        row_ids = arrays['source_rows_validation']
    y = torch.tensor(np.log(y_usd), dtype=DTYPE).reshape(-1, 1)
    selected = json.loads((ROOT/'results/sklearn_log/model_summary.json').read_text())
    assert checksum == selected['prepared_arrays_sha256']
    strength = selected['lambda']
    # For this quadratic loss, a training-only Hessian provides a stable step
    # size. 0.9/L is below 1/L; no validation learning-rate search is needed.
    augmented = torch.cat([x, torch.ones((len(x), 1), dtype=DTYPE)], dim=1)
    penalty = torch.diag(torch.tensor([strength]*x.shape[1]+[0.0], dtype=DTYPE))
    hessian = 2 * (augmented.T @ augmented / len(x) + penalty)
    largest_eigenvalue = torch.linalg.eigvalsh(hessian)[-1].item()
    learning_rate = 0.9/largest_eigenvalue
    reference_model = joblib.load(ROOT/'results/sklearn_log/model.joblib')
    reference = reference_model.predict(x_val)
    comparison = [
        {'model': 'Training-median baseline', **metrics(y_val, np.full(len(y_val), np.median(y_usd)))},
        {'model': 'scikit-learn log-price Ridge', **metrics(y_val, reference)},
    ]
    predictions = pd.DataFrame({'source_row': row_ids, 'actual_price_USD': y_val,
                                'sklearn_price_USD': reference})
    out = ROOT/'results/pytorch'
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, standard in [('manual', False), ('standard', True)]:
        weights, bias, state, history, elapsed = fit(x, y, strength, learning_rate, standard)
        artifact = {'weights': weights, 'bias': bias, 'state_dict': state,
                    'target_transform': 'natural_log', 'inverse_transform': 'exp',
                    'input_columns': x.shape[1], 'lambda': strength,
                    'prepared_arrays_sha256': checksum}
        torch.save(artifact, out/f'{name}.pt')
        restored = torch.load(out/f'{name}.pt', weights_only=True)
        pred = np.exp(x_val @ weights.numpy()+bias.item())
        loaded_prediction = np.exp(x_val @ restored['weights'].numpy()+restored['bias'].item())
        np.testing.assert_allclose(pred, loaded_prediction, rtol=1e-12)
        if standard:
            restored_module = nn.Linear(x.shape[1], 1, dtype=DTYPE)
            restored_module.load_state_dict(restored['state_dict'])
            with torch.no_grad():
                module_pred = restored_module(torch.tensor(x_val, dtype=DTYPE)).exp().numpy().ravel()
            np.testing.assert_allclose(pred, module_pred, rtol=1e-12)
        assert np.isfinite(pred).all() and (pred > 0).all()
        np.testing.assert_allclose(pred, reference, rtol=1e-5, atol=0.01)
        result = {'model': f'PyTorch {name}', **metrics(y_val, pred)}
        comparison.append(result)
        predictions[f'{name}_price_USD'] = pred
        summaries[name] = {
            **result, 'updates': history[-1]['step'], 'seconds': elapsed,
            'final_training_objective': history[-1]['training_objective'],
            'max_absolute_gradient': history[-1]['max_absolute_gradient'],
            'max_prediction_difference_from_sklearn_USD': float(np.max(np.abs(pred-reference))),
            'max_coefficient_difference_from_sklearn': float(np.max(np.abs(weights.numpy()-reference_model.regressor_.coef_))),
        }
        pd.DataFrame(history).to_csv(out/f'{name}_training_history.csv', index=False)
        print(json.dumps(summaries[name], indent=2), flush=True)
    assert checksum == hashlib.sha256(path.read_bytes()).hexdigest()
    table = pd.DataFrame(comparison)
    table.to_csv(out/'validation_comparison.csv', index=False)
    predictions.to_csv(out/'validation_predictions.csv', index=False)
    summary = {'lambda': strength, 'learning_rate': learning_rate,
               'learning_rate_rule': '0.9 / largest training Hessian eigenvalue',
               'dtype': 'float64', 'batch': 'Full training set',
               'training_rows': len(x), 'validation_rows': len(y_val),
               'gradient_tolerance': GRADIENT_TOLERANCE, 'maximum_updates': MAX_STEPS,
               'prepared_arrays_sha256': checksum, 'test_evaluated': False,
               'implementations': summaries,
               'checks': 'Converged training gradients; positive finite predictions; saved-model reload; sklearn prediction agreement (rtol=1e-5, atol=0.01 USD); unchanged prepared arrays'}
    (out/'comparison_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(table.round(4).to_string(index=False))
    print('All checks passed. Test partition not accessed or evaluated.')


if __name__ == '__main__':
    main()
