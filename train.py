"""Run with `uv run python main.py train`. Rebuild all artifacts from local data."""
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from features import EXCLUDED, engineer, make_preprocessor
from modeling import LogisticModel
from evaluation import evaluate, select_threshold

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
SEED = 42
LAMBDAS = [0.001, 0.01, 0.1]
GRAD_TOL = 1e-7
MAX_STEPS = 20000


def objective(x, y, w, b, lam):
    logits = x @ w + b
    return (torch.nn.functional.softplus(logits) - y * logits).mean() + lam / 2 * w.square().sum()


def train_torch(x_np, y_np, lam, lr, standard=False):
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64)
    history = []
    if standard:
        model = LogisticModel(x.shape[1])
        with torch.no_grad():
            model.linear.weight.zero_()
            model.linear.bias.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    started = time.perf_counter()
    for step in range(MAX_STEPS + 1):
        if standard:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y) + lam / 2 * model.linear.weight.square().sum()
            loss.backward()
            grad = max(model.linear.weight.grad.abs().max().item(), model.linear.bias.grad.abs().max().item())
        else:
            loss = objective(x, y, w, b, lam)
            loss.backward()
            grad = max(w.grad.abs().max().item(), b.grad.abs().item())
        if step % 100 == 0 or grad < GRAD_TOL or step == MAX_STEPS:
            history.append({'step': step, 'objective': loss.item(), 'gradient_inf_norm': grad})
        if grad < GRAD_TOL or step == MAX_STEPS:
            break
        if standard:
            optimizer.step()
        else:
            # Manual gradient descent: autograd computes gradients; we update
            # leaf tensors without recording the update in the computation graph.
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()
    if grad >= GRAD_TOL:
        raise RuntimeError(f'Optimization did not converge: gradient={grad:g}')
    if standard:
        weights, bias = model.linear.weight.detach().flatten(), model.linear.bias.detach().squeeze()
        torch.save(model.state_dict(), OUT / 'standard.pt')
    else:
        weights, bias = w.detach(), b.detach()
        torch.save({'weights': weights, 'bias': bias}, OUT / 'manual.pt')
    return weights.numpy(), bias.item(), history, time.perf_counter() - started


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    source = ROOT / 'data/bank-additional-full.csv'
    raw = pd.read_csv(source, sep=';')
    assert len(raw) == 41188 and set(raw.y.unique()) == {'yes', 'no'}
    y = raw.y.eq('yes').astype(int).to_numpy()
    n_train, n_val_end = int(0.6 * len(raw)), int(0.8 * len(raw))
    slices = {'train': slice(0, n_train), 'validation': slice(n_train, n_val_end), 'test': slice(n_val_end, len(raw))}
    split_names = np.repeat('test', len(raw)).astype(object)
    for name, s in slices.items():
        split_names[s] = name
    # No deduplication: identical covariates may represent different customers.
    # The dataset has no customer ID, so this limitation cannot be resolved.
    split_table = pd.DataFrame({'row_id': np.arange(len(raw)), 'split': split_names, 'actual': y})
    split_table.to_csv(OUT / 'split_membership.csv', index=False)
    candidates = []
    fitted = {}
    variants = ['raw', 'history', 'history_context']
    for variant in variants:
        features = engineer(raw, variant)
        prep = make_preprocessor(features.iloc[slices['train']])
        x_train = prep.fit_transform(features.iloc[slices['train']]).astype(np.float64)
        x_val = prep.transform(features.iloc[slices['validation']]).astype(np.float64)
        for lam in LAMBDAS:
            start = time.perf_counter()
            # sklearn mean loss + ||w||²/(2*C*n), hence C=1/(lambda*n).
            model = LogisticRegression(C=1 / (lam * n_train), solver='lbfgs', max_iter=5000, tol=1e-11)
            model.fit(x_train, y[slices['train']])
            duration = time.perf_counter() - start
            val_p = model.predict_proba(x_val)[:, 1]
            candidates.append({'variant': variant, 'lambda': lam, 'features': x_train.shape[1],
                               'validation_log_loss': float(log_loss(y[slices['validation']], val_p)),
                               'sklearn_fit_seconds': duration})
            fitted[(variant, lam)] = (prep, model, x_train, x_val, duration)
    candidate_table = pd.DataFrame(candidates).sort_values(['validation_log_loss', 'features'])
    candidate_table.to_csv(OUT / 'feature_experiments.csv', index=False)
    # Student chose correct sentinel semantics over the slightly better raw control.
    best = candidate_table[candidate_table.variant != 'raw'].iloc[0]
    variant, lam = str(best['variant']), float(best['lambda'])
    prep, sklearn_model, x_train, x_val, sklearn_time = fitted[(variant, lam)]
    feature_names = prep.get_feature_names_out().tolist()
    # Conservative full-batch step from the TRAIN matrix's curvature bound.
    augmented = np.column_stack([x_train, np.ones(n_train)])
    lipschitz = 0.25 * np.linalg.eigvalsh(augmented.T @ augmented / n_train)[-1] + lam
    lr = float(1 / lipschitz)
    print(f'Selected {variant}, lambda={lam}, {len(feature_names)} features; shared learning rate={lr:.6f}', flush=True)
    joblib.dump(prep, OUT / 'preprocessor.joblib')
    joblib.dump(sklearn_model, OUT / 'sklearn.joblib')
    baseline = DummyClassifier(strategy='prior').fit(x_train, y[slices['train']])
    joblib.dump(baseline, OUT / 'baseline.joblib')
    trained = {'sklearn': (sklearn_model.coef_[0], sklearn_model.intercept_[0])}
    histories, durations = {}, {'sklearn': sklearn_time}
    for name, standard in [('manual', False), ('standard', True)]:
        w, b, history, elapsed = train_torch(x_train, y[slices['train']], lam, lr, standard)
        trained[name] = (w, b)
        histories[name], durations[name] = history, elapsed
        print(f'{name}: {history[-1]["step"]} steps, {elapsed:.2f}s, gradient {history[-1]["gradient_inf_norm"]:.3g}', flush=True)
    # Freeze every decision before transforming/evaluating the test block.
    threshold = select_threshold(y[slices['validation']], sklearn_model.predict_proba(x_val)[:, 1])
    all_x = prep.transform(engineer(raw, variant)).astype(np.float64)
    predictions = split_table.copy()
    for name, (w, b) in trained.items():
        predictions[name] = torch.sigmoid(torch.tensor(all_x @ w + b)).numpy()
    predictions['baseline'] = baseline.predict_proba(all_x)[:, 1]
    predictions.to_csv(OUT / 'predictions.csv', index=False)
    rows = []
    for split in ['validation', 'test']:
        mask = split_names == split
        for name in ['sklearn', 'manual', 'standard', 'baseline']:
            rows.append({'split': split, 'model': name, **evaluate(y[mask], predictions.loc[mask, name], threshold)})
        for name, p in [('call_all', np.ones(mask.sum())), ('call_none', np.zeros(mask.sum()))]:
            rows.append({'split': split, 'model': name, **evaluate(y[mask], p, 0.5)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    coefficients = pd.DataFrame({'feature': feature_names, **{k: v[0] for k, v in trained.items()}})
    coefficients.to_csv(OUT / 'coefficients.csv', index=False)
    agreement = {}
    for name in ['manual', 'standard']:
        w, b = trained[name]
        agreement[name] = {'max_probability_difference_all_rows': float(np.max(np.abs(predictions[name] - predictions.sklearn))),
                           'max_weight_difference': float(np.max(np.abs(w - trained['sklearn'][0]))),
                           'intercept_difference': float(abs(b - trained['sklearn'][1])),
                           'test_decision_disagreements': int(np.sum((predictions.loc[split_names == 'test', name] >= threshold) != (predictions.loc[split_names == 'test', 'sklearn'] >= threshold)))}
    diagnostics = {'rows': len(raw), 'columns': raw.shape[1], 'null_cells': int(raw.isna().sum().sum()),
                   'exact_duplicate_rows': int(raw.duplicated().sum()),
                   'sentinel_with_positive_previous_count': int((raw.pdays.eq(999) & raw.previous.gt(0)).sum()),
                   'unknown_counts': {c: int(raw[c].eq('unknown').sum()) for c in raw.select_dtypes(include=['object', 'str'])},
                   'split_summary': {k: {'rows': len(y[s]), 'positive_rate': float(y[s].mean()),
                                          'first_row_id': s.start, 'last_row_id': s.stop - 1} for k, s in slices.items()},
                   'unseen_categories': {split: {c: sorted(set(engineer(raw, variant).iloc[s][c].dropna()) - set(engineer(raw, variant).iloc[slices['train']][c].dropna()))
                    for c in engineer(raw, variant).select_dtypes(exclude='number')} for split, s in slices.items() if split != 'train'}}
    metadata = {'seed': SEED, 'variant': variant, 'lambda': lam, 'sklearn_C': float(sklearn_model.C),
                'learning_rate': lr, 'gradient_tolerance': GRAD_TOL, 'threshold': threshold,
                'call_cost_eur': 2, 'missed_subscriber_cost_eur': 50, 'n_features': len(feature_names),
                'training_seconds': durations, 'agreement': agreement, 'diagnostics': diagnostics,
                'excluded_columns': EXCLUDED, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'versions': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','torch','gradio']},
                'intercepts': {k: float(v[1]) for k, v in trained.items()}}
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (OUT / 'training_history.json').write_text(json.dumps(histories, indent=2))
    print(metrics[metrics.split == 'test'].to_string(index=False))
    print('Training complete. App will load saved artifacts without fitting anything.')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
