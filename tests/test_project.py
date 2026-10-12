"""Checks aimed at leakage, objective equivalence, persistence and policy errors."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from features import engineer, make_preprocessor
from evaluation import policy_counts, select_threshold
from train import objective
from app import Dashboard

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def dashboard():
    return Dashboard()


def test_pre_call_features_cannot_see_outcome_or_final_call(dashboard):
    raw = dashboard.raw.iloc[:8].copy()
    expected = engineer(raw, 'history_context')
    for col in dashboard.meta['excluded_columns'] + ['y']:
        raw[col] = 'forbidden'
    pd.testing.assert_frame_equal(expected, engineer(raw, 'history_context'))


def test_sentinel_and_unknown_are_preserved_explicitly(dashboard):
    raw = dashboard.raw.iloc[:2].copy()
    raw['pdays'] = [999, 0]
    raw['housing'] = ['unknown', 'yes']
    x = engineer(raw, 'history_context')
    assert x.recency_recorded.tolist() == [0., 1.]
    assert x.log_recency_days.tolist() == [0., 0.]
    assert x.housing.tolist() == ['unknown', 'yes']
    assert x.unknown_customer_fields.iloc[0] >= 1


def test_chronological_partition_and_train_only_statistics(dashboard):
    p = dashboard.predictions
    assert len(p) == len(dashboard.raw) == p.row_id.nunique()
    train, val, test = [p.loc[p.split == s, 'row_id'] for s in ['train', 'validation', 'test']]
    assert train.max() < val.min() <= val.max() < test.min()
    ref = make_preprocessor(dashboard.features.iloc[train])
    expected = ref.fit_transform(dashboard.features.iloc[train])
    actual = dashboard.preprocessor.transform(dashboard.features.iloc[train])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    perturbed = dashboard.features.iloc[val].copy()
    perturbed['age'] = 10000
    before = dashboard.preprocessor.named_transformers_['numeric'].named_steps['standardscaler'].mean_.copy()
    dashboard.preprocessor.transform(perturbed)
    np.testing.assert_array_equal(before, dashboard.preprocessor.named_transformers_['numeric'].named_steps['standardscaler'].mean_)


def test_manual_gradient_matches_analytic_and_finite_difference():
    x = torch.tensor([[1., 2.], [-1., .5], [0., -1.]], dtype=torch.float64)
    y = torch.tensor([1., 0., 1.], dtype=torch.float64)
    w = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
    lam = .01
    loss = objective(x, y, w, b, lam)
    loss.backward()
    error = torch.sigmoid(x @ w.detach() + b.detach()) - y
    torch.testing.assert_close(w.grad, x.T @ error / len(y) + lam * w.detach())
    torch.testing.assert_close(b.grad, error.mean())
    eps = 1e-6
    for j in range(2):
        delta = torch.zeros(2, dtype=torch.float64)
        delta[j] = eps
        numeric = (objective(x, y, w.detach() + delta, b.detach(), lam) - objective(x, y, w.detach() - delta, b.detach(), lam)) / (2 * eps)
        assert numeric.item() == pytest.approx(w.grad[j].item(), abs=1e-9)


def test_sklearn_minimizes_the_same_regularized_objective(dashboard):
    ids = dashboard.predictions.split.eq('train').to_numpy()
    x = torch.tensor(dashboard.preprocessor.transform(dashboard.features[ids]), dtype=torch.float64)
    y = torch.tensor(dashboard.predictions.loc[ids, 'actual'].to_numpy(), dtype=torch.float64)
    w = torch.tensor(dashboard.sklearn.coef_[0], requires_grad=True)
    b = torch.tensor(dashboard.sklearn.intercept_[0], requires_grad=True)
    objective(x, y, w, b, dashboard.meta['lambda']).backward()
    assert max(w.grad.abs().max().item(), b.grad.abs().item()) < 1e-6
    assert dashboard.sklearn.C == pytest.approx(1 / (len(y) * dashboard.meta['lambda']))


def test_loaded_predictions_match_exports_and_methods_agree(dashboard):
    ids = [0, 24712, 30000, 32950, 41187]
    actual = dashboard.predict_rows(dashboard.raw.iloc[ids])
    for name, p in actual.items():
        np.testing.assert_allclose(p, dashboard.predictions.iloc[ids][name], atol=1e-12)
    for name in ['manual', 'standard']:
        assert np.max(np.abs(dashboard.predictions[name] - dashboard.predictions.sklearn)) < 1e-3
        assert dashboard.meta['agreement'][name]['test_decision_disagreements'] <= 2


def test_threshold_selection_including_ties_and_endpoints():
    y, p = np.array([1, 0, 1, 0]), np.array([.9, .8, .3, .3])
    t = select_threshold(y, p)
    costs = [policy_counts(y, p, v)['cost_eur'] for v in [0, .3, .8, .9, 1]]
    assert policy_counts(y, p, t)['cost_eur'] == min(costs)
    assert policy_counts(y, p, 0)['calls'] == 4
    assert policy_counts(y, p, 1)['calls'] == 0
    assert policy_counts(y, p, 0)['cost_eur'] == 8
    assert policy_counts(y, p, 1)['cost_eur'] == 100


def test_frozen_threshold_is_derived_only_from_validation(dashboard):
    v = dashboard.predictions.query('split == "validation"')
    assert dashboard.meta['threshold'] == pytest.approx(select_threshold(v.actual, v.sklearn), abs=1e-14)


def test_app_can_load_and_predict_when_fit_is_forbidden(monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from sklearn.compose import ColumnTransformer
    def forbidden(*args, **kwargs):
        raise AssertionError('App attempted to fit')
    monkeypatch.setattr(LogisticRegression, 'fit', forbidden)
    monkeypatch.setattr(ColumnTransformer, 'fit', forbidden)
    monkeypatch.setattr(ColumnTransformer, 'fit_transform', forbidden)
    d = Dashboard()
    assert len(d.predict_rows(d.raw.iloc[[0]])['sklearn']) == 1
    summary, cm, cost = d.policy('manual', 'test', 0, 2, 50)
    assert '8,238 calls' in summary
    assert sum(sum(row) for row in cm.data[0].z) == 8238


def test_dashboard_costs_reproduce_exported_metrics(dashboard):
    for split in ['validation', 'test']:
        rows = dashboard.predictions.query('split == @split')
        for name in ['sklearn', 'manual', 'standard', 'baseline']:
            c = policy_counts(rows.actual, rows[name], dashboard.meta['threshold'])
            saved = dashboard.metrics.query('split == @split and model == @name').iloc[0]
            for key in ['tn', 'fp', 'fn', 'tp', 'calls', 'cost_eur']:
                assert c[key] == saved[key]


def test_conflicting_history_is_not_silently_erased(dashboard):
    mask = dashboard.raw.pdays.eq(999) & dashboard.raw.previous.gt(0)
    assert mask.sum() == 4110
    x = engineer(dashboard.raw.loc[mask], 'history_context')
    assert x.recency_recorded.eq(0).all()
    assert x.log_previous_contacts.gt(0).all()
    assert x.poutcome.eq('failure').all()
