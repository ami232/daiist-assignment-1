"""Inference regression checks; run with python -m pytest test_dashboard.py."""
from unittest.mock import patch
import numpy as np
import pytest


def test_saved_inference_matches_test_artifacts_without_training():
    with patch('sklearn.linear_model.Ridge.fit', side_effect=AssertionError('Unexpected training')), \
         patch('sklearn.compose.ColumnTransformer.fit', side_effect=AssertionError('Unexpected preprocessing fit')), \
         patch('torch.optim.SGD.step', side_effect=AssertionError('Unexpected training')):
        import app
        rows = app.pd.read_csv(app.ROOT/'data/prepared/test.csv')
        x = app.PREPROCESSOR.transform(app.make_features(rows))
        np.testing.assert_allclose(app.SKLEARN.predict(x), app.TEST[app.MODEL_NAMES[0]])
        app.build_dashboard()
        d = app.DEFAULT
        args = [d.car, d.model, d.body, int(d.year), float(d.mileage)*1000,
                float(d.engV), d.engType, d.drive, d.registration]
        reference, adjusted, table, _ = app.predict_price(*args, -10)
        expected = float(app.SKLEARN.predict(app.PREPROCESSOR.transform(app.make_features(app.TRAIN.iloc[[0]])))[0])
        assert reference == f'${expected:,.0f}'
        assert adjusted == f'${expected*0.9:,.0f}'
        np.testing.assert_allclose(table.iloc[:,1], expected, atol=0.01, rtol=0)
        bad = args.copy(); bad[4] = -1
        with pytest.raises(app.gr.Error):
            app.predict_price(*bad, 0)
        bad = args.copy(); bad[3] = 2020
        with pytest.raises(app.gr.Error):
            app.predict_price(*bad, 0)
        for label in app.DISTRIBUTIONS:
            assert len(app.distribution(label).data) > 0
        assert len(app.comparison_plot(app.MODEL_NAMES).data) == 4
