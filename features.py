"""Pre-call features shared by training and saved-model inference."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUSTOMER_CATEGORIES = ['job', 'marital', 'education', 'default', 'housing', 'loan']
EXCLUDED = ['duration', 'campaign', 'contact', 'month', 'day_of_week',
            'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']


def engineer(raw: pd.DataFrame, variant: str = 'history') -> pd.DataFrame:
    """Whitelist fields: never expose final-call summaries or the outcome to X.

    Economic indicators are also omitted: release-time/vintage information is
    absent, so this exercise cannot verify their availability at scoring time.
    """
    x = raw[['age', *CUSTOMER_CATEGORIES, 'poutcome']].copy()
    x[CUSTOMER_CATEGORIES + ['poutcome']] = x[CUSTOMER_CATEGORIES + ['poutcome']].fillna('unknown')
    if variant == 'raw':
        x['pdays'] = raw['pdays']
        x['previous'] = raw['previous']
    elif variant in ('history', 'history_context'):
        known = raw['pdays'].ne(999) & raw['pdays'].notna()
        x['recency_recorded'] = known.astype(float)
        # 0 is a neutral placeholder; the separate indicator distinguishes it
        # from an actual zero-day interval. Never treat 999 as elapsed days.
        x['log_recency_days'] = np.log1p(raw['pdays'].where(known, 0))
        x['log_previous_contacts'] = np.log1p(raw['previous'])
        if variant == 'history_context':
            x['loan_combination'] = x['housing'].astype(str) + '/' + x['loan'].astype(str)
            x['unknown_customer_fields'] = x[CUSTOMER_CATEGORIES].eq('unknown').sum(axis=1)
    else:
        raise ValueError(f'Unknown feature variant: {variant}')
    return x


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric = x.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in x if c not in numeric]
    return ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ], verbose_feature_names_out=False)
