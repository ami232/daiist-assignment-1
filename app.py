"""
Assignment 1 — F1 Pit Stop Strategy Dashboard

Loads pre-trained model artifacts (never retrains).
Four tabs:
  1. Model Comparison — ROC curves, metrics table, confusion matrices
  2. Feature Analysis  — feature importance + interactive distributions
  3. Threshold Tool   — slider to tune decision threshold with live business-cost
  4. Monaco Simulation — lap-by-lap circuit view with pit-stop alerts
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, precision_score, recall_score
from pathlib import Path
import gradio as gr

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"

# ── Load pre-computed artifacts ────────────────────────────────────────────────
with open(MODELS_DIR / "dashboard_data.pkl", "rb") as f:
    D = pickle.load(f)

FEATURE_NAMES = D['feature_names']
X_TEST        = D['X_test']
Y_TEST        = D['y_test']
RESULTS       = D['results']
TRAIN_SAMPLE  = D['train_sample']
SK_COEFFS     = D['sk_coeffs']
HISTORY_MAN   = D['history_manual']
HISTORY_NN    = D['history_nn']

PROBAS = {
    'sklearn': D['proba_sklearn'],
    'manual':  D['proba_manual'],
    'nn':      D['proba_nn'],
}
MODEL_LABEL = {
    'sklearn': 'scikit-learn',
    'manual':  'Manual PyTorch',
    'nn':      'PyTorch nn.Module',
}
COLORS = {
    'sklearn': '#E10600',
    'manual':  '#1E3A5F',
    'nn':      '#00A86B',
    'naive':   '#888888',
}

FP_COST = 2_000
FN_COST = 30_000


# ── Monaco simulation setup ────────────────────────────────────────────────────

class LogReg(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.fc = nn.Linear(n_in, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


COMPOUND_SOFTNESS = {'WET': 0, 'INTERMEDIATE': 1, 'HARD': 2, 'MEDIUM': 3, 'SOFT': 4}
COMPOUND_LIFE     = {'WET': 20, 'INTERMEDIATE': 25, 'HARD': 45, 'MEDIUM': 35, 'SOFT': 25}
COMPOUND_COLOR    = {
    'SOFT':         '#FF1801',
    'MEDIUM':       '#FFF200',
    'HARD':         '#ECECEC',
    'INTERMEDIATE': '#39B54A',
    'WET':          '#0067FF',
}

# Real 2025 F1 Monaco GP grid
REAL_DRIVERS = [
    'NOR', 'LEC', 'VER', 'HAM', 'RUS', 'PIA', 'ALO', 'STR',
    'GAS', 'OCO', 'HUL', 'ALB', 'SAI', 'TSU', 'LAW',
    'ANT', 'BEA', 'HAD', 'DOO',
]

# ── Monaco circuit waypoints (approximate, clockwise) ─────────────────────────
# Coordinate space: (0,0) bottom-left, x = east, y = north
# S/F is at bottom-center; circuit climbs north to Casino then returns via tunnel

# Waypoints derived carefully from the official Monaco circuit map (clockwise).
# Coordinate space: x = west→east (0-100), y = south→north (0-100).
# S/F is on the west harbourfront; the circuit climbs north to Casino,
# swings right to the Mirabeau hairpin, descends through Portier, dives
# into the underground tunnel going west, then returns via the Pool section.
_WP = np.array([
    # ── S/F → Sainte Devote ──────────────────────────────────────────────────
    [20, 44],  #  0  START / FINISH  (harbour, west side)
    [20, 52],  #  1  straight up
    [20, 65],  #  2  SAINTE DEVOTE (1) — sharp right
    # ── Beau Rivage / Massenet / Casino ──────────────────────────────────────
    [24, 67],  #  3  exit Sainte Devote going right
    [40, 67],  #  4  Beau Rivage straight
    [56, 67],  #  5  BEAU RIVAGE (2)
    [66, 67],  #  6  approaching Massenet
    [71, 67],  #  7  MASSENET (3)
    [70, 73],  #  8  CASINO (4) — right, heading north-east
    # ── Mirabeau hairpin complex ──────────────────────────────────────────────
    [76, 78],  #  9  post-Casino, bearing right
    [83, 79],  # 10  MIRABEAU HAUTE (5) — right
    [89, 76],  # 11  corner 6
    [92, 72],  # 12  MIRABEAU BAS (7) — tightest right-hand hairpin
    [92, 67],  # 13  exit hairpin, heading south
    [92, 64],  # 14  PORTIER (8) — right
    # ── Tunnel ───────────────────────────────────────────────────────────────
    [90, 57],  # 15  tunnel approach
    [89, 54],  # 16  TUNNEL ENTRANCE  ← underground starts here
    [78, 53],  # 17  tunnel
    [66, 53],  # 18  tunnel mid
    [54, 52],  # 19  TUNNEL EXIT      ← surface resumes here
    # ── Nouvelle Chicane / Tabac ──────────────────────────────────────────────
    [53, 50],  # 20  NOUVELLE CHICANE (10) — right
    [49, 50],  # 21  chicane exit (11) — left
    [39, 50],  # 22  TABAC (12) — left
    # ── Swimming Pool ─────────────────────────────────────────────────────────
    [30, 49],  # 23  Pool entry (13)
    [25, 45],  # 24  Louis Chiron (14)
    [23, 41],  # 25  Pool mid (15)
    [26, 36],  # 26  Pool exit (16)
    # ── La Rascasse → Anthony Noghes → S/F ───────────────────────────────────
    [32, 29],  # 27  LA RASCASSE (17) — left
    [21, 26],  # 28  toward Anthony Noghes
    [17, 23],  # 29  ANTHONY NOGHES (18)
    [14, 24],  # 30  corner 19
    [17, 34],  # 31  back straight
    [20, 44],  # 32  CLOSE — back to S/F
])

_TUNNEL_IN  = 16   # waypoint index where circuit goes underground
_TUNNEL_OUT = 19   # waypoint index where it resurfaces


def _build_circuit(wp: np.ndarray, n: int = 600):
    """Interpolate waypoints into a smooth n-point circuit path. Returns (cx, cy, t_wp)."""
    diffs     = np.diff(wp, axis=0)
    segs      = np.hypot(diffs[:, 0], diffs[:, 1])
    cum       = np.concatenate([[0], np.cumsum(segs)])
    t_wp      = cum / cum[-1]               # fraction of total circuit at each waypoint
    t         = np.linspace(0, 1, n)
    cx        = np.interp(t, t_wp, wp[:, 0])
    cy        = np.interp(t, t_wp, wp[:, 1])
    return cx, cy, t_wp


CIRCUIT_X, CIRCUIT_Y, _WP_T = _build_circuit(_WP)
_TUNNEL_T_IN  = _WP_T[_TUNNEL_IN]
_TUNNEL_T_OUT = _WP_T[_TUNNEL_OUT]

# Slices for drawing tunnel vs surface
_surf_mask   = np.ones(len(CIRCUIT_X), dtype=bool)
_tunnel_mask = (_WP_T[_TUNNEL_IN] <= np.linspace(0, 1, len(CIRCUIT_X))) & \
               (np.linspace(0, 1, len(CIRCUIT_X)) <= _WP_T[_TUNNEL_OUT])
_surf_mask[_tunnel_mask] = False

# Named corners — (plot_x, plot_y, anchor)
_CORNERS = [
    ('Sainte Devote',      16,  68, 'right'),
    ('Beau Rivage',        44,  70, 'center'),
    ('Massenet',           73,  70, 'left'),
    ('Casino',             68,  76, 'left'),
    ('Mirabeau Haute',     80,  82, 'left'),
    ('Mirabeau Bas',       96,  72, 'left'),
    ('Portier',            96,  64, 'left'),
    ('Tunnel',             72,  49, 'center'),
    ('Nouvelle Chicane',   51,  46, 'center'),
    ('Tabac',              36,  46, 'center'),
    ('La Rascasse',        35,  26, 'center'),
    ('Anthony Noghes',     10,  20, 'center'),
]


def _track_xy(fraction: float):
    """Return (x, y) on circuit for a given arc-fraction [0, 1)."""
    fraction = fraction % 1.0
    idx = int(fraction * (len(CIRCUIT_X) - 1))
    return float(CIRCUIT_X[idx]), float(CIRCUIT_Y[idx])


def _make_features_df(df: pd.DataFrame) -> np.ndarray:
    """Replicate the feature engineering from train.py."""
    d = df.copy()
    d['compound_code']   = d['Compound'].map(COMPOUND_SOFTNESS).fillna(2).astype('float32')
    d['life_expected']   = d['Compound'].map(COMPOUND_LIFE).fillna(35).astype('float32')
    d['tyre_wear_ratio'] = (d['TyreLife'] / d['life_expected']).astype('float32')
    d['tyre_life_sq']    = (d['TyreLife'] ** 2).astype('float32')
    d['compound_x_age']  = (d['compound_code'] * d['TyreLife']).astype('float32')
    d['race_remaining']  = (1.0 - d['RaceProgress']).astype('float32')
    d['is_soft']         = (d['Compound'] == 'SOFT').astype('float32')
    d['is_medium']       = (d['Compound'] == 'MEDIUM').astype('float32')
    d['is_hard']         = (d['Compound'] == 'HARD').astype('float32')
    cols = FEATURE_NAMES  # same order as training
    return d[cols].fillna(0).values.astype('float32')


def _load_monaco_data() -> pd.DataFrame | None:
    """Load Monaco 2025 data for real F1 drivers, run all three models."""
    raw = pd.read_csv(ROOT / "train (2).csv")
    df  = raw[(raw['Race'] == 'Monaco Grand Prix') & (raw['Year'] == 2025)].copy()
    df  = df[df['Driver'].isin(REAL_DRIVERS)].copy()
    if df.empty:
        return None

    X = _make_features_df(df)

    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    Xs = scaler.transform(X).astype('float32')

    with open(MODELS_DIR / "sklearn_logreg.pkl", "rb") as f:
        sk = pickle.load(f)
    sk_p = sk.predict_proba(Xs)[:, 1]

    Xt = torch.from_numpy(Xs)
    w = torch.load(MODELS_DIR / "manual_weights.pt", weights_only=True)
    with torch.no_grad():
        man_p = torch.sigmoid(Xt @ w['w'] + w['b']).squeeze().numpy()

    n_feat   = Xs.shape[1]
    nn_model = LogReg(n_feat)
    nn_model.load_state_dict(torch.load(MODELS_DIR / "nn_model.pt", weights_only=True))
    nn_model.eval()
    with torch.no_grad():
        nn_p = torch.sigmoid(nn_model(Xt)).squeeze().numpy()

    df = df.reset_index(drop=True)
    df['p_sklearn'] = sk_p
    df['p_manual']  = man_p
    df['p_nn']      = nn_p
    df['pred_sk']   = (sk_p  >= 0.5).astype(int)
    df['pred_man']  = (man_p >= 0.5).astype(int)
    df['pred_nn']   = (nn_p  >= 0.5).astype(int)
    df['any_pit']   = ((sk_p >= 0.5) | (man_p >= 0.5) | (nn_p >= 0.5)).astype(int)
    return df


print("Loading Monaco 2025 data...")
MONACO_DF   = _load_monaco_data()
MONACO_LAPS = sorted(MONACO_DF['LapNumber'].unique()) if MONACO_DF is not None else []
MONACO_MIN  = int(MONACO_LAPS[0])  if MONACO_LAPS else 1
MONACO_MAX  = int(MONACO_LAPS[-1]) if MONACO_LAPS else 1
print(f"  {len(MONACO_DF):,} rows across {len(MONACO_LAPS)} laps")


def _monaco_circuit_traces() -> list:
    """Return base plotly traces for the circuit (call once, reuse per lap)."""
    traces = []
    t_arr = np.linspace(0, 1, len(CIRCUIT_X))
    pre   = t_arr < _TUNNEL_T_IN
    post  = t_arr > _TUNNEL_T_OUT
    tun   = ~pre & ~post

    # ── Track surface (thick dark line = asphalt) ──────────────────────────
    traces.append(go.Scatter(
        x=np.concatenate([CIRCUIT_X[pre], [None], CIRCUIT_X[post]]),
        y=np.concatenate([CIRCUIT_Y[pre], [None], CIRCUIT_Y[post]]),
        mode='lines',
        line=dict(color='#3a3a3a', width=10),
        hoverinfo='none', showlegend=False, name='surface',
    ))
    # White centre line on surface
    traces.append(go.Scatter(
        x=np.concatenate([CIRCUIT_X[pre], [None], CIRCUIT_X[post]]),
        y=np.concatenate([CIRCUIT_Y[pre], [None], CIRCUIT_Y[post]]),
        mode='lines',
        line=dict(color='#cccccc', width=2, dash='dot'),
        hoverinfo='none', showlegend=False, name='centre',
    ))
    # ── Tunnel (thinner + dashed to show underground) ─────────────────────
    traces.append(go.Scatter(
        x=CIRCUIT_X[tun], y=CIRCUIT_Y[tun],
        mode='lines',
        line=dict(color='#5a5a5a', width=6, dash='dash'),
        hoverinfo='none', showlegend=False, name='tunnel',
    ))
    # ── Start/Finish marker ───────────────────────────────────────────────
    sf_x, sf_y = float(CIRCUIT_X[0]), float(CIRCUIT_Y[0])
    traces.append(go.Scatter(
        x=[sf_x], y=[sf_y], mode='markers+text',
        marker=dict(symbol='line-ns', size=18, color='white',
                    line=dict(color='white', width=3)),
        text=['  S/F'], textposition='middle right',
        textfont=dict(size=9, color='white'),
        hoverinfo='none', showlegend=False,
    ))
    # ── Corner labels ─────────────────────────────────────────────────────
    for (name, cx, cy, anchor) in _CORNERS:
        traces.append(go.Scatter(
            x=[cx], y=[cy], mode='text',
            text=[name],
            textfont=dict(size=8, color='#bbbbbb'),
            textposition='middle center',
            hoverinfo='none', showlegend=False,
        ))
    return traces


_BASE_TRACES = _monaco_circuit_traces()


def make_monaco_figure(lap: int) -> go.Figure:
    fig = go.Figure(data=list(_BASE_TRACES))  # shallow copy is fine for base traces

    if MONACO_DF is None:
        fig.update_layout(title="Monaco data not available")
        return fig

    lap_df = MONACO_DF[MONACO_DF['LapNumber'] == lap].copy()
    n_cars = len(lap_df)

    if n_cars == 0:
        fig.update_layout(title=f"Monaco GP 2025 — Lap {lap} (no data for these drivers)")
        _apply_circuit_layout(fig, lap)
        return fig

    lap_df = lap_df.sort_values('Position').reset_index(drop=True)

    # Spread cars around the circuit. Leader (pos=1) just crossed the finish line.
    # Each position behind is 4% of the circuit further back.
    GAP = 0.04
    for _, row in lap_df.iterrows():
        pos      = int(row['Position'])
        driver   = row['Driver']
        compound = row['Compound']
        pit_pred = bool(row['any_pit'])

        frac       = (1.0 - (pos - 1) * GAP) % 1.0
        car_x, car_y = _track_xy(frac)

        dot_size   = 14 if pit_pred else 10
        dot_symbol = 'circle'
        border_col = '#FFD700' if pit_pred else '#1a1a1a'
        border_w   = 3 if pit_pred else 1
        dot_color  = COMPOUND_COLOR.get(compound, '#888')

        hover = (
            f"<b>P{pos} — {driver}</b><br>"
            f"Compound: {compound}<br>"
            f"Tyre age: {row['TyreLife']:.0f} laps<br>"
            f"Stint: {int(row['Stint'])}<br>"
            f"Lap time delta: {row['LapTime_Delta']:+.2f}s<br>"
            f"──────────────────<br>"
            f"sklearn  pit prob: {row['p_sklearn']:.0%}<br>"
            f"Manual   pit prob: {row['p_manual']:.0%}<br>"
            f"nn.Module pit prob: {row['p_nn']:.0%}<br>"
            f"Actual pit next lap: {'YES' if row['PitNextLap'] == 1.0 else 'no'}"
            "<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=[car_x], y=[car_y],
            mode='markers+text',
            name=driver,
            text=[f" {driver}"],
            textposition='top center',
            textfont=dict(size=8, color='white'),
            marker=dict(
                size=dot_size, color=dot_color, symbol=dot_symbol,
                line=dict(color=border_col, width=border_w),
            ),
            hovertemplate=hover,
        ))

    _apply_circuit_layout(fig, lap)
    return fig


def _apply_circuit_layout(fig: go.Figure, lap: int) -> None:
    n_on_track = len(MONACO_DF[MONACO_DF['LapNumber'] == lap]) if MONACO_DF is not None else 0
    fig.update_layout(
        title=dict(
            text=f"Monaco Grand Prix 2025 — Lap {lap} / {MONACO_MAX}  "
                 f"({n_on_track} driver{'s' if n_on_track != 1 else ''} with data)",
            font=dict(color='white', size=14),
        ),
        xaxis=dict(range=[-5, 102], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-5, 90], showgrid=False, zeroline=False, visible=False,
                   scaleanchor='x', scaleratio=1),
        plot_bgcolor='#0f1117',
        paper_bgcolor='#0f1117',
        font=dict(color='white'),
        showlegend=False,
        height=560,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    # Compound legend (static annotation)
    for i, (comp, col) in enumerate(COMPOUND_COLOR.items()):
        fig.add_annotation(
            x=2, y=74 - i * 6,
            text=f'● {comp}', showarrow=False,
            font=dict(size=9, color=col),
            xanchor='left',
        )
    fig.add_annotation(
        x=2, y=74 - len(COMPOUND_COLOR) * 6,
        text='● gold ring = pit predicted',
        showarrow=False, font=dict(size=8, color='#FFD700'), xanchor='left',
    )


def make_pit_alerts(lap: int) -> str:
    if MONACO_DF is None:
        return "_No Monaco data loaded._"

    lap_df = MONACO_DF[MONACO_DF['LapNumber'] == lap].sort_values('Position')

    if lap_df.empty:
        return f"### Lap {lap}\n_No data for real drivers on this lap._"

    pit_df = lap_df[lap_df['any_pit'] == 1]

    if pit_df.empty:
        n = len(lap_df)
        return (
            f"### Lap {lap} — No pit alerts\n\n"
            f"All three models agree: none of the {n} driver{'s' if n != 1 else ''} "
            f"with data on this lap will pit next lap."
        )

    lines = [f"### Lap {lap} — {len(pit_df)} Pit Alert{'s' if len(pit_df) > 1 else ''}\n"]

    for _, row in pit_df.iterrows():
        actual  = row.get('PitNextLap', np.nan)
        outcome = "✅ **Pitted**" if actual == 1.0 else ("❌ Did not pit" if actual == 0.0 else "—")

        # Which models flagged it
        flags = []
        if row['pred_sk']:  flags.append("sklearn")
        if row['pred_man']: flags.append("Manual PyTorch")
        if row['pred_nn']:  flags.append("nn.Module")

        lines.append(
            f"---\n"
            f"**P{int(row['Position'])} — {row['Driver']}**\n\n"
            f"| | |\n|---|---|\n"
            f"| Compound | {row['Compound']} |\n"
            f"| Tyre age | {row['TyreLife']:.0f} laps |\n"
            f"| Stint | {int(row['Stint'])} |\n"
            f"| Lap Δ time | {row['LapTime_Delta']:+.2f} s |\n"
            f"| sklearn prob | {row['p_sklearn']:.1%} |\n"
            f"| Manual prob | {row['p_manual']:.1%} |\n"
            f"| nn.Module prob | {row['p_nn']:.1%} |\n"
            f"| **Models flagging** | {', '.join(flags)} |\n"
            f"| **Actual outcome** | {outcome} |\n"
        )

    return "\n".join(lines)


# ── Static plot builders (existing tabs) ──────────────────────────────────────

def make_roc_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash='dash', color='#cccccc', width=1))
    for key, proba in PROBAS.items():
        fpr, tpr, _ = roc_curve(Y_TEST, proba)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{MODEL_LABEL[key]} (AUC={roc_auc:.3f})",
            line=dict(color=COLORS[key], width=2.5),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name='Naive (AUC=0.500)',
        line=dict(color=COLORS['naive'], width=1, dash='dot'),
    ))
    fig.update_layout(
        title='ROC Curves — all three models vs. naive baseline',
        xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        height=420, legend=dict(x=0.55, y=0.15),
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


def make_metrics_df() -> pd.DataFrame:
    rows = []
    for key in ['sklearn', 'manual', 'nn', 'naive']:
        m = RESULTS[key]
        rows.append({
            'Model':     MODEL_LABEL.get(key, 'Naive baseline'),
            'Accuracy':  f"{m['accuracy']:.4f}",
            'F1':        f"{m['f1']:.4f}",
            'AUC-ROC':   f"{m['auc']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall':    f"{m['recall']:.4f}",
        })
    return pd.DataFrame(rows)


def make_cm_subplots() -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[MODEL_LABEL[k] for k in ['sklearn', 'manual', 'nn']],
    )
    labels = ['No Pit', 'Pit']
    for col_idx, key in enumerate(['sklearn', 'manual', 'nn'], start=1):
        cm = RESULTS[key]['conf_mat'].tolist()
        fig.add_trace(
            go.Heatmap(z=cm, x=labels, y=labels, colorscale='Blues',
                       texttemplate="%{z}", showscale=(col_idx == 3)),
            row=1, col=col_idx,
        )
    fig.update_layout(
        title='Confusion Matrices at decision threshold = 0.5',
        height=360, margin=dict(l=60, r=20, t=80, b=50),
    )
    fig.update_xaxes(title_text='Predicted')
    fig.update_yaxes(title_text='Actual', col=1)
    return fig


def make_loss_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(HISTORY_MAN)+1)), y=HISTORY_MAN,
                             name='Manual PyTorch', line=dict(color=COLORS['manual'])))
    fig.add_trace(go.Scatter(x=list(range(1, len(HISTORY_NN)+1)),  y=HISTORY_NN,
                             name='PyTorch nn.Module', line=dict(color=COLORS['nn'])))
    fig.update_layout(
        title='Training Loss Curves (PyTorch methods)',
        xaxis_title='Epoch', yaxis_title='Weighted BCE Loss',
        height=350, margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


def make_coeff_figure() -> go.Figure:
    coeffs = np.array(SK_COEFFS)
    order  = np.argsort(np.abs(coeffs))[::-1][:12]
    names  = [FEATURE_NAMES[i] for i in order]
    vals   = [coeffs[i] for i in order]
    colors = [COLORS['sklearn'] if v > 0 else COLORS['manual'] for v in vals]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors))
    fig.update_layout(
        title='Feature Coefficients — scikit-learn logistic regression',
        xaxis_title='Feature', yaxis_title='Coefficient',
        height=380, margin=dict(l=50, r=20, t=50, b=100),
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def make_dist_figure(feature: str) -> go.Figure:
    fig = go.Figure()
    for label, display, color in [(0, 'No Pit (0)', COLORS['manual']),
                                   (1, 'Pit (1)',    COLORS['sklearn'])]:
        vals = TRAIN_SAMPLE.loc[TRAIN_SAMPLE['target'] == label, feature]
        fig.add_trace(go.Histogram(x=vals, name=display, opacity=0.65,
                                   marker_color=color, nbinsx=60))
    fig.update_layout(
        barmode='overlay',
        title=f'Distribution of {feature} by target class',
        xaxis_title=feature, yaxis_title='Count',
        height=380, margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(x=0.75, y=0.95),
    )
    return fig


def threshold_update(model_display: str, threshold: float):
    key_map = {v: k for k, v in MODEL_LABEL.items()}
    key     = key_map[model_display]
    proba   = PROBAS[key]
    y_pred  = (proba >= threshold).astype(int)

    cm             = confusion_matrix(Y_TEST, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_cost     = fp * FP_COST + fn * FN_COST
    f1_val         = f1_score(Y_TEST, y_pred, zero_division=0)
    prec_val       = precision_score(Y_TEST, y_pred, zero_division=0)
    rec_val        = recall_score(Y_TEST, y_pred, zero_division=0)

    labels = ['No Pit', 'Pit']
    cm_fig = go.Figure(go.Heatmap(
        z=cm.tolist(), x=labels, y=labels,
        colorscale='Blues', texttemplate="%{z}", showscale=True,
    ))
    cm_fig.update_layout(
        title=f'{model_display} — threshold={threshold:.2f}',
        xaxis_title='Predicted', yaxis_title='Actual',
        height=380, margin=dict(l=60, r=20, t=60, b=50),
    )

    cost_md = f"""
### Strategy Cost at threshold {threshold:.2f}

| | No Pit predicted | Pit predicted |
|---|---|---|
| **Actually no pit** | {tn:,} (TN) | {fp:,} (FP) |
| **Actually pits**   | {fn:,} (FN) | {tp:,} (TP) |

| Cost driver | Count | Unit cost | Subtotal |
|---|---|---|---|
| False Positives (wasted prep) | {fp:,} | €{FP_COST:,} | €{fp * FP_COST:,.0f} |
| False Negatives (missed undercut) | {fn:,} | €{FN_COST:,} | €{fn * FN_COST:,.0f} |
| **Total strategy cost** | | | **€{total_cost:,.0f}** |

**F1:** {f1_val:.4f} &nbsp;|&nbsp; **Precision:** {prec_val:.4f} &nbsp;|&nbsp; **Recall:** {rec_val:.4f}
"""
    return cm_fig, cost_md


# ── Gradio app ────────────────────────────────────────────────────────────────
with gr.Blocks(title="F1 Pit Stop Strategy") as demo:

    gr.Markdown("""
# F1 Pit Stop Strategy Dashboard
**Predict when a rival team will pit — protect your track position.**
All predictions come from pre-trained models. Nothing is retrained here.
""")

    with gr.Tabs():

        # ── Tab 1: Model Comparison ────────────────────────────────────────────
        with gr.Tab("Model Comparison"):
            gr.Markdown("### ROC Curves")
            gr.Plot(value=make_roc_figure())
            gr.Markdown("### Performance Metrics (threshold = 0.5)")
            gr.DataFrame(value=make_metrics_df(), interactive=False)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Confusion Matrices")
                    gr.Plot(value=make_cm_subplots())
                with gr.Column():
                    gr.Markdown("### Training Loss (PyTorch methods)")
                    gr.Plot(value=make_loss_figure())

        # ── Tab 2: Feature Analysis ────────────────────────────────────────────
        with gr.Tab("Feature Analysis"):
            gr.Markdown("### Feature Importance (scikit-learn coefficients)")
            gr.Plot(value=make_coeff_figure())
            gr.Markdown("### Feature Distributions by Target")
            feat_dd   = gr.Dropdown(choices=FEATURE_NAMES, value='TyreLife', label="Feature")
            dist_plot = gr.Plot()
            feat_dd.change(fn=make_dist_figure, inputs=feat_dd, outputs=dist_plot)
            demo.load(fn=lambda: make_dist_figure('TyreLife'), outputs=dist_plot)

        # ── Tab 3: Threshold Tool ──────────────────────────────────────────────
        with gr.Tab("Threshold Tool"):
            gr.Markdown("""
### Tune the Decision Threshold
- **Lower threshold** → more alerts → fewer missed undercuts (FN↓) but more false alarms (FP↑)
- **Higher threshold** → fewer alerts → fewer false alarms (FP↓) but more missed undercuts (FN↑)

**Assumed costs:** €2,000 per false alarm · €30,000 per missed rival undercut
""")
            with gr.Row():
                model_radio = gr.Radio(
                    choices=list(MODEL_LABEL.values()), value='scikit-learn', label="Model",
                )
                thr_slider  = gr.Slider(
                    minimum=0.01, maximum=0.99, value=0.5, step=0.01, label="Decision threshold",
                )
            with gr.Row():
                thr_cm_plot = gr.Plot()
                thr_cost_md = gr.Markdown()

            demo.load(fn=lambda: threshold_update('scikit-learn', 0.5),
                      outputs=[thr_cm_plot, thr_cost_md])
            model_radio.change(fn=threshold_update, inputs=[model_radio, thr_slider],
                               outputs=[thr_cm_plot, thr_cost_md])
            thr_slider.change(fn=threshold_update, inputs=[model_radio, thr_slider],
                              outputs=[thr_cm_plot, thr_cost_md])

        # ── Tab 4: Monaco Simulation ───────────────────────────────────────────
        with gr.Tab("Monaco Simulation"):
            gr.Markdown(f"""
### Monaco Grand Prix 2025 — Lap-by-Lap Strategy View

Dots show the {len(REAL_DRIVERS)} real 2025 F1 drivers. Not all appear on every lap
(the dataset has partial coverage per driver). **Colour = tyre compound.
⭐ star = any model predicts this car will pit next lap.** Hover over a dot for full details.
""")
            lap_slider = gr.Slider(
                minimum=MONACO_MIN, maximum=MONACO_MAX,
                value=MONACO_MIN, step=1, label="Lap",
            )
            with gr.Row():
                circuit_plot = gr.Plot(scale=3)
                with gr.Column(scale=2):
                    pit_alerts_md = gr.Markdown()

            def _update_monaco(lap):
                return make_monaco_figure(int(lap)), make_pit_alerts(int(lap))

            demo.load(fn=lambda: _update_monaco(MONACO_MIN),
                      outputs=[circuit_plot, pit_alerts_md])
            lap_slider.change(fn=_update_monaco, inputs=lap_slider,
                              outputs=[circuit_plot, pit_alerts_md])


demo.launch(theme=gr.themes.Base())
