"""Dealership dashboard. Loads saved artifacts; never trains models."""
from pathlib import Path
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from prepare_data import make_features

ROOT = Path(__file__).resolve().parent
REQUIRED = ['data/prepared/preprocessor.joblib', 'data/prepared/train.csv',
            'results/sklearn_log/model.joblib', 'results/pytorch/manual.pt',
            'results/pytorch/standard.pt', 'results/test/comparison.csv',
            'results/test/predictions.csv']
missing = [name for name in REQUIRED if not (ROOT/name).exists()]
if missing:
    raise RuntimeError('Missing saved artifacts. Run python main.py train first: '+', '.join(missing))
PREPROCESSOR = joblib.load(ROOT/REQUIRED[0])
TRAIN = pd.read_csv(ROOT/'data/prepared/train.csv')
SKLEARN = joblib.load(ROOT/'results/sklearn_log/model.joblib')
TORCH_MODELS = {name: torch.load(ROOT/f'results/pytorch/{name}.pt', weights_only=True)
                for name in ['manual', 'standard']}
TEST = pd.read_csv(ROOT/'results/test/predictions.csv')
METRICS = pd.read_csv(ROOT/'results/test/comparison.csv')
MODEL_NAMES = ['scikit-learn log-price Ridge', 'PyTorch manual', 'PyTorch standard']
BRANDS = sorted(TRAIN.car.dropna().unique().tolist())
DEFAULT = TRAIN.iloc[0]


def models_for_brand(brand):
    return sorted(TRAIN.loc[TRAIN.car == brand, 'model'].dropna().unique().tolist())


def update_models(brand):
    choices = models_for_brand(brand)
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def predict_price(brand, model, body, year, mileage_km, engine, fuel, drive, registration, adjustment):
    numbers = [year, mileage_km, engine, adjustment]
    if any(value is None or not np.isfinite(value) for value in numbers):
        raise gr.Error('Enter finite values for year, mileage, engine size and price adjustment.')
    if year != int(year) or not TRAIN.year.min() <= year <= 2016:
        raise gr.Error(f'Year must be a whole number from {int(TRAIN.year.min())} to 2016.')
    if mileage_km < 0 or not 0.5 <= engine <= 10 or not -30 <= adjustment <= 30:
        raise gr.Error('Mileage must be nonnegative, engine size 0.5–10 L, and adjustment between -30% and +30%.')
    if brand not in BRANDS or model not in models_for_brand(brand):
        raise gr.Error('Choose a model belonging to the selected brand.')
    for field, value in [('body', body), ('engType', fuel), ('drive', drive), ('registration', registration)]:
        if value not in TRAIN[field].dropna().unique():
            raise gr.Error(f'Select a valid {field} value.')
    row = pd.DataFrame([{'car': brand, 'model': model, 'body': body, 'year': year,
                         'mileage': mileage_km/1000, 'engV': engine, 'engType': fuel,
                         'drive': drive, 'registration': registration}])
    x = PREPROCESSOR.transform(make_features(row))
    prices = [float(SKLEARN.predict(x)[0])]
    for name in ['manual', 'standard']:
        artifact = TORCH_MODELS[name]
        prices.append(float(np.exp(x @ artifact['weights'].numpy()+artifact['bias'].item())[0]))
    if not np.isfinite(prices).all() or min(prices) <= 0:
        raise gr.Error('These inputs do not produce a usable price. Check the vehicle details.')
    table = pd.DataFrame({'Implementation': MODEL_NAMES, 'Estimated asking price (USD)': np.round(prices, 2)})
    price = prices[0]
    note = ('The employee-adjusted price is a scenario, not another model prediction. '
            'The data cannot estimate how a discount changes selling time.')
    if mileage_km > TRAIN.mileage.max()*1000:
        note += ' Mileage is above the observed training range; this is extrapolation.'
    count = int(((TRAIN.car == brand) & (TRAIN.model == model)).sum())
    if count < 20:
        note += f' This brand/model has only {count} training advertisements and is grouped as a rare category.'
    return f'${price:,.0f}', f'${price*(1+adjustment/100):,.0f}', table, note


def comparison_plot(selected):
    fig = go.Figure()
    for name in selected or []:
        fig.add_trace(go.Scatter(x=TEST.actual_price_USD, y=TEST[name], mode='markers',
                                name=name, marker={'size': 5, 'opacity': 0.45}))
    maximum = float(max(TEST.actual_price_USD.max(), TEST[MODEL_NAMES].max().max()))
    fig.add_trace(go.Scatter(x=[0, maximum], y=[0, maximum], mode='lines',
                            name='Perfect prediction', line={'dash': 'dash', 'color': 'gray'}))
    fig.update_layout(template='plotly_white', xaxis_title='Actual asking price (USD)',
                      yaxis_title='Predicted asking price (USD)', title='Held-out test advertisements', height=470)
    return fig


DISTRIBUTIONS = {'Asking price (USD)': 'price', 'Mileage (thousand km)': 'mileage',
                 'Engine size (L)': 'engV', 'Manufacturing year': 'year',
                 'Brand': 'car', 'Body type': 'body', 'Fuel': 'engType', 'Drive': 'drive'}


def distribution(label):
    column = DISTRIBUTIONS[label]
    if pd.api.types.is_numeric_dtype(TRAIN[column]):
        fig = px.histogram(TRAIN, x=column, nbins=40, labels={column: label})
    else:
        counts = TRAIN[column].value_counts().rename_axis(label).reset_index(name='Advertisements')
        fig = px.bar(counts, x=label, y='Advertisements')
    fig.update_layout(template='plotly_white', title=f'Training data: {label}', height=430,
                      yaxis_title='Advertisements')
    return fig


def build_dashboard():
    with gr.Blocks(title='Car Price Advisor') as demo:
        gr.Markdown('# Car Price Advisor\nA reference asking price for dealership employees.')
        gr.Markdown('**Historical prototype · Ukraine, 2016 · USD.** Estimates reflect advertised prices, '
                    'not completed sales or current market prices. Test MAPE: **25.65%**; '
                    'individual errors can be substantially higher.')
        with gr.Tab('Price recommendation'):
            with gr.Row():
                with gr.Column():
                    brand = gr.Dropdown(BRANDS, value=DEFAULT.car, label='Brand')
                    model = gr.Dropdown(models_for_brand(DEFAULT.car), value=DEFAULT.model, label='Model')
                    body = gr.Dropdown(sorted(TRAIN.body.unique()), value=DEFAULT.body, label='Body type')
                    year = gr.Number(value=int(DEFAULT.year), label='Manufacturing year (up to 2016)', precision=0)
                    mileage = gr.Number(value=float(DEFAULT.mileage)*1000, label='Mileage (km)', minimum=0)
                with gr.Column():
                    engine = gr.Number(value=float(DEFAULT.engV), label='Engine displacement (L)', minimum=0.5, maximum=10)
                    fuel = gr.Dropdown(sorted(TRAIN.engType.unique()), value=DEFAULT.engType, label='Fuel (Other = unknown)')
                    drive = gr.Dropdown(sorted(TRAIN.drive.unique()), value=DEFAULT.drive, label='Drive type')
                    registration = gr.Dropdown(sorted(TRAIN.registration.unique()), value=DEFAULT.registration,
                                               label='Registered in Ukraine')
                    adjustment = gr.Slider(-30, 30, value=0, step=1, label='Employee price adjustment (%)')
            button = gr.Button('Calculate reference price', variant='primary')
            with gr.Row():
                reference = gr.Textbox(label='Model reference price (USD)', interactive=False)
                adjusted = gr.Textbox(label='Employee-adjusted asking price (USD)', interactive=False)
            table = gr.Dataframe(label='Three implementations of the same model', interactive=False)
            note = gr.Markdown('Select the vehicle details and calculate a reference price.')
            brand.change(update_models, inputs=brand, outputs=model, api_name=False)
            button.click(predict_price, inputs=[brand, model, body, year, mileage, engine, fuel, drive, registration, adjustment],
                         outputs=[reference, adjusted, table, note], api_name='recommend_price')
        with gr.Tab('Model evaluation'):
            gr.Markdown('All three models use identical inputs and the same log-price regression. '
                        'Their predictions overlap because they solve the same problem. '
                        'These results use 1,259 held-out test advertisements. MAPE is an error measure, not accuracy.')
            gr.Dataframe(METRICS[['model', 'MAPE_percent', 'MAE_USD', 'median_absolute_percentage_error']].round(2), interactive=False)
            selected = gr.CheckboxGroup(MODEL_NAMES, value=MODEL_NAMES, label='Models to display')
            plot = gr.Plot(comparison_plot(MODEL_NAMES))
            selected.change(comparison_plot, inputs=selected, outputs=plot, api_name='comparison_plot')
        with gr.Tab('Explore the data'):
            gr.Markdown('Distributions from the 5,855 training advertisements. Mileage in this plot uses '
                        'the dataset unit (thousand km); the recommendation form accepts km.')
            feature = gr.Dropdown(list(DISTRIBUTIONS), value='Asking price (USD)', label='Variable')
            chart = gr.Plot(distribution('Asking price (USD)'))
            feature.change(distribution, inputs=feature, outputs=chart, api_name='distribution')
        gr.Markdown('The employee makes the final pricing decision. This app loads saved models and does not train at startup.')
    return demo


if __name__ == '__main__':
    build_dashboard().launch(server_name='127.0.0.1', server_port=int(os.environ.get('GRADIO_SERVER_PORT', '7860')),
                             inbrowser=False, share=False)
