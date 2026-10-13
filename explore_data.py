"""Steps 1–2: inspect the raw dataset without cleaning or training anything.

Run from the project folder: uv run python explore_data.py
"""
from pathlib import Path
import argparse

import pandas as pd
import plotly.express as px
import plotly.io as pio

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'exploration')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # This CSV uses Windows-1251 encoding, which preserves Cyrillic model names.
    data = pd.read_csv(ROOT / 'data' / 'car_ad.csv', encoding='cp1251')

    # Step 1: a DataFrame is a table with named columns and numbered rows.
    print('FIRST FIVE ROWS')
    print(data.head().to_string(index=False))
    print(f'\nRows: {data.shape[0]:,}; columns: {data.shape[1]}')
    print('\nCOLUMN TYPES')
    print(data.dtypes.to_string())

    # Step 2: count issues; do not remove rows or change values.
    missing = pd.DataFrame({
        'missing_count': data.isna().sum(),
        'missing_percent': (data.isna().mean() * 100).round(2),
    })
    checks = pd.Series({
        'Exact duplicate rows beyond the first occurrence': int(data.duplicated().sum()),
        'Zero advertised prices': int(data['price'].eq(0).sum()),
        'Negative advertised prices': int(data['price'].lt(0).sum()),
        'Zero mileage (may be legitimate)': int(data['mileage'].eq(0).sum()),
        'Negative mileage': int(data['mileage'].lt(0).sum()),
        'Engine below 0.5 litres (review flag)': int(data['engV'].lt(0.5).sum()),
        'Engine above 10 litres (review flag)': int(data['engV'].gt(10).sum()),
        'Manufacturing year after collection year 2016': int(data['year'].gt(2016).sum()),
        'Fuel type labelled Other': int(data['engType'].eq('Other').sum()),
    }, name='count')
    numeric = data[['price', 'mileage', 'engV', 'year']].describe(
        percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99])
    print('\nMISSING VALUES\n' + missing.to_string())
    print('\nQUALITY CHECKS\n' + checks.to_string())
    print('\nNUMERIC SUMMARY\n' + numeric.to_string())

    missing.to_csv(args.output_dir / 'missing_values.csv')
    checks.to_csv(args.output_dir / 'quality_checks.csv')
    numeric.to_csv(args.output_dir / 'numeric_summary.csv')

    # Keep all values visible, including zeros and unusually expensive cars.
    figures = []
    for column, label in [('price', 'Advertised price (USD)'),
                          ('mileage', 'Mileage (thousands of km)'),
                          ('engV', 'Engine displacement (litres)'),
                          ('year', 'Manufacturing year')]:
        fig = px.histogram(data, x=column, nbins=60,
                           title=f'Distribution: {label}', labels={column: label})
        fig.update_layout(template='plotly_white', yaxis_title='Number of advertisements')
        figures.append(fig)
    counts = data['car'].value_counts().head(15)
    fig = px.bar(x=counts.index, y=counts.values, title='15 most frequent brands',
                 labels={'x': 'Brand', 'y': 'Number of advertisements'})
    fig.update_layout(template='plotly_white')
    figures.append(fig)

    sections = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Car advertisements — raw data exploration</title>',
        '<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;'
        'color:#172033}table{border-collapse:collapse;margin:16px 0;display:block;overflow:auto}'
        'td,th{padding:8px;border:1px solid #ddd}h1,h2{color:#183b68}</style></head><body>',
        '<h1>Car advertisements: raw data exploration</h1>',
        f'<p>{len(data):,} advertisements · {len(data.columns)} columns · Ukraine, 2016</p>',
        '<p>This is an inspection of the original CSV. No rows have been removed, '
        'no missing values filled, and no model trained. Review flags are not proof '
        'of incorrect data. Missing numeric values are omitted by histograms.</p>',
        '<h2>First five rows</h2>', data.head().to_html(index=False),
        '<h2>Missing values</h2>', missing.to_html(),
        '<h2>Quality checks</h2>', checks.to_frame().to_html(),
        '<h2>Numeric summary</h2>', numeric.round(2).to_html(),
        '<h2>Distributions</h2>',
        '<p>The entire numeric range is included. Use the chart controls to zoom '
        'into the dense regions. Charts do not establish causal relationships.</p>',
    ]
    for i, fig in enumerate(figures):
        # Embed Plotly once so the HTML works without an internet connection.
        sections.append(pio.to_html(fig, full_html=False, include_plotlyjs=True if i == 0 else False))
    sections.append('</body></html>')
    report = args.output_dir / 'raw_data_exploration.html'
    report.write_text('\n'.join(sections), encoding='utf-8')
    print(f'\nExploration saved to: {report}')


if __name__ == '__main__':
    main()
