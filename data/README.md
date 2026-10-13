# Dataset and cleaning record

Source: [Car Sale Advertisements](https://www.kaggle.com/datasets/antfarol/car-sale-advertisements),
Anton Bobanov. Advertisements collected in Ukraine in 2016.

- `car_ad.csv`: unmodified original, Windows-1251 encoding, 9,576 rows.
- `processed/car_ad_clean.csv`: cleaned copy, UTF-8, 8,364 rows.
- `processed/removed_rows.csv`: 1,212 removed rows with original CSV line numbers
  and all applicable removal reasons.
- `processed/cleaning_summary.json`: counts and original-file checksum.

Run `uv run python clean_data.py` to reproduce the cleaned copy.

## Rules applied

- Keep the first occurrence of an exact duplicate.
- Remove missing, nonfinite or nonpositive target prices.
- Remove negative or nonfinite mileage where mileage is present.
- Remove nonfinite, nonpositive, fractional manufacturing years, or years after 2016.
- Remove observed engine sizes outside 0.5–10 litres. This is a heuristic scope
  restriction, not a verified label that every excluded observation is erroneous.
  Some valid unusual vehicles may be excluded. Missing engine sizes and drive types are removed by student choice.

There were 267 zero-price rows, 113 duplicate rows and 57 rows outside the engine
range. These counts overlap: their sum is not the number of unique rows removed.
No negative mileage or invalid manufacturing year was found under these rules.

High prices and zero mileage are retained because they are not inherently invalid.
Rows with missing engine displacement or drive type are excluded. Other unknown
inputs are handled by training-only preprocessing.
No prices or other missing values have been imputed. The `source_row` audit column
must not be included as a model feature.

## Units

- `price`: advertised asking price in USD, not final transaction price.
- `mileage`: thousands of kilometres.
- `engV`: thousands of cubic centimetres, equivalent to litres.
- `year`: manufacturing year, not publication date.
- `registration`: registration in Ukraine.

The report's rationale should be written or reviewed by the student; this file
records the actual implementation and does not claim a causal or business finding.

## Prepared data

Run `uv run python prepare_data.py` to rerun cleaning and preparation.
`prepared/` contains train/validation/test CSVs, source-row assignments,
`arrays.npz`, a fitted `preprocessor.joblib` and `preparation_summary.json`.
This preparation script does not fit prediction models. The complete training
pipeline subsequently fits and saves them under `results/`. See REPORT.md for
the modeling choices and limitations.
