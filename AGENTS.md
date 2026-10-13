# AGENTS.md — Spanish Energy Market Analysis (DL)

Instructions for any coding agent (or human) working in this repo. Read this
before touching anything. If a request conflicts with a rule below, the rule
wins unless the user explicitly overrides it in the conversation.

## 1. What this project is

A deep-learning study of the Spanish energy market: scrape market/grid data,
build a polars-based feature pipeline, train sk-learn and torch models tracked with
MLflow, run statistical/simulation analysis on top, and (later) serve
results through a Gradio dashboard.

This repo started from a course scaffold (`daiist-assignment1`) and keeps
its grading contract — see §2. Everything else is being rebuilt to match the
structure (domain packages under a `src/`, a single
`constants.py` + `config.py` + `logger.py`, notebooks per pipeline stage),
but simplified: that repo's `Config` class (string-dispatch `study` branches,
deeply nested `PATHS` dict) is _not_ to be copied as-is. Keep the same idea
— one place for paths/params, one place for literals — flatter and smaller.

## 2. Hard constraints — do not violate

- **Never move, rename, or gut** `main.py`, `train.py`, `app.py`, `README.md`,
  `REPORT.md`. CI (`.github/workflows/verify.yml`) runs
  `uv run python main.py train` then `uv run python main.py app` and fails
  the build if either breaks. `main.py`'s two subcommands are fixed.
- `train.py` and `app.py` are **entrypoints only**: they parse no business
  logic of their own beyond orchestration — they call into `src/`. No
  feature engineering, model code, or plotting logic lives directly in
  either file.
- `app.py` must never retrain anything at startup — it loads artifacts
  (`models/`, MLflow runs, saved parquet) that `train.py` already produced.
- `README.md` / `REPORT.md` content stays whatever the user is drafting
  there — do not restructure or overwrite their prose.

## 3. Directory layout

```
.
├── main.py, train.py, app.py, README.md, REPORT.md   # fixed, see §2
├── AGENTS.md                                          # this file
├── CODE_MAP.md                                        # living index, see §9
├── pyproject.toml, uv.lock, .python-version
├── src/
│   ├── __init__.py
│   ├── constants.py        # literals only: column names, URLs, enums, paths' fixed parts
│   ├── config.py            # single Config object: resolved paths + run params
│   ├── logger.py            # get_logger(name) factory, stdlib logging
│   ├── data/                 # scrape, ingest, clean, validate, engineer features: raw -> interim -> processed -> features
│   ├── model/                 # torch model defs, train/eval loops, MLflow logging
│   ├── stats/                   # correlation, independence, seasonality, distribution shift, simulation, downstream analysis
│   ├── plotting/                  # plotly chart builders + styled-dataframe display helper
│   └── synthetic/                   # synthetic data generators, schema-faithful, used by tests/
├── notebooks/                # thin, mirror src/ stages 1:1, no duplicated logic
│   ├── 00_scraping.ipynb
│   ├── 01_pipeline_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb
│   ├── 04_statistics_simulation.ipynb
│   └── 05_downstream_analysis.ipynb
├── scripts/                  # thin .sh wrappers, no logic beyond flags/env
│   ├── run_scrape.sh
│   ├── run_pipeline.sh
│   ├── run_train.sh
│   ├── run_tests.sh
│   └── run_mlflow_ui.sh
├── tests/                    # pytest, mirrors src/ package-for-package
│   ├── conftest.py
│   └── test_<module>.py
├── data/                      # gitignored: raw/, interim/, processed/
├── mlruns/                     # gitignored: local MLflow tracking store
└── models/                      # gitignored: serialized artifacts app.py loads
```

Rules for this tree:

- Root directories are exactly `src/`, `notebooks/`, `scripts/`, `data/`,
  `tests/`, `mlruns/`, `models/` — no others without asking first.
- A subpackage only gets created when there is real content for it — no
  empty scaffolding committed ahead of need.
- `src/data/` (code: scraping, cleaning, feature engineering) and
  top-level `data/` (storage: raw/interim/processed files) are different
  things that happen to share a name — `src/data` is a Python package,
  `data/` never contains code. Be explicit ("the `src/data` package" vs
  "the `data/` folder") whenever it could be ambiguous.
- Add `data/`, `mlruns/`, `models/` to `.gitignore` the moment they exist
  with real content.

## 4. Coding conventions

- **OOP, modular, boring.** One class = one responsibility. No abstraction
  (base classes, plugin registries, config-driven dispatch) until there are
  ≥2 concrete cases that need it.
- **Simplicity is the tie-breaker.** If two designs are both correct, ship
  the one with less code and fewer moving parts. Delete rather than
  deprecate; no compatibility shims.
- **Type hints on every function signature and attribute.** Skip the
  docstring whenever the name already says it — most functions get none.
  When one is genuinely needed, keep it to ~4-6 words, one line (e.g.
  `"""Drops duplicate market ticks."""`). Longer prose only for logic
  complex enough to warrant it (a non-obvious algorithm, a real edge case,
  units/assumptions the type hints can't carry).
- **Long signatures: one argument per line.** Any function/method
  definition or call that doesn't fit cleanly on one line — and always
  once it has more than ~2 arguments — breaks one argument per line with a
  trailing comma:
  ```python
  def build_rolling_features(
      df: pl.DataFrame,
      window: int,
      target_col: str,
      *,
      min_periods: int | None = None,
  ) -> pl.DataFrame:
      ...
  ```
- **No dead flexibility.** No `**kwargs` passthroughs, no unused parameters
  "for later," no config flags without an actual second caller.
- Stack is fixed: **polars** (never pandas), **plotly** (never matplotlib),
  **torch**, **MLflow** for experiment tracking, **gradio** for the
  dashboard (already the `app.py` contract), **tqdm** for progress on any
  loop over more than a handful of network calls or files. Don't introduce
  alternatives to any of these without asking.
- Notebooks import and reuse code from `src/` (or any python helper module
  under `scripts/`) instead of redefining it. Because `src/` keeps
  changing while a kernel stays alive, put `%load_ext autoreload` +
  `%autoreload 2` at the top of every notebook, or call
  `importlib.reload(module)` right after an edit — never restart the
  kernel just to pick up a `src/` change. When a notebook cell's logic
  stabilizes, promote it into the matching `src/` module and import it.

## 5. `constants.py` / `config.py` / `logger.py`

Modeled on `sucden_internship/config_manager`, flattened:

- `constants.py` — only literals: column name strings, source URLs/endpoints,
  fixed enums/maps. No logic, no path joining.
- `config.py` — one `Config` (dataclass or plain class), built once, that
  resolves every filesystem path (via `PROJECT_DIR = Path(__file__)...`) and
  holds run parameters (window sizes, split dates, MLflow experiment name).
  No string-dispatched branches like the reference repo's `study` argument —
  if `data`/`model`/`stats` need different params, give each its own
  small config section, not an if/else inside one giant method.
- `logger.py` — a single `get_logger(name: str) -> logging.Logger` factory
  (console + optional file handler). No custom `Logger` wrapper class unless
  a second, genuinely different need for one shows up.

## 6. Statistics & simulation (`src/stats/`)

Expected surface, one module per concern:

- `correlation.py` — rolling/expanding correlation, cross-correlation.
- `independence.py` — independence/autocorrelation tests (e.g. Ljung-Box,
  runs test, chi-square).
- `seasonality.py` — seasonal decomposition, periodicity detection.
- `distribution_shift.py` — distributional drift detection (e.g.
  Kolmogorov-Smirnov, population stability index, rolling moments).
- `simulation.py` — Monte Carlo / bootstrap simulation helpers.

Downstream, decision-facing analysis (scenario/backtests) belongs here too
once there's enough of it to warrant its own `analysis.py` — it's built on
the primitives above plus `model` outputs. Keep those primitives reusable
and untangled from any one downstream analysis.

## 7. Plotting & display (`src/plotting/`)

- One module for plotly figure builders (time series, distributions,
  correlation heatmaps, etc.), returning `go.Figure` objects — never call
  `.show()` inside a library function.
- One small helper, e.g. `style_df(df: pl.DataFrame, precision: int = 2) -> ...`,
  wrapping the pandas-Styler pattern the user wants
  (`df.style.format(precision=...).background_gradient()...`) for clean
  notebook table display. Since polars has no `.style`, this helper converts
  to pandas only at the display boundary — never upstream of it.

## 8. Testing (`tests/`)

- `pytest`, one `test_<module>.py` per `src/` module that has real logic.
- `src/synthetic/` provides schema-faithful fake data generators (polars
  frames matching the raw/processed schemas) so pipeline, feature, and
  model tests never hit the network or require committed fixture data.
- Cover the _pipelines_ end-to-end (scrape → pipeline → features → train
  → analysis, using synthetic data at the scrape boundary), plus the
  statistics functions against known analytical/synthetic ground truth
  (e.g. a manufactured seasonal series must be detected as seasonal).
- No mocks that fake away the thing under test — use synthetic data instead.

## 9. `CODE_MAP.md` — mandatory, updated every change

`CODE_MAP.md` is the single index of everything in `src/`: every file, its
classes (with attributes and methods), and its module-level functions.
**Any change that adds, removes, renames, or changes the signature of
something in `src/` is not complete until `CODE_MAP.md` reflects it** —
treat it the same as updating a caller after a rename.

Format, one section per file:

```markdown
## src/data/clean.py

### class RawDataCleaner

Attributes:

- df: pl.DataFrame — raw input frame

Methods:

- `__init__`(self, df: pl.DataFrame) -> None
- drop_duplicates(self) -> pl.DataFrame — dedupes on (date, node_id)

### functions

- normalize_timestamps(df: pl.DataFrame, tz: str = "Europe/Madrid") -> pl.DataFrame
```

## 10. Scripts (`scripts/*.sh`) & incremental fetches

Thin wrappers only — `uv run python -m src.<module> ...` or
`uv run python main.py <stage>` plus flags/env vars. No pipeline logic in
shell. One script per pipeline stage (scrape, pipeline, train, tests,
MLflow UI); `scripts/fetch_omie.sh` (backed by `src/data/omie.py`) is the
reference example.

Any scraper/fetcher must be **adaptive/idempotent**, not re-download
everything on every run:

- Partition the work into small units keyed by their output path (e.g. one
  parquet per `(variable, year, month)`).
- Before fetching a unit, check whether its output file already exists; if
  so, skip it. Add a `--force` flag to override.
- Never skip the still-in-progress current period (today's day/month/etc.)
  just because a stale partial file exists for it — always re-fetch that
  one.
- Wrap the unit loop in `tqdm` so progress is visible on multi-hour pulls.

This is what makes re-running a fetch over a wider or shifted date range
cheap: only the genuinely missing units are downloaded.

## 11. Environment & tooling

- Python pinned via `.python-version` (currently 3.12).
- Conda env `ie_env` provides the interpreter; `uv` manages
  dependencies/lockfile against it — point uv at the conda env instead of
  its default `.venv`:
  ```bash
  conda activate ie_env
  export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
  uv sync
  ```
- Add new deps with `uv add <pkg>` (commits `pyproject.toml` + `uv.lock`).
- MLflow tracking store is local (`./mlruns`, gitignored); one experiment
  per project, one run per training invocation, params/metrics/artifacts
  logged — no manual bookkeeping in notebooks.
- Gradio (`app.py`) is a later phase — don't scaffold it early; build it
  once `train.py` produces real artifacts to load.
