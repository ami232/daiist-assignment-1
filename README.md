# Bank marketing: pre-call prioritization

Assignment 1 uses one logistic-regression model trained with scikit-learn, manual PyTorch autograd and gradient updates, and `nn.Module` + `torch.optim.SGD`. The dashboard loads saved models and never retrains them.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then run from this folder:

```bash
uv sync --frozen
uv run python main.py train
uv run python main.py app
```

Open http://127.0.0.1:7860. Stop the server with Ctrl+C. Training rebuilds `artifacts/` from the committed CSV without downloads, prompts, API keys or notebook execution. Python 3.12 and the required packages are pinned through `.python-version`, `pyproject.toml` and `uv.lock`. The first `uv sync` needs Internet access; training then uses local data. The dashboard runs locally with public sharing and analytics disabled.

```bash
uv run python -m pytest -q
```

`main.py` is unchanged from the assignment. The unused starter notebooks are preserved in `docs/` to avoid conflicting root entrypoints. The supplied GitHub workflow still uses the two required commands. See `docs/DELIVERY.md` for current publication status. Blackboard submission remains for the student.

## Business decision

Before a planned call, estimate whether a customer will subscribe to a term deposit. The student chose hypothetical costs of EUR 2 per call and EUR 50 per missed subscriber. The retrospective policy cost is:

```text
cost = 2 * (true positives + false positives) + 50 * false negatives
```

A call costs money even when it succeeds. These are assumptions, not bank profit measurements. Historical subscriptions do not identify the causal effect of calling.

The complete ordered UCI dataset has 41,188 rows and is about 5.8 MB. It exceeds the brief's approximate row suggestion but remains small, committed, and fast to retrain. See `data/README.md` for provenance and attribution.

## Evaluation and features

- Earliest 60% train; next 20% validate; latest 20% test. No shuffle.
- All imputers, scaling statistics and category vocabularies are learned on training only.
- Compare raw fields, engineered campaign history, and engineered history plus loan combination / unknown-field count.
- The student chose correct `pdays=999` semantics over the small validation advantage of the raw control. Select the best engineered configuration using validation log loss.
- Compare only three regularization strengths. Select the decision threshold using validation cost, then freeze it for all models on test.
- Exclude duration, final-call details and completed-campaign counts. Exclude macroeconomic series without verified publication vintages.

For all methods, the objective is mean binary cross-entropy plus `lambda/2 * ||w||²`, with an unpenalized intercept. Match scikit-learn with `C = 1 / (n_train * lambda)`. The two PyTorch versions use float64, identical zero initialization, the same full-batch learning rate and the same training-gradient stopping condition. The learning rate is computed from a training-only curvature bound, not selected on the test set.

## Dashboard

1. **Overview:** final test outcome, direct cost comparison with calling everyone, and the recommendation in plain language.
2. **Call policy:** percentage threshold, explicit call-all/call-none/reset buttons, call/miss counters, labeled confusion matrix and cost breakdown. Assumptions and sensitivity curves expand when needed.
3. **Models:** visual score comparison, explanations of metrics and calibration, with full tables and individual predictions available on demand.
4. **Data:** human-readable feature labels, target percentages across chronological blocks, and expandable feature definitions/experiments.
5. **Records:** real saved-model inference for a historical record, plus optional technical coefficient details.

Test controls are explanatory, not permission to retune the frozen evaluation.

## Files to read

- `REPORT.md`: assignment report, student decisions and interpretation.
- `docs/DECISIONS.md`: decision log.
- `features.py`: feature definitions and preprocessing.
- `train.py`: model training, validation-only selection and exported evaluation.
- `modeling.py`: PyTorch logistic module and inference helpers.
- `evaluation.py`: threshold selection, metrics and business-cost arithmetic.
- `app.py`: saved-model dashboard.
- `tests/test_project.py`: checks against leakage, objective mismatch and inference/training coupling.
- `artifacts/metadata.json`: exact split sizes, source hash, versions, selected settings and agreement.
- `artifacts/feature_experiments.csv`, `metrics.csv`, `coefficients.csv`, `predictions.csv`: measured evidence.
- `output/pdf/Project_Explanation_and_Defense_Guide.pdf`: separate AI-generated study guide.

## Reproduce and prepare the backup

```bash
uv run python main.py train
uv run python -m pytest -q
uv run python scripts/package_project.py
```

The packaging script excludes `.git`, `.venv`, caches and temporary files, and creates a ZIP beside the project folder. It does not upload, push, open a PR or submit anything. Review `REPORT.md` and the study guide before uploading the PR link and ZIP to Blackboard yourself.

Original assignment: https://github.com/ami232/daiist-assignment-1. Its original README is retained in `docs/ASSIGNMENT.md`.

Explanatory prose was revised with Walter Writes and checked for factual accuracy by Codex. The full AI-use disclosure is in `REPORT.md`.
