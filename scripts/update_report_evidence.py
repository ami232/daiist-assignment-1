"""Format measured facts and the student's supplied interpretation into REPORT.md."""
from pathlib import Path
import csv
import json
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'artifacts/metadata.json').read_text())
with (ROOT/'artifacts/metrics.csv').open() as f: metrics=list(csv.DictReader(f))
with (ROOT/'artifacts/feature_experiments.csv').open() as f: experiments=list(csv.DictReader(f))
text='''# Assignment 1 Report

- **Name**: Javier Sánchez Esquivel
- **Student ID**: 19731
- **Email**: jsanchez.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I selected bank marketing and call prioritization. The source is UCI's **Bank Marketing** dataset, specifically `bank-additional-full.csv`: 41,188 client/campaign records, 20 original inputs and one binary target, `y` (term-deposit subscription). It is ordered by date from May 2008 to November 2010, according to the included data dictionary. A record includes customer attributes and campaign-contact summaries; exact timestamps and customer identifiers are not supplied.

I chose the full ordered version to evaluate later campaigns. Its 5,834,924-byte size makes it practical to commit and retrain, even though its row count exceeds the assignment's approximate 20k-row guideline. The CSV is bundled unchanged, with provenance in `data/README.md` and its SHA-256 in `artifacts/metadata.json`. There is no runtime data download.

Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), Moro, Rita and Cortez, DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306), licensed CC BY 4.0. Publication: Moro, Cortez and Rita (2014), *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems 62, 22-31, DOI [10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).

## Business / real-life framing

The decision is which customers to prioritize **before a planned call**, using information available beforehand. I selected hypothetical costs of **EUR 2 per call** and **EUR 50 per missed subscriber**. A call costs money even when successful:

```text
policy cost = 2 * (TP + FP) + 50 * FN
```

These are illustrative assumptions, not measured bank profit. Historical subscription is the prediction target; the data does not reveal whether an additional call causes a subscription. The model is not an uplift estimator.

The chronological split is fixed: earliest 60% for training, next 20% for validation and latest 20% for testing. This matches my choice to evaluate future campaign conditions. All learned preprocessing is fitted on training only. Validation log loss compares a small number of model configurations; validation policy cost selects the threshold. The same frozen threshold is then used for all three models on test. No test-based retuning is performed.

| Block | Rows | Zero-based source row IDs | Subscriber rate |
|---|---:|---|---:|
'''
for split,d in m['diagnostics']['split_summary'].items():
 text+=f'| {split} | {d["rows"]:,} | {d["first_row_id"]:,}-{d["last_row_id"]:,} | {d["positive_rate"]:.3%} |\n'
text+='''
## Data preparation & feature engineering

My feature ideas were to distinguish previous-contact status and campaign history, because returning customers may respond differently, and to test loan combinations and incomplete customer records, because financial commitments and missing information may distinguish response patterns.

The implementation retains age; job, marital status, education, default, housing and loan categories; and prior-campaign outcome. It excludes duration, completed-campaign contact count, and final-call channel/month/weekday because their pre-call availability is not established. It also conservatively excludes macroeconomic series without publication-time/vintage information. This reduces available predictive signal but keeps the implemented information boundary explicit. Customer profiles and prior-campaign history are assumed available in the CRM before a new call.

| Feature | Implementation |
|---|---|
| Recency recorded | Indicator for `pdays != 999`; 999 is a sentinel, not elapsed days. It does not reliably mean no historical contact, since 4,110 records combine it with `previous > 0`. |
| Log recency | `log1p(pdays)` when observed; otherwise zero, distinguished by the indicator. |
| Log previous contacts | `log1p(previous)`, compressing count differences. |
| Loan combination | Combined housing/loan category, including unknown values. |
| Unknown-field count | Count of `unknown` across the six customer categories. |

Numeric inputs are median-imputed defensively and standardized using training statistics. Categorical inputs are one-hot encoded from the training vocabulary, retaining `unknown`; unseen categories are handled without refitting. There are no blank cells, but there are unknown categories (for example, 8,597 defaults and 1,731 education values). The 12 exact duplicate rows are retained, since missing identifiers prevent reliable distinction between accidental duplication and distinct customers with identical recorded attributes. No rows are dropped. The 4,110 sentinel/history inconsistencies are preserved and documented rather than silently rewritten; the indicator is named `recency_recorded`, with `previous` and `poutcome` retaining the other history information.

### Validation experiments

Three feature recipes and three regularization strengths were tested. Raw is an explicit control that treats 999 numerically. After reviewing the validation evidence, I chose the engineered representation with the correct sentinel meaning, even though the raw control had slightly lower validation log loss. The best engineered configuration is selected by validation log loss. I do not claim that these engineered features improved the measured score over the raw control.

| Feature recipe | L2 lambda | Encoded columns | Validation log loss |
|---|---:|---:|---:|
'''
for e in experiments:
 text+=f'| {e["variant"]} | {e["lambda"]} | {e["features"]} | {float(e["validation_log_loss"]):.6f} |\n'
text+=f'''
Selected: **{m['variant']}**, **lambda={m['lambda']}**, **{m['n_features']} encoded features**. The selected validation log loss is 0.370993, versus 0.369414 for the raw control. The difference is approximately 0.00158; no statistical-significance claim is made. The selected semantic-versus-score tradeoff is recorded in `docs/DECISIONS.md`.

## Modeling: three implementations, one model

All three implementations are logistic regression on exactly the same transformed training matrix and labels. They minimize:

```text
z = X @ w + b
p = sigmoid(z)
J = mean(softplus(z) - y*z) + lambda/2 * sum(w**2)
```

The intercept is unpenalized and there are no class weights. Scikit-learn uses L-BFGS with `C = 1 / (n_train * lambda) = {m['sklearn_C']:.10f}`. The raw PyTorch version uses leaf tensors, autograd, explicit gradient updates under `torch.no_grad()`, and gradient clearing. The standard version uses `nn.Linear`, `BCEWithLogitsLoss`, an explicit L2 term and `torch.optim.SGD`.

The PyTorch versions both use zero initialization, float64 CPU tensors, full batches, and learning rate {m['learning_rate']:.8f}. The learning rate is computed as the inverse of a training-only conservative logistic-loss curvature bound. Both stop after 6,433 updates when the largest absolute gradient is below 1e-7. It is an optimization stopping rule, not test-based early stopping.

### Test comparison

The frozen validation-selected threshold is **{m['threshold']:.10f}**. Threshold selection groups tied scores, evaluates all attainable policies, favors fewer calls when costs tie, and puts the threshold between adjacent scores to avoid floating-point boundary differences after saving/loading.

| Method | ROC-AUC | Average precision | Log loss | Brier | Recall | Calls | Cost (EUR) |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for row in metrics:
 if row['split']=='test' and row['model'] not in ['call_all','call_none']:
  text+=f'| {row["model"]} | {float(row["roc_auc"]):.6f} | {float(row["average_precision"]):.6f} | {float(row["log_loss"]):.6f} | {float(row["brier"]):.6f} | {float(row["recall"]):.3%} | {int(row["calls"]):,} | {float(row["cost_eur"]):,.0f} |\n'
text+=f'''
The naive probability baseline predicts the training prevalence, 0.048074, for every record. At the frozen threshold it calls everyone. Explicit decision baselines cost **EUR 16,476 for call-all** and **EUR 127,000 for call-none** on test. Call-all/none are decision policies, not useful probability forecasts; their full metric rows are retained in the export but their clipped hard-probability log losses should not be used to assess probability quality.

The maximum probability difference from scikit-learn is **{m['agreement']['manual']['max_probability_difference_all_rows']:.8f}**, with **zero test decision disagreements** for either PyTorch implementation. The maximum coefficient difference is {m['agreement']['manual']['max_weight_difference']:.8f}. The small differences reflect numerical solvers and finite stopping tolerance; tiny rank-metric differences can arise from near ties. Objective-gradient and prediction-persistence tests independently check equivalence. Individual training timings are in metadata.json; they are not a controlled hardware benchmark.

### Student interpretation and recommendation

'''
text+=(ROOT/'docs/STUDENT_REFLECTION.txt').read_text()+'\n'
text+='''
The three implementations' agreement supports their computational consistency, but it does not establish business value. My feature decision prioritizes correct sentinel semantics, not a claim of predictive improvement. The final cost comparison uses 97 skipped customers; earlier exploratory figures used a boundary-valued threshold and were superseded by the stable midpoint threshold before delivery.

## Limitations & next steps

My priority is a temporal backtest with calibration checks on successive later campaign blocks, followed by real cost and revenue estimates before reconsidering the policy. Any recalibration must be fitted on an appropriate validation block and evaluated on a later untouched period, not optimized against this test block.

Specific limitations relevant to that investigation are the observed prevalence change (4.807% to 30.833%), limited early prior-contact experience, and absent exact timestamps/customer IDs. Training includes only zero or one previous contact, while later data includes larger counts. The file does not permit reliable customer-disjoint splits or clean campaign/day boundary separation. The historical sample is not representative evidence for today's bank customers in general.

No randomized no-call outcome or real business-cost data is present, so causal lift and actual profit remain unmeasured. Associations involving customer categories are not causal explanations. The fixed chronological test exposes these limits; it is not replaced by a more favorable random split.

## Dashboard and reproducibility

Run `uv sync --frozen`, then `uv run python main.py train` and `uv run python main.py app`. Open `http://127.0.0.1:7860`. The app loads the saved preprocessor, all three saved models and a saved naive baseline. It provides model comparison; calibration and record-level prediction-versus-actual plots; feature/target distributions by split; threshold-controlled confusion matrix and business cost; and real inference for a selected dataset row. Nothing is retrained at startup or through a control.

Run `uv run python -m pytest -q` for the checks. They cover input leakage, sentinel semantics, training-only preprocessing, gradients, objective equivalence, artifact reloads, threshold selection and app cost consistency. Local end-to-end and browser checks are documented in `docs/VERIFICATION.md`. The original GitHub Actions workflow is retained; no remote Actions run is claimed because nothing has been pushed.

## Generative AI use disclosure

I used Codex to inspect source documentation, implement and debug the Python pipeline and dashboard, write verification checks, and format measured evidence and this report. Through the project discussion I selected the bank-marketing topic, pre-call decision point, EUR 2/EUR 50 assumptions, chronological evaluation, prior-history and loan/unknown-field feature ideas, and the choice to retain correct sentinel semantics despite the small validation-score tradeoff.

The deployment recommendation and next-investigation proposal above are my supplied interpretation. Codex corrected its arithmetic to the final verified threshold and distinguished a proposed cause from an established result. Measured tables were formatted from actual output files. A separate, explicitly AI-generated PDF is a learning aid for understanding and explaining the implementation. This disclosure does not transfer the requirement to understand the work. No submission has been performed.
'''
# Preserve reviewed Walter wording when regenerating the evidence report.
revisions = json.loads((ROOT/'docs/walter_report_revisions.json').read_text())
for original, revised in revisions.items():
 text = text.replace(original, revised)
(ROOT/'REPORT.md').write_text(text)
print('REPORT.md updated from saved measurements and student reflection.')
