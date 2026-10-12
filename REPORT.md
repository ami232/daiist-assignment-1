# Assignment 1 Report

- **Name**: Javier Sánchez Esquivel
- **Student ID**: 19731
- **Email**: jsanchez.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I chose UCI's Bank Marketing dataset, specifically `bank-additional-full.csv`. It contains 41,188 client/campaign entries, 20 original inputs and a single binary target, `y` (term-deposit subscription). According to the supplied data dictionary, the entries are ordered by date from May 2008 through November 2010. Each entry contains customer characteristics and campaign-contact summaries, but exact timestamps and customer identifiers are not provided.

To evaluate later campaigns, I selected the entire ordered version of the file. At 5,834,924 bytes, it is practical to commit and retrain, although its 41,188 rows exceed the assignment's approximate 20k-row guideline. I left the CSV unaltered, with provenance described in `data/README.md` and its SHA-256 hash in `artifacts/metadata.json`. There is no runtime data download.

Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), Moro, Rita and Cortez, DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306), licensed CC BY 4.0. Publication: Moro, Cortez and Rita (2014), *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems 62, 22-31, DOI [10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).

## Business / real-life framing

The decision is who to prioritize before a scheduled phone call, using information available at that point. I assumed a cost of **EUR 2 per call**, regardless of success or failure, and **EUR 50 per missed subscriber**:

```text
policy cost = 2 * (TP + FP) + 50 * FN
```

These assumed costs do not represent measured bank profit. Historical subscription status is the variable being predicted; the data does not show whether an additional call would cause a customer to subscribe. This is predictive modeling, not an estimate of the incremental gain associated with an additional call.

A fixed time-based split uses the earliest 60% of observations for training, the next 20% for validation and the most recent 20% for testing. I chose this approach to evaluate future campaign conditions. All learned preprocessing operations are fitted on training data alone. Validation log loss compares a small number of model configurations, while validation policy cost determines the threshold. That threshold is frozen and used across all three models on test. Neither models nor thresholds are retuned using test performance.

| Block | Rows | Zero-based source row IDs | Subscriber rate |
|---|---:|---|---:|
| train | 24,712 | 0-24,711 | 4.807% |
| validation | 8,238 | 24,712-32,949 | 11.071% |
| test | 8,238 | 32,950-41,187 | 30.833% |

## Data preparation & feature engineering

My feature ideas include distinguishing previous-contact status and campaign history, since returning customers may respond differently, and testing loan combinations and incomplete customer records, since financial commitments and missing data may help distinguish response patterns.

The implementation includes age; job, marital status, education, default, housing and loan categories; and prior-campaign outcome. It excludes duration, completed-campaign contact count, and final-call channel/month/weekday because the file does not establish their availability before the planned call. I also excluded macroeconomic series because publication-time and vintage information are absent. Although this limits the available predictive signal, it makes the information boundary explicit. Customer profiles and prior-campaign history are assumed to exist in the CRM before a new call.

| Feature | Implementation |
|---|---|
| Recency recorded | Indicator for `pdays != 999`; 999 is a sentinel, not elapsed days. It does not reliably mean no historical contact, since 4,110 records combine it with `previous > 0`. |
| Log recency | `log1p(pdays)` when observed; otherwise zero, distinguished by the indicator. |
| Log previous contacts | `log1p(previous)`, compressing count differences. |
| Loan combination | Combined housing/loan category, including unknown values. |
| Unknown-field count | Count of `unknown` across the six customer categories. |

Numeric inputs are median-imputed defensively and standardized using training statistics. One-hot encoding uses the training vocabulary and retains `unknown`; unseen categories are handled without refitting. There are no blank fields, but some categories are unknown, including 8,597 default and 1,731 education values. The 12 exact duplicate rows are retained because, without identifiers, identical attributes cannot reliably distinguish accidental duplication from distinct customers. No rows were deleted. The 4,110 sentinel/history inconsistencies are documented and preserved. The `recency_recorded` indicator identifies a recorded day interval; `previous` and `poutcome` retain the other history information.

### Validation experiments

I evaluated three feature recipes and three regularization strengths. The raw representation serves as a control that treats 999 as an ordinary number. After reviewing the validation results, I chose the engineered representation with the appropriate sentinel meaning, even though the raw control produced slightly lower validation log loss. The best engineered configuration is selected using validation log loss. I am not claiming that the engineered features improve the measured score over the raw control.

| Feature recipe | L2 lambda | Encoded columns | Validation log loss |
|---|---:|---:|---:|
| raw | 0.001 | 39 | 0.369414 |
| raw | 0.01 | 39 | 0.370913 |
| history_context | 0.01 | 46 | 0.370993 |
| history | 0.01 | 40 | 0.371278 |
| history_context | 0.1 | 46 | 0.372910 |
| history | 0.1 | 40 | 0.373344 |
| history_context | 0.001 | 46 | 0.373842 |
| history | 0.001 | 40 | 0.373874 |
| raw | 0.1 | 39 | 0.374740 |

Selected: **history_context**, **lambda=0.01**, **46 encoded features**. The selected validation log loss is 0.370993, versus 0.369414 for the raw control. The difference is approximately 0.00158; no statistical-significance claim is made. The selected semantic-versus-score tradeoff is recorded in `docs/DECISIONS.md`.

## Modeling: three implementations, one model

All three logistic-regression implementations use exactly the same transformed training matrix and labels. They minimize:

```text
z = X @ w + b
p = sigmoid(z)
J = mean(softplus(z) - y*z) + lambda/2 * sum(w**2)
```

The intercept is not penalized, and no class weights are used. Scikit-learn uses L-BFGS with `C = 1 / (n_train * lambda) = 0.0040466170`. Manual PyTorch uses leaf tensors, autograd, explicit gradient updates under `torch.no_grad()` and gradient clearing. Standard PyTorch uses `nn.Linear`, `BCEWithLogitsLoss`, an explicit L2 penalty and `torch.optim.SGD`.

Both PyTorch versions use zero initialization, float64 CPU tensors, full batches and a learning rate of 0.79045220. This rate is the inverse of a conservative logistic-loss curvature bound computed from training data alone. Both stop after 6,433 updates, when the maximum absolute gradient is below 1e-7. This is an optimization stopping criterion, not test-based early stopping.

### Test comparison

The frozen validation-selected threshold is **0.0344274137**. Threshold selection groups tied scores, examines all attainable call policies and prefers fewer calls when costs are equal. It places the threshold between adjacent scores to avoid floating-point boundary issues when models are saved and loaded.

| Method | ROC-AUC | Average precision | Log loss | Brier | Recall | Calls | Cost (EUR) |
|---|---:|---:|---:|---:|---:|---:|---:|
| sklearn | 0.640933 | 0.485801 | 0.843136 | 0.257931 | 98.740% | 8,141 | 17,882 |
| manual | 0.640933 | 0.485804 | 0.843136 | 0.257931 | 98.740% | 8,141 | 17,882 |
| standard | 0.640933 | 0.485803 | 0.843136 | 0.257931 | 98.740% | 8,141 | 17,882 |
| baseline | 0.500000 | 0.308327 | 0.969856 | 0.280993 | 100.000% | 8,238 | 16,476 |

The naive probability baseline predicts the training prevalence, 0.048074, for each record. At the frozen threshold, it calls everyone. The explicit decision baselines cost **EUR 16,476 for call-all** and **EUR 127,000 for call-none** on test. Call-all and call-none are decision policies, not useful probability forecasts. Their full metric rows are included in the export, but their clipped hard-probability log losses should not be used to evaluate probability quality.

The maximum probability difference from scikit-learn is **0.00000561**, with no test decision disagreements for either PyTorch implementation. The maximum coefficient difference is 0.00001256. These small differences reflect numerical solvers and finite stopping tolerance; tiny ranking differences can arise when scores are nearly tied. Objective-gradient and prediction-persistence tests independently verify equivalence. Individual training times are provided in metadata.json; they are not a controlled hardware benchmark.

### Student interpretation and recommendation

I would not deploy this policy as is. Under the EUR 2/EUR 50 assumptions, skipping a customer only pays off if their subscription probability is below 4%. In the final verified run, the 97 customers the model skipped included 32 subscribers (about 33%), so the policy saved EUR 194 in call costs and incurred EUR 1,600 in assumed missed-subscriber costs, ending EUR 1,406 worse than calling everyone.

My proposed explanation is that the model was trained on a population with a 4.8% subscription rate and then tested on one with 30.8%, leaving its probabilities too pessimistic for the later population. This is a hypothesis to investigate, not an established causal explanation.

The most useful next step is a temporal backtest with a calibration check on successive later campaign blocks, to see whether the shift is persistent and whether recalibrating on recent data fixes it. Real cost and revenue estimates should then replace the hypothetical EUR 2/EUR 50 before the policy is reconsidered.

Consistency among the three implementations supports their computational equivalence, but does not establish business value. My feature decision prioritizes correct sentinel semantics rather than claiming predictive improvement. The final cost comparison uses 97 skipped customers. Earlier exploratory figures used a threshold directly on a score boundary; the stable midpoint threshold superseded those figures before delivery.

## Limitations & next steps

My next step is a temporal backtest with calibration checks across successive later campaign blocks, followed by real cost and revenue estimates before reconsidering the policy. Any recalibration must be fitted on an appropriate validation block and tested on a later untouched period, rather than optimized against this test block.

Relevant limitations include the observed prevalence change from 4.807% to 30.833%, limited early prior-contact history, and missing exact timestamps and customer IDs. The training set contains only zero or one previous contact, while later records contain larger counts. The file does not allow reliable customer-disjoint splits or clean campaign/day boundaries. This historical sample is not representative evidence for bank customers today.

There are no randomized no-call outcomes or real business-cost data, so causal lift and actual profit remain unmeasured. Associations involving customer categories are not causal explanations. The fixed chronological test reveals these constraints; it has not been replaced by a more favorable random split.

## Dashboard and reproducibility

Run `uv sync --frozen`, then `uv run python main.py train` and `uv run python main.py app`. Open `http://127.0.0.1:7860`. The application loads the saved preprocessor, all three saved models and a saved naive baseline. It provides model comparison, calibration and record-level prediction-versus-actual plots, feature/target distributions by split, a threshold-controlled confusion matrix, business costs and real inference for a selected row. None of these controls retrains a model, and there is no training at startup.

Run `uv run python -m pytest -q` for the checks. They cover input leakage, sentinel semantics, training-only preprocessing, gradients, objective equivalence, artifact reloads, threshold selection and dashboard cost consistency. Local end-to-end and browser checks are documented in `docs/VERIFICATION.md`. The original GitHub Actions workflow is retained.

## Generative AI use disclosure

I used Codex to examine source documentation, implement and debug the Python pipeline and dashboard, write verification checks, and format the measured evidence and report. During the project discussion, I chose the bank-marketing topic, pre-call decision point, EUR 2/EUR 50 assumptions, chronological evaluation, and the prior-history and loan/unknown-field feature ideas. I also chose correct sentinel semantics despite the small increase in validation log loss.

The deployment recommendation and proposed next investigation are my supplied interpretation. Codex corrected the arithmetic to match the final verified threshold and distinguished a possible cause from an established result. Measured tables were formatted directly from output files. A separate AI-generated PDF helps explain the implementation. Walter Writes revised explanatory prose, and Codex reviewed and corrected its output for factual accuracy. The student's original recommendation was preserved. This assistance does not remove my responsibility to understand the work. Blackboard submission remains for me to complete.