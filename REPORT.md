# Assignment 1 Report

- **Name**: Wilrick Dong
- **Group**: BBADBA 5B

---

## Dataset

**What it is.** The CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2022, distributed on
Kaggle as *Personal Key Indicators of Heart Disease*
([kamilpytlak/personal-key-indicators-of-heart-disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)).
BRFSS is the largest continuously running health survey in the world, completing more than
400,000 adult telephone interviews a year.

**What one row is.** One adult who answered the 2022 telephone survey. Every column is that
person's own answer — self-reported health conditions, behaviours, demographics and access to
care. There are no clinical measurements and no dates.

**Size.** The full file is 445,132 rows × 40 columns. This project uses the committed
`data/heart_2022_sample.csv`: a **20,000-row stratified sample** that preserves the outcome rate
exactly, small enough that the whole pipeline trains on CPU in under a minute. Split 60/20/20 it
gives 12,000 training, 4,000 validation and 4,000 test rows.

**Target.** `HadHeartAttack` — "has a doctor ever told you that you had a heart attack?" —
recoded to 1 for Yes. **5.68%** of rows are positive, identical across all three splits because
the split is stratified. Rows with a missing target are dropped.

**Why this dataset.** It matches the business problem below at the level of the individual
respondent, it is large enough that a 20,000-row sample still contains ~1,136 positives, and its
severe class imbalance makes it a realistic test of why accuracy is the wrong metric.

---

## Business / real-life framing

### The organisation

**Corvanta Cardio-Oncology Institute is a fictional organisation, invented for this assignment.**

In the scenario, Corvanta is the country's leading referral centre for patients who have both
cancer and cardiovascular disease, operating an outreach programme across its own catchment
region. The specialty itself is real and growing: anthracyclines, trastuzumab and chest
radiotherapy are all cardiotoxic, and cancer survivors carry elevated cardiovascular risk for
years afterwards, which is why dedicated cardio-oncology clinics now exist to risk-stratify these
patients before and during treatment.

Corvanta's clinical excellence is not the constraint. Its **referral pipeline** is. Patients who
reach a cardio-oncology service only after an acute cardiac event arrive sicker, cost more to
treat, and have fewer treatment options than patients identified before the event.

### The decision the model supports

Corvanta runs a **community outreach screening programme** across its catchment region. It
fields a health questionnaire — the BRFSS instrument — to adults in the region, scores each
respondent, and invites the high scorers for a cardiac workup (nurse consultation, ECG, and
onward referral where indicated).

The programme's budget allows a limited number of invitations. The model's only job is to answer:
**given this questionnaire, should we invite this person for a workup?**

This framing is what makes the dataset appropriate. The unit of decision is a survey respondent,
which is exactly what a BRFSS row is. A model trained on survey answers is being asked to score
survey answers.

### What the framing implies for the pipeline

**Target definition.** The outcome is the survey's own `HadHeartAttack` field. Section *Limitations*
is explicit that this is a *lifetime history* question, not an incident-risk one, and what that
costs the interpretation.

**No time-based split.** A time-based split guards against training on the future and testing on
the past. That risk does not exist here: the data is a single 2022 cross-section with no date
column, and every row is a different person, so there is no temporal ordering to respect and no
future to leak. The split is random and **stratified** on the target, which keeps the 5.68%
positive rate identical in all three parts — important when the test set contains only 227
positives. A stratified split was chosen over a plain random one for this reason alone.

**Which metric drives the threshold.** Accuracy is actively misleading at 5.68% prevalence: a
model that flags nobody scores **94.3% accuracy while catching zero heart attacks** (notebook 01,
section 2). Models are therefore compared on **PR-AUC**, which is threshold-free and focuses on
the positive class. But PR-AUC does not pick an operating point, and the programme has to decide
whom to invite. That decision comes from the two costs below.

### The two costs

| | Figure | What it represents |
|---|---|---|
| `COST_FP` | **\$300** | One unnecessary invitation: outreach contact, nurse consultation and baseline ECG for someone who turns out not to be high risk. |
| `COST_FN` | **\$2,436** | The programme's expected loss per high-risk adult it fails to invite: the share of avoidable downstream acute-care spend the programme is accountable for, plus the lost contribution margin from a patient who later presents in crisis elsewhere. |

A missed case therefore costs **8.1×** a false alarm. Two things about these numbers:

- **Only the ratio moves the threshold.** The absolute dollars scale the reported cost column but
  do not change a single decision. The cost-minimising cutoff is
  `t* = COST_FP / (COST_FP + COST_FN) = 300 / 2,736 = 0.110`.
- **\$2,436 is a programme-level figure, not the cost of a heart attack.** An acute MI admission
  costs far more. The figure is the slice of that cost the outreach programme's budget actually
  carries. Treating it as the full clinical cost would be wrong, and would push the ratio much
  higher and the threshold much lower.

**Why the threshold is chosen on validation, not by formula.** `train.py` sweeps thresholds from
0.01 to 0.99, computes total cost on the **validation** set, and keeps the cheapest. The formula
above is only correct if the model's probabilities are perfectly calibrated, so the gap between
the two is a calibration check rather than an assumption. The empirical answer is **0.11** against
a theoretical **0.110** — they agree, which means the probabilities can be taken at face value.
Choosing the threshold on the test set would mean reporting a number tuned to the data being
reported.

**A consequence worth stating plainly.** Recall is an *output* of the cost ratio, not a target.
Holding the baseline model fixed and changing only the costs: at the template's placeholder 10:1
the threshold is 0.09 and recall 61.7%, while at the programme's real 8.1:1 a false alarm is
relatively more expensive, the bar rises to 0.12, and recall falls to 51.1% as precision climbs
from 27.8% to 34.0%. Nothing about the fitted model changed — only the cutoff. (The final model
lands at threshold 0.11 with recall 56.4%, the improvement coming from the features, not the
costs.) Tuning a model to hit a recall figure chosen in advance would be optimising for a number
nobody pays, instead of for the cost the programme actually bears.

---

## Data preparation & feature engineering

### Preprocessing (identical for baseline and final model)

Fitted **on the training split only**, inside an sklearn `Pipeline`, so no statistic from
validation or test can reach the model:

1. **Impute** — numeric columns with the median, categorical with the most frequent value.
2. **One-hot encode** categoricals with `drop="first"`, avoiding the dummy-variable trap.
3. **Standardise** every column, one-hot columns included. This is what makes the problem
   well-conditioned enough for the two gradient-descent implementations to reach the same answer
   as scikit-learn, and it makes coefficients comparable as "change in log-odds per 1 SD".

### The rule every engineered feature obeys

Each feature is computed **row by row**, from that person's own answers plus hand-written constant
tables (`"Poor" → 0`). None uses a statistic computed from the dataset. Target encoding — replacing
each state with its observed heart attack rate — was considered and **rejected** for this reason:
it would have to be fitted inside the pipeline on training rows only, and as a row-by-row
transform it would leak.

### Method: why a single validation split was not trusted

Every idea was first measured on one split, then re-measured across repeated reshuffles. The two
disagree often enough that the single split alone would have selected a worse set:

- `male_x_age` scored **+0.0000** on one split and won **10 of 10** reshuffles.
- `teeth_rank` and `smoking_rank` looked *negative* in single-split leave-one-out and are both in
  the final set.

The final set was therefore chosen by a search scoring candidate sets on **mean** validation
PR-AUC over several reshuffles, then re-checked on **20 fresh shuffles (seeds 200–219) that played
no part in the selection**. The test set was never touched during any of this.

### The ten features kept

| Feature | What it does | EDA evidence (notebook 01) |
|---|---|---|
| `health_rank` | GeneralHealth → 0–4 | §5: each step roughly doubles the rate — a straight line in log-odds |
| `age_rank` | AgeCategory → 0–12, replacing 12 one-hot columns | §5: steady climb; youngest bands hold only 3–11 positives each |
| `male_x_age` | (age − 50) for men, 0 for women | §9: after 45 the men's log-odds line is **not parallel** to the women's |
| `checkup_recent` | LastCheckupTime → "within the past year?" | §5: a *jump*, not a trend — an evenly spaced rank would misdescribe it |
| `smoking_rank` | SmokerStatus → 0–3 | §8: never < former < current holds once age is fixed |
| `teeth_rank` | RemovedTeeth → 0–3 | §5: clean climb 2.6% → 6.2% → 12.7% → 18.0%, survives the age check |
| `difficulties_count` | Six disability columns → one count | §11: the six point the same way; more difficulties means more events in every age band |
| `region` | State (53 columns) → 5 Census regions | §12: South 1.16× expected, but a typical state holds only ~15 positives |
| `drop_height_weight` | Keep BMI, drop height and weight | §10: BMI = weight/height² (r = 0.9999); BMI–weight r = 0.86 |
| `blank_as_category` | A blank becomes its own level, `"Not answered"` | §4: 126 people left HadAngina blank and **36.5%** of them had a heart attack, vs 5.7% overall |

Two of these deserve a note.

**`blank_as_category` is the single strongest idea (+0.0185 alone).** The baseline fills blanks
with the most common answer, which *destroys* the signal in section §4. The label matters: the
one-hot encoder drops the alphabetically first category, so `"(no answer)"` would have sorted
first and silently changed the reference level of every Yes/No column. `"Not answered"` sorts
after `"No"`, leaving the baseline's reference intact.

**Replacing beats adding.** `difficulties_count` replaces its six columns and gains +0.0024 (8/10
shuffles); the otherwise identical `difficulties_count_added`, which *keeps* the six, is **negative**
(3/10). This is the expected result for a linear model — a sum of yes/no columns is something the
model can already construct from those columns — and it was tested explicitly rather than assumed.

### What was tested and rejected

| Rejected | Why |
|---|---|
| `n_blank_answers` | A per-question blank flag (`blank_as_category`) already captures it, with more detail |
| `n_cardiometabolic` | The cleanest single-split plot in the EDA, yet −0.0008 across shuffles (3/10) |
| `n_chronic` | Worst performer: −0.0016, **0 of 10** shuffles |
| `care_score` | Mixed signal, possibly a consequence of the event; also caused an optimisation problem (below) |
| `ecig_rank`, `diabetes_x_age`, `sleep_short_long` | Neutral-to-negative once the others were in |
| `age_years`, `checkup_rank`, `checkup_years`, `ever_smoked`, `stroke_belt` | Lost to a rival in the same family |
| `bmi_recompute` | **Borrowed** from a public Kaggle notebook (cited in notebook 03 §2e): recompute missing BMI from the person's own height and weight. Worth **+0.0000**. Only **90** of 1,730 blank-BMI rows are recoverable in a 20,000-row sample, against 2,223 in the full 445,132-row file. An idea that genuinely helps at full scale can be worth nothing on a sample. |

### Leakage: the HadAngina decision

Notebook 01 flagged three columns that might be a medical paper trail rather than a predictor.
Notebook 03 §8 measures what dropping each would cost, on validation:

| Scenario | Baseline PR-AUC | Final model PR-AUC |
|---|---|---|
| **Keep everything** (chosen) | 0.4315 | **0.4571** |
| Drop ChestScan | 0.4300 | 0.4540 |
| Drop HadAngina | 0.2677 | 0.2785 |
| Drop both | 0.2675 | 0.2763 |

**Decision: both are kept, and `DROP_COLUMNS` is empty.** The reasoning is the availability test —
*would the intended user know this at decision time?* In the outreach framing the respondent is
filling in a questionnaire, and *"has a doctor ever told you that you had angina?"* is a question
they can answer at that moment. Angina is chest pain caused by coronary artery disease, which is
the same disease that causes heart attacks: it is a genuine clinical risk factor, not a record
generated by the outcome. The same argument covers ChestScan, which costs almost nothing either way.

This decision is reported rather than hidden because it is **worth 38% of the model's PR-AUC** —
by far the largest single lever in the project, larger than all ten engineered features combined.
A reader who disagrees with the availability argument can read the exact cost of the alternative
straight off the table. It also changes which features matter: most of `blank_as_category`'s gain
comes from the HadAngina blanks specifically (+0.0185 with it, +0.0038 without).

### Result

| | Baseline | Final |
|---|---|---|
| Encoded columns | 121 | **66** |
| Validation PR-AUC | 0.4315 | **0.4571** (+0.0255) |

Across the 20 held-out shuffles the final set beat the baseline by **+0.0240 on average, winning
20 of 20**. The gain is robust, and it comes with *fewer* columns, not more.

One honest note on the margin: a 13-idea alternative scored **+0.0003** higher in search, with a
standard deviation of 0.0013 and winning 10 of 20 — a coin flip, not an improvement. On the
reported split the ten-feature set is in fact better (0.4571 vs 0.4559). The smaller set was kept
because it is indistinguishable in accuracy and simpler to defend.

---

## Modeling: three implementations, one model

### The model, and why

**Regularised logistic regression.** The target is binary, so linear regression is inapplicable.
Logistic regression also produces **calibrated probabilities**, which this use case requires: the
threshold is derived from a cost ratio, and that derivation is only valid if a predicted 0.11
really means an 11% chance. The calibration check below confirms it holds. A tree ensemble would
likely score higher on raw discrimination but would not satisfy the assignment's three-implementation
requirement, and would need separate calibration before the cost arithmetic could be trusted.

**Regularisation.** `C` is swept over `[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]` on validation.
The grid was extended downwards after an earlier run selected the smallest value on the original
grid — a winner at the edge means the grid never looked where the optimum was. The chosen value is
**C = 0.001**, giving `λ = 1/(C·n) = 0.0833`.

### All three minimise the same objective

mean binary cross-entropy + (λ/2)·‖w‖², with the bias **not** penalised.

scikit-learn writes it as `0.5·‖w‖² + C·Σ(errors)`. Dividing by `C·n` gives the form above with
**λ = 1/(C·n)**. Getting this conversion right is the whole reason the three agree:

- **scikit-learn** — `LogisticRegression(C=0.001, max_iter=10_000, tol=1e-10)`, lbfgs.
- **Manual PyTorch loop** — raw tensors, `loss.backward()` for gradients, weights updated by hand
  under `torch.no_grad()`, gradients zeroed each step. 3,000 epochs, full batch, LR 0.5.
- **PyTorch `nn.Module`** — one `nn.Linear`, `BCEWithLogitsLoss`, `torch.optim.SGD`. The penalty
  enters as `weight_decay=λ` on the weight group only (`weight_decay=0.0` for the bias), because
  SGD's weight decay adds `λ·w` to the gradient, which is exactly the gradient of `(λ/2)·‖w‖²`.

Both PyTorch versions start from zeros so any difference is the API, not initialisation, and
everything runs in `float64` so that any remaining gap reflects optimisation rather than float32
rounding.

### Test-set results

Thresholds chosen on validation; scores reported on the test set, used once.

| Model | ROC-AUC | PR-AUC | Log loss | Threshold | Precision | Recall | Cost / 1,000 |
|---|---|---|---|---|---|---|---|
| Flag nobody | 0.5000 | 0.0568 | 0.2179 | — | 0.0000 | 0.0000 | \$138,243 |
| Flag everyone | 0.5000 | 0.0568 | 0.2179 | — | 0.0568 | 1.0000 | \$282,975 |
| **scikit-learn** | **0.8762** | **0.3842** | **0.1608** | 0.11 | 0.3325 | 0.5639 | **\$79,566** |
| **Manual PyTorch loop** | 0.8762 | 0.3842 | 0.1608 | 0.11 | 0.3325 | 0.5639 | \$79,566 |
| **PyTorch nn.Module** | 0.8762 | 0.3842 | 0.1608 | 0.11 | 0.3325 | 0.5639 | \$79,566 |

- **PR-AUC 0.3842 against a 0.0568 random baseline — 6.8× better than chance.**
- **Cost per 1,000 screened: \$79,566**, against \$138,243 for doing nothing — a **42% reduction**,
  about **\$58,677 saved per 1,000 people screened**. Flagging everyone costs more than twice as
  much as doing nothing, which is the quantitative case for having a model at all.
- The model flags **9.6%** of people, catching 56.4% of cases at 33.3% precision — roughly one in
  three invitations reaching someone who really is high risk, against one in eighteen if invitations
  were sent at random.
- Test PR-AUC (0.3842) sits below validation (0.4571). That is expected: C, the feature set and the
  threshold were all chosen on validation, so validation is optimistic. With only 227 positives in
  the test set either figure can move by a few hundredths through luck alone. This is precisely why
  the test set is used once.

### Do the three agree? Yes

| Comparison | Max coefficient gap | Max probability gap |
|---|---|---|
| scikit-learn vs manual loop | 4.23e-08 | 7.95e-08 |
| scikit-learn vs `nn.Module` | 4.23e-08 | 7.95e-08 |
| manual loop vs `nn.Module` | 5.55e-17 | 2.22e-16 |

Final training objective: **0.16839841081362136** (scikit-learn) against **0.16839841081362072**
(both PyTorch versions) — agreement to 15 significant figures. The two PyTorch implementations
match each other at machine precision, which is the expected result since they run the same
algorithm from the same start; the residual ~1e-08 against scikit-learn is the gap between
gradient descent and lbfgs, not a disagreement about the answer.

**They agree because gradient descent converged — and it was verified, not assumed.** Notebook 02
§7 shows what non-convergence looks like: at 10 epochs the largest coefficient is off by 0.16, at
50 by 0.04, at 200 by 0.003, and from 1,000 epochs it matches to 1e-7. At LR 0.01 even 3,000
epochs leave a gap of 0.036. What matters is roughly *learning rate × epochs*.

### An optimisation problem that changed a feature decision

An intermediate 17-feature set did **not** converge. The wider feature set made `train.py` select
a weaker penalty (C = 0.1), and the coefficient gap grew to **0.009**. The cause was
near-duplicate columns:

- `care_score` is built from the same answers as the FluVax, PneumoVax, tetanus and HIV columns.
- `CovidPos = "Not answered"` and `HighRiskLastYear = "Not answered"` were left blank by almost
  exactly the same people — **2,221 of 2,279**.

Near-duplicate columns create a long flat valley in the loss: many different coefficient
combinations give nearly the same value, and gradient descent crawls along it. The weaker the
penalty, the flatter the valley. Removing `care_score` alone moved the selected C to 0.01 and the
gap back to ~5e-7.

The final set resolves this at the root rather than by raising the epoch count: it excludes
`care_score`, and with 66 columns the sweep selects the strongly regularised C = 0.001, where the
problem is well-conditioned. Convergence was re-verified at 3,000, 6,000 and 10,000 epochs —
**4.23e-08, unchanged** — confirming 3,000 is sufficient rather than merely adequate.

### Which features matter

Because every column is standardised, coefficients are directly comparable: each is the change in
log-odds per one standard deviation. The final model's largest, in order:

| Feature | Coefficient |
|---|---|
| `HadAngina_Yes` | +0.363 |
| `male_x_age` | +0.161 |
| `HadStroke_Yes` | +0.158 |
| `age_rank` | +0.146 |
| `teeth_rank` | +0.120 |
| `ChestScan_Yes` | +0.116 |
| `HadAngina_Not answered` | +0.113 |
| `Sex_Male` | +0.086 |
| `health_rank` | **−0.111** (higher self-rated health lowers risk — the expected sign) |

Three things worth reading off this table:

- **`HadAngina_Yes` dominates at 2.3× the next feature**, which is the quantitative form of the
  leakage concern discussed above.
- **Four of the top seven are engineered features** — `male_x_age`, `age_rank`, `teeth_rank` and
  `HadAngina_Not answered`. The feature work is not decoration; it is carrying the model.
- **`HadAngina_Not answered` is itself in the top seven.** That single column is `blank_as_category`
  earning its keep: the 126 people who skipped the angina question, 36.5% of whom had a heart
  attack, get their own coefficient instead of being silently recoded as "No". It also explains why
  `blank_as_category`'s gain collapses from +0.0185 to +0.0038 when HadAngina is dropped — the two
  decisions are entangled, and the sensitivity table above is the honest way to present them.

---

## Limitations & next steps

**1. The target is lifetime history, not incident risk.** This is the most important limitation.
`HadHeartAttack` asks whether a doctor has *ever* said so, and every predictor is measured at the
same moment. The model therefore identifies people who **look like** someone with a cardiac
history, not people who will have an event next year. For the outreach programme that is still
useful — these people genuinely need cardiology contact — but it is not a risk-prediction model and
must not be described as one. *Next step:* a longitudinal source with baseline exposures and
follow-up events (a cohort study, or linked claims with an index date and a defined outcome window)
would let the target be redefined as *incident MI within N years*. No amount of feature work on a
cross-section can fix this.

**2. Survey weights are ignored.** BRFSS uses a complex stratified design with post-stratification
weights, and the analysis treats rows as a simple random sample. Unweighted estimates are not
nationally representative, so the absolute heart attack rate here is not the population rate.
*Next step:* refit with the supplied `_LLCPWT` weights via `sample_weight`, and compare coefficients
against the unweighted fit to quantify the distortion.

**3. The model is trained on US adults nationally, applied to one catchment region.** The outreach
programme operates in a specific area; `region` is in the model, but a nationally fitted model may
not transfer to one region's demographic mix. *Next step:* refit on Corvanta's own region and
compare, or add a region-specific intercept.

**4. Self-report only, no clinical measurements.** There is no blood pressure, cholesterol,
troponin or ECG in the data — only what someone said on the phone, including a self-reported height
and weight behind BMI. The model is a *triage* instrument for deciding who gets measured, and
cannot substitute for measurement. *Next step:* once the programme has run, join invitation outcomes
back and refit on respondents whose workup results are known.

**5. The costs are planning assumptions.** \$2,436 and \$300 are the programme's figures, not
measured ones, and the threshold follows directly from their ratio. *Next step:* the cost curve is
flat near its minimum, so moderate error is tolerable — but a sensitivity analysis over a range of
ratios (say 4:1 to 20:1) would show how far the operating point moves before the decision changes.

**6. Only 20,000 of 445,132 rows are used.** The sample keeps the pipeline fast, at the cost of
227 test positives, which is why differences under ~0.002 PR-AUC were treated as noise throughout.
It also made one borrowed feature (`bmi_recompute`) worthless that is genuinely useful at full
scale. *Next step:* refit on the full file; features rejected here for lack of support are worth
retesting, as is the C grid.

**7. The leakage decision is arguable.** Keeping HadAngina is defended on availability grounds, but
a reader who regards angina as part of the same diagnostic episode would drop it — and PR-AUC would
fall from 0.4571 to 0.2785. The measurement is reported so that the disagreement can be had with
numbers rather than opinions. *Next step:* if the programme's questionnaire were ever shortened to
exclude the angina item, the model would have to be refitted, not merely rescored.

**8. No fairness audit.** `region`, `Sex` and `RaceEthnicityCategory` are all in the model, and it
allocates a scarce benefit — an invitation. Performance has not been examined by subgroup.
*Next step:* report PR-AUC, recall and flag rate per demographic group before any deployment, since
equal overall accuracy can still mean unequal access.

---

## Generative AI use disclosure

I used Claude (Anthropic), through Claude Code, for both the code and helping me rewrite the report in better way than my own words.

I used it first for planning: I set out the direction and the features I wanted to
engineer, and Claude turned that into a concrete plan. It then built the testing
pipeline and wrote the supporting code, including the exploratory data analysis
notebooks.

With the plan agreed, Claude implemented it and ran the tests. I read the results,
decided on that basis which changes I wanted, and asked Claude to make those
changes and push the work to GitHub.

The direction, the feature choices and the decisions taken after each round of
testing are mine. 



