# Assignment 1 Report

- **Name**: Rubén Segura
- **Student ID**: 19061
- **Email**: rsegura.ieu2022@student.ie.edu
- **Group**: [BBADBA 5B]

## Dataset

I'm using the Online Shoppers Purchasing Intention dataset from the UCI Machine
Learning Repository (dataset 468, donated in 2018 by Sakar et al.). It's a single
1 MB CSV, committed to `data/` so training never depends on UCI being up.

Each row is one browsing session on an e-commerce site, and every session belongs
to a different user over a one-year period, so no single campaign or heavy user
dominates. There are 12,330 rows and 18 columns: 17 features plus the binary
target `Revenue`, which says whether the session ended in a purchase. No missing
values. Only 1,908 sessions (15.5%) converted.

I picked it because it's a real decision a shop would care about, the features
leave room for actual engineering instead of dropping raw columns into a model,
and the class imbalance means the decision threshold genuinely matters, which is
what I wanted to build the dashboard around.

## Business / real-life framing

**The scenario.** An online store scoring sessions live. About 100 seconds into a
visit, the model estimates how likely that session is to end in a purchase, and
if the probability falls below my threshold, the site offers a 10% discount
coupon. My assumptions: €80 average basket, 35% gross margin (€28 per
conversion), an €8 coupon, and a 5% chance the coupon recovers someone who wasn't
going to buy.

**What the errors cost.** The coupon fires on sessions predicted *not* to convert,
which flips the usual intuition about which mistake hurts:

| Case | Action | Impact |
|---|---|---|
| Buys, predicted buy | none | €0 |
| Buys, predicted no-buy | coupon | −€8 (discount on a sale I already had) |
| No buy, predicted no-buy | coupon | +€1 (5% × €20 net margin) |
| No buy, predicted buy | none | €0 (missed opportunity) |

Net value = (non-buyers correctly spotted × €1) − (buyers I mislabelled × €8).

Three things follow from this. First, I can only use features the site knows
mid-session, which makes `PageValues` suspicious: it's built from pages seen
*before a completed transaction*, so it carries the outcome. Second, the split
has to be time-based, because the model would be trained on past traffic and run
on future traffic. Third, accuracy is useless (always saying "no purchase" is
right 84.5% of the time), so I pick the threshold by maximising net value, and
with an 8-to-1 asymmetry the optimum is cautious: only intervene when the
evidence of no purchase is strong.


## Data preparation & feature engineering

I built four things, and the reasoning for each matters more to me than the count.

**1. Browsing intensity (3 features).** The data gives me a count *and* a total
duration for each page type (administrative, informational, product related).
Separately, neither tells you much: 300 seconds could be one slow page or ten
fast ones. Dividing duration by count gives seconds per page, which separates
someone reading carefully from someone clicking through. Both numbers exist at
100 seconds into the session, so this is legal in my scenario.

**2. Session composition (2 features).** `total_pages` is everything the user
looked at, and `product_page_share` is the fraction of that which was product
pages. A session that's 90% product pages is shopping; one that's mostly
informational pages is probably support or research. One number, easy to explain,
and easy to read off a logistic regression coefficient.

**3. Log transforms.** The counts and durations have very long tails — a handful
of sessions sit far above everything else. Logistic regression fits a straight
line in the features, so those extreme values pull the coefficients around more
than they should. I applied `log1p` to all durations, counts and the intensity
ratios, which compresses the tail without throwing data away.

**4. The PageValues ablation.** Instead of guessing whether it leaks, I trained
everything twice, with and without it, and reported both (see below).

Two smaller decisions. `OperatingSystems`, `Browser`, `Region` and `TrafficType`
arrive as integers, but they're IDs, not quantities — browser 4 isn't twice
browser 2 — so I cast them to strings and one-hot encoded them. And I dropped
`Month` as a feature: since I split on it, the test months would be categories
the model had never seen, which is useless at best and misleading at worst.

Preprocessing is a `ColumnTransformer` (standard scaling for numeric, one-hot for
categorical) **fitted on the training set only** and then applied to validation
and test, so no information from future months leaks backwards. After encoding,
the models see 70 inputs.

**The split.** Train on February to September (7,056 sessions, 11.6% conversion),
use October as validation (549 sessions, 21.0%), test on November and December
(4,725 sessions, 20.7%). Validation exists for one job only: choosing the
decision threshold. If I picked the threshold on the test set I'd be reporting a
number I had already tuned to, which would be optimistic — when I did exactly
that by mistake earlier, the net value came out at around €1,220 instead of the
€703 below.

## Modeling: three implementations, one model

Logistic regression, because the target is binary and because a linear model's
coefficients are something I can actually interpret and defend.

All three see the identical matrix and the identical split. The scikit-learn one
is `LogisticRegression(C=1.0)`. The manual PyTorch loop holds raw `W` and `b`
tensors, computes the logits, writes binary cross-entropy out by hand in its
numerically stable form, calls `backward()`, and updates the parameters inside
`torch.no_grad()` with plain gradient descent — 2,000 full-batch steps at a
learning rate of 0.5. The standard version is an `nn.Module` wrapping
`nn.Linear`, trained with Adam and `BCEWithLogitsLoss` over 60 epochs of
mini-batches. I set the PyTorch weight decay to `1 / (C × n)` so the L2 penalty
is comparable to sklearn's.

Results on November–December, with the threshold chosen on October:

| Model | AUC | Avg. precision | Threshold | Net value |
|---|---|---|---|---|
| scikit-learn | 0.8409 | 0.6174 | 0.10 | €703 |
| Manual PyTorch | 0.8413 | 0.6170 | 0.10 | €717 |
| Standard PyTorch | 0.8413 | 0.6145 | 0.13 | €685 |
| Baseline: coupon to everyone | 0.5 | 0.2066 | — | −€4,059 |

AUC is the chance the model ranks a random buyer above a random non-buyer, and
average precision summarises how well it does on the rare class; the baseline
sits at 0.5 and at the 20.7% conversion rate of the test period by definition.

The three agree to the fourth decimal, which is what should happen: same model,
same data, so the only differences are optimisation details. The gap in net value
(€685 vs €717) comes almost entirely from the standard version landing on a
threshold of 0.13 rather than 0.10, which tells me the threshold choice matters
more to the business than which optimiser I used. All three beat the no-model
baseline by a wide margin, mostly because blanket couponing hands €8 to every
single buyer.

Same thing with `PageValues` removed:

| Model | AUC | Avg. precision | Net value |
|---|---|---|---|
| scikit-learn | 0.7191 | 0.3592 | €599 |
| Manual PyTorch | 0.7200 | 0.3583 | €437 |
| Standard PyTorch | 0.7085 | 0.3491 | €433 |

One column is worth 0.12 of AUC and nearly half the average precision, which
confirms the leak I suspected in the framing: `PageValues` is close to reading
the answer. Interestingly the three implementations also drift apart here
(0.7085 vs 0.7200) — when the signal is weak the optimisation differences stop
cancelling out. In my live-scoring scenario the honest expectation is the 0.72
model, not the 0.84 one.

## Limitations & next steps

**Training data doesn't match deployment.** Every row is a finished session, but I
would score at 100 seconds. Even my "legal" features are end-of-session totals.
Fixing this properly means event-level logs and rebuilding each session truncated
at the scoring moment; with this dataset I can't, so every number here is an
upper bound.

**PageValues.** Keeping it inflates the result, dropping it loses signal that
might partly be legitimate. The next step is getting the raw page-value table as
it stood *before* the session, which is computable in a real shop.

**The validation month is small.** October has 549 sessions and roughly 115
buyers, so the threshold is estimated on very little data and is noisy. I'd move
to rolling-origin validation across several months and average the chosen
threshold.

**The economics are assumptions, not measurements.** The 5% recovery rate is the
weakest link: net value scales almost linearly with it, and at 0% the whole
programme loses money. Only an A/B test with a holdout group would settle it.

**The dataset converts at 15.5%**, while real e-commerce is usually 1–3%. Every
euro figure here is therefore inflated, and the optimal threshold would shift if
buyers were genuinely rarer.

**The test period is unusual.** November and December convert at 20.7% against
11.6% in training. That's the point of a time split, but it also means I'm
reporting on campaign months, not a typical one.


## Generative AI use disclosure

I used Claude (Anthropic) as an assistant on this assignment, within the limits
the brief allows.

Code: AI wrote most of the boilerplate — data loading, the preprocessing
`ColumnTransformer`, the scaffolding of the three training implementations,
artifact saving, the exploratory plots and the Gradio dashboard. I ran, checked
and adjusted all of it.

Decisions: the dataset, the live-scoring scenario, the cost assumptions (100-second
trigger, 10% coupon, 35% margin, 5% recovery rate) and which features to engineer
were mine, taken after discussing the options and trade-offs.

Writing: I used AI to help draft and tidy this report. The results, the
interpretation of them and the conclusions come from my own.