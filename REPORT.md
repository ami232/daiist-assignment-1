# Assignment 1 Report

- **Name**: Jesús López Cubas
- **Student ID**: 20414
- **Email**: jlopez.ieu2022@student.ie.edu
- **Group**: [BBADBA 5B]

---

**Predicting when a volatility-targeting portfolio will be under-risked: one logistic regression, three implementations**

**Abstract.** Multi-manager hedge funds, the so-called pod shops, earn modest returns with very low volatility and lever them to attractive levels, which makes misjudged volatility and tail events their central risk. Volatility-targeting portfolios, the simplest version of that logic, size their positions on recent realized volatility and therefore assume that next month will look like last month. This study asks whether daily S&P 500 prices and the VIX can predict, one month ahead, when that assumption will fail, meaning when realized volatility over the next 21 trading days will exceed the trailing 21-day volatility the portfolio is sized on, and whether acting on that prediction reduces the cost of sizing errors. An L2-regularized logistic regression is trained on 1990 to 2014, tuned on 2015 to 2019 and evaluated once on 2020 to 2026, implemented three ways: scikit-learn, a hand-written PyTorch training loop and the standard `nn.Module` and `torch.optim` workflow. Out of sample the model reaches an AUC of 0.79, against 0.65 for the rule of thumb a portfolio manager would use (vol tends to rise when it sits below its 1-year average). The strongest predictor is the gap between implied volatility (VIX) and realized volatility. Used as a binary decision with a miss priced at three times a false alarm, the model lowers the cost of sizing errors from 17.8 to 13.2 bps per decision, but in a monthly backtest it does not outperform simply running 25% less risk at all times. The three implementations converge to the same solution, with predicted probabilities within 3.1e-5 of each other and identical decisions on all 1,681 test days.

## Dataset

**Source and structure.** The data are daily S&P 500 index prices (open, high, low, close, volume) and the daily VIX index, downloaded from Yahoo Finance with `download_data.py` and committed to `data/raw/`, so training never touches the network and the results can be reproduced exactly. One row is one US trading day. The raw files contain 9,261 S&P rows and 9,263 VIX rows from 2 January 1990 to 9 October 2026, under 1MB in total. After cleaning and feature construction, the modelling table has 9 features, a binary target and a few helper columns (trailing and future realized volatility, leverage, next-month portfolio return) used for the baselines and the business cost.

**Why this dataset.** Volatility is the one quantity in equity markets that is genuinely forecastable, because it clusters in time and reverts to a long-run level, and sizing on it is a central risk decision for any systematic or multi-manager fund. The dataset also forces the three problems this assignment is about: the target does not exist in the raw data and has to be constructed, the observations form a time series in which a careless split leaks the future into training, and the decision threshold has a clear economic cost behind it.

## Business / real-life framing

**Industry context.** The hedge fund industry is increasingly dominated by multi-manager platforms, the so-called pod shops, such as Citadel and Millennium. Their business model combines many small, independent investment teams, each running a book close to market neutral, into one portfolio that earns a modest return with very low volatility, in stylized terms of the order of 4% to 5% a year before leverage. Because that return comes with so little volatility, the platform can lever it several times over and turn it into an attractive return on capital. Leverage at these firms has risen markedly: gross leverage at multi-strategy funds was reported at 11.8x at the end of the third quarter of 2024 (Ghose, 2025). The model has one structural weakness. Leverage only works as long as volatility behaves as estimated. When volatility is underestimated, and above all in tail events, losses that would be tolerable without leverage are multiplied by it, portfolio managers hit their drawdown limits and lose capital or their seat, and at the platform level a large enough shock can be existential. For these firms, measuring volatility well and anticipating when it is about to rise is not a refinement but the core of the business. This study addresses one concrete piece of that problem.

**Setting.** Consider a portfolio that holds the S&P 500 at a leverage equal to a 10% volatility target divided by trailing 21-day realized volatility, capped at 2x, and resets that sizing monthly. This is the standard volatility-targeting rule. Its weakness is that it implicitly forecasts next month's volatility with last month's. When next month turns out more volatile, the portfolio runs above its risk target, and in a multi-manager fund that is how drawdown limits are breached. The decision studied here is therefore taken every day: **should the portfolio cut gross exposure by 25% for the next month?**

**Research question.** Can information available at the close of day t predict whether realized volatility over the following 21 trading days will exceed the trailing 21-day volatility, and does a model of that event beat the rule a practitioner would use without one, both statistically and in terms of the cost of sizing errors?

**Target definition.** The label is 1 when realized volatility over the next 21 trading days ends above the trailing 21-day volatility the portfolio is sized on, and 0 otherwise. This relative definition was preferred to an absolute one ("next month's volatility above 20%") for two reasons measured on the training years. First, the absolute question is largely answered by the sizing rule itself: predicting that high volatility stays high already gives 0.86 accuracy, against 0.80 for always predicting calm, so a model has little to add. Second, the frequency of the absolute event shifts strongly across eras (9% of days in the 1990s, 34% in the 2000s), while the relative event occurs on 46% to 49% of days in every block. The horizon is 21 days because that is the window over which the portfolio is sized and held. A 5-day horizon would provide more independent observations, but each side of the comparison would rest on only five returns, and the strongest single predictor (the VIX gap) falls from an AUC of 0.75 to 0.60.

**Split design.** A random split is not valid here. Consecutive days share 20 of their 21 future returns, so their labels are nearly identical, and a random split would place near-duplicate observations on both sides and inflate every metric. The data are therefore split into three fixed chronological blocks: **training, 1990 to 2014** (6,028 days), **validation, 2015 to 2019** (1,237 days) and **test, 2020 to 2026** (1,681 days). A day is kept in a block only if its entire 21-day label window ends inside that block, which purges the last 21 days before each boundary (42 days in total). Every modelling choice (regularization, learning rate, decision threshold, feature selection) is made on validation, and the test block is evaluated once. The test years are a demanding out-of-sample period, containing the COVID crash, the 2022 bear market and the April 2025 tariff shock. A single fixed split was preferred to walk-forward retraining because the comparison between implementations requires identical data, and the model was not refitted on training plus validation after tuning, because the threshold is chosen for the model estimated on 1990 to 2014 and should be applied to that same model.

**Decision metric.** The threshold is set by business cost rather than accuracy or F1. A **false alarm**, cutting exposure in a month where volatility did not rise, gives up 25% of the portfolio's next-month return in a calm month: **29.4 bps** on average over the training years. A **miss** is harder to price, and the data show why. In terms of average return a miss costs almost nothing: the portfolio's mean return in vol-rising months is 0.006%, so a haircut saves 0.15 bps on average. Its cost lies in the tail. On the training years, 97% of the portfolio's worst 5% of months are vol-rising months, the probability of a monthly loss beyond 5% is 7.2% when volatility rises against 0.1% when it does not, and the haircut lowers that probability to 3.1%. The haircut therefore works as insurance, with no average payoff but a large payoff when a drawdown limit is at stake. A miss is priced at **three times a false alarm (88.3 bps)** as an explicit risk-appetite assumption, whose influence on the results is reported in a sensitivity analysis and exposed as a slider in the dashboard. The decision threshold is the probability that minimizes total cost on validation. Regularization, by contrast, is tuned on validation log loss, because its role is to produce good probabilities, while the threshold converts probabilities into decisions.

**Baselines.** Two baselines serve different purposes. **Never cutting** (the majority class) is a floor that only tests whether the model learned anything. The **mean reversion rule**, which predicts a rise whenever 21-day volatility sits below its 1-year average, is the benchmark: it is the decision a practitioner would take without a model. On the training years it reaches 0.64 accuracy, whereas a momentum rule (volatility rose last month, so it keeps rising) reaches only 0.39, worse than chance. That contrast is the first evidence that this target is driven by mean reversion.

## Data preparation & feature engineering

**Data quality.** Four problems were found in the raw files and handled explicitly in `volmodel/data.py`. First, the S&P **open price is not genuine** for much of the sample: it equals the previous close on 84% of days between 1990 and 2005 and on 15% of days between 2006 and 2013, so it is dropped and no feature uses it, which rules out overnight gaps and the Garman Klass estimator. High and low prices are reliable: the ratio of Parkinson high-low variance to close-to-close variance lies between 0.54 and 0.90 in every year except 2020 (0.37, when overnight gaps dominated), with no structural break in the early years. Second, the **VIX open, high and low are flat** on all 506 trading days of 1990 and 1991, so only the VIX close is used. Third, two **VIX observations fall on US holidays** without S&P trading (25 May and 7 September 2026) and are removed by the join. Fourth, the final 21 days have no label yet. They are kept as "live" observations, predicted in the dashboard but never scored, and the first 252 days are consumed by the 1-year features.

**Features.** Every feature uses only information available at the close of day t. This is enforced by a test in `tests/test_pipeline.py`, which checks that features computed on data truncated at a given date are identical to features computed on the full history.

| Feature | Definition | Rationale |
|---|---|---|
| `log_rv_1d`, `log_rv_5d`, `log_rv_21d` | Log annualized realized volatility over the last 1, 5 and 21 days | Volatility has short and long memory at once (the HAR structure of Corsi, 2009). Logs because volatility is right-skewed |
| `log_rv_21d_vs_1y` | Log of 21-day volatility over 1-year volatility | Distance from the long-run level, the continuous version of the mean reversion rule |
| `vix_gap` | Log VIX minus log 21-day realized volatility | Risk priced by the options market that has not yet appeared in returns |
| `vix_change_5d` | 5-day change in log VIX | Implied volatility moving ahead of realized volatility |
| `range_vs_close_5d` | Log of Parkinson high-low volatility over close-to-close volatility, last 5 days | Intraday turbulence that closing prices miss |
| `return_21d` | S&P 500 log return over the last 21 days | Leverage effect: declines raise volatility more than rallies |
| `drawdown_1y` | Log distance from the 1-year closing high | Volatility tends to rise in drawdowns |

**Preprocessing.** Each feature is winsorized at its 1st and 99th training percentiles and then standardized with the training mean and standard deviation. No statistic is computed on validation or test data. Winsorizing matters because the test years include COVID, when several features moved far outside their training range, and a linear model would translate those values directly into extreme probabilities. Standardizing matters for two reasons: gradient descent converges poorly when features have very different scales, and the L2 penalty treats all coefficients alike, which is only meaningful when features share a common scale. All three implementations receive exactly the same float32 matrix.

**Feature selection.** Eleven candidate features were built and nine retained. The table adds feature groups cumulatively and reports validation performance with the exact solver.

| Feature group added | Validation AUC | Validation log loss |
|---|---|---|
| HAR realized volatility (1d, 5d, 21d) | 0.728 | 0.630 |
| + Mean reversion (21d vs 1y volatility) | 0.731 | 0.623 |
| + VIX (gap to realized, 5d change) | 0.787 | 0.558 |
| + Intraday range (Parkinson vs close) | 0.790 | 0.549 |
| + Window shape (spike share, down share) | 0.789 | 0.551 |
| + Price path (21d return, 1y drawdown) | 0.791 | 0.551 |

The **VIX group carries most of the predictive power**: it adds 0.055 of AUC and reduces log loss from 0.623 to 0.558, and removing it from the full model costs 0.027 of AUC, more than any other group. The **intraday range** adds a small further improvement. The **window shape features did not work.** They encoded two hypotheses: that a single large day inflating trailing volatility will drop out of the window and leave next month calmer (the share of 21-day variance from the largest day), and that variance from down days matters more (the share from negative returns). Adding them lowered validation AUC, raised log loss and produced coefficients close to zero, because the 1-day and 5-day volatility terms and the 21-day return already carry that information. They were removed. One caveat applies to the table: removing the HAR group or the mean reversion term from the full candidate model slightly *raises* validation AUC (to 0.793 and 0.795). These terms overlap by construction (`log_rv_21d_vs_1y` equals `log_rv_21d` minus log 1-year volatility), and with roughly 59 independent months in validation, AUC differences of 0.005 are within sampling noise. They were kept because their coefficients are large and economically interpretable, and only the features that showed no signal on either criterion were dropped.

**Estimated relationships.** On standardized features, the largest coefficient belongs to the **VIX gap (+0.91)**: when options price more risk than has been realized, volatility tends to rise. Trailing 21-day volatility (**−0.42**) and its distance from the 1-year level (**−0.24**) capture mean reversion. Five-day volatility enters positively (**+0.27**), so volatility accelerating within the month raises the probability. The 21-day return enters negatively (**−0.26**), consistent with the leverage effect. The intraday range (+0.17), the drawdown (−0.10), the VIX change (+0.08) and 1-day volatility (+0.04) play smaller roles. Every sign is consistent with the economic interpretation, and the three implementations agree on all of them.

## Modeling: three implementations, one model

**Model.** The model is an L2-regularized logistic regression, chosen because the target is binary and the decision compares a probability with a cost-based threshold. All three implementations minimize the same objective: mean binary cross-entropy plus (λ/2)‖w‖², with an unpenalized intercept. scikit-learn parameterizes this as C × Σ cross-entropy + ½‖w‖², so dividing by C × n yields **λ = 1 / (C × n)**. This mapping is what allows the three implementations to agree.

**Implementations** (`volmodel/models.py`). The **scikit-learn** version solves the problem with L-BFGS to a tolerance of 1e-10 and serves as the reference solution. The **manual PyTorch loop** follows the approach of Session 5: raw tensors for the weights and intercept initialized at zero, the cross-entropy written from its formula in numerically stable form, `loss.backward()` to compute gradients, a hand-written update `w -= lr * w.grad` inside `torch.no_grad()`, and gradients reset after every step. The **standard workflow** uses an `nn.Module` containing one `nn.Linear` layer (randomly initialized with a fixed seed), `BCEWithLogitsLoss`, and `torch.optim.SGD` with `weight_decay = λ` applied to the weights only, which adds exactly λw to the gradient. Both PyTorch versions use full-batch gradient descent for 3,000 epochs.

**Hyperparameters.** C was selected from {0.001, 0.01, 0.1, 1, 10, 100} by validation log loss, giving **C = 0.01** (log loss 0.553 against 0.561 at C = 1) and λ = 0.0166. Moderate regularization helps because the HAR terms are strongly correlated. C was tuned once with the exact solver and transferred to PyTorch through λ, since it is a property of the model rather than of the optimizer. The learning rate affects only how quickly gradient descent reaches the optimum, not the optimum itself, so it was evaluated by the distance to the scikit-learn objective after 3,000 epochs. At 0.01 both PyTorch versions remained 5e-4 away from the optimum. At 0.1, 0.5 and 1.0 they reached it within float32 rounding, and **lr = 0.1**, the smallest rate that converged, was retained.

**Results on the test block (2020 to 2026).** None of these data were used for any modelling choice.

| Model | AUC | Log loss | Brier | Accuracy | Precision | Recall | Days flagged | Cost per decision (bps) |
|---|---|---|---|---|---|---|---|---|
| scikit-learn | 0.7946 | 0.5477 | 0.1852 | 0.603 | 0.553 | 0.947 | 83% | 13.2 |
| PyTorch manual loop | 0.7946 | 0.5477 | 0.1852 | 0.603 | 0.553 | 0.947 | 83% | 13.2 |
| PyTorch nn.Module + optim | 0.7946 | 0.5477 | 0.1852 | 0.603 | 0.553 | 0.947 | 83% | 13.2 |
| Baseline: never cut (majority class) | 0.5000 | 0.6927 | 0.2498 | 0.515 | n/a | 0.000 | 0% | 42.8 |
| Baseline: mean reversion rule | 0.6545 | n/a | n/a | 0.608 | 0.570 | 0.779 | 66% | 17.8 |
| Policy: always cut | n/a | n/a | n/a | n/a | n/a | 1.000 | 100% | 15.2 |

Validation results are consistent: AUC 0.790 for all three implementations against 0.708 for the rule, and 12.6 against 17.6 bps per decision. Accuracy is not an informative criterion here. The rule's accuracy is similar to the model's, but the model's errors are mostly false alarms, which are cheap, while the rule's are mostly misses, which are expensive (180 misses against 43).

**Agreement between implementations.** The three implementations agree to float32 precision. They reach the same training objective, 0.543864464, to the ninth decimal. On the 1,681 test days their predicted probabilities differ by at most **3.1e-5** between scikit-learn and either PyTorch version and by 4.4e-5 between the two PyTorch versions, and they produce **identical decisions on every day**. The residual differences have three sources: PyTorch computes in float32 while scikit-learn uses float64, gradient descent stops after a finite number of steps, and the two PyTorch versions start from different initial values (zeros and random weights). That last point is itself a check: because logistic regression is convex, both starting points lead to the same minimum. Two further results show that this agreement is not automatic. With a learning rate of 0.01 the PyTorch versions had not converged and their coefficients were visibly different. With the original eleven features, the largest probability gap was ten times larger (3.7e-4, with one day decided differently), because the redundant features flattened the objective along some directions and the solvers stopped at slightly different coefficients with the same loss. Removing them improved the conditioning of the problem.

**Decision threshold.** The cost-minimizing threshold on validation is **0.16**. For perfectly calibrated probabilities the optimal threshold would be 1 / (1 + 3) = 0.25. The model is reasonably calibrated on the test block (days with predicted probability below 0.2 saw a rise 15% of the time, days above 0.8 saw one 89% of the time), and the validation cost curve is flat between the two values (12.6 bps at 0.16, 13.1 at 0.25). On test, the theoretical threshold would have performed slightly better (12.8 bps). The validation choice was kept, since changing it after observing test results would use the test set for a modelling decision.

**Sensitivity to the cost of a miss.** The threshold is re-selected on validation for each multiple, and the cost is measured on test in bps per decision.

| Miss cost (multiple of a false alarm) | Threshold | Days flagged | Model | Always cut | Never cut | Mean reversion rule |
|---|---|---|---|---|---|---|
| 0.5x | 0.82 | 13% | 5.8 | 15.2 | 7.1 | 10.0 |
| 1x | 0.47 | 47% | 8.1 | 15.2 | 14.3 | 11.5 |
| 1.5x | 0.42 | 51% | 10.4 | 15.2 | 21.4 | 13.1 |
| 2x | 0.42 | 51% | 12.3 | 15.2 | 28.5 | 14.7 |
| 3x | 0.16 | 83% | 13.2 | 15.2 | 42.8 | 17.8 |
| 5x | 0.16 | 83% | 14.7 | 15.2 | 71.4 | 24.2 |

The model is the cheapest policy at every multiple, but its advantage over always cutting narrows as misses become more expensive. At 3x it flags most days and saves only 2 bps per decision relative to a permanent cut. The model is most useful where the decision is genuinely two-sided, between 1x and 2x.

**Portfolio backtest.** A monthly-rebalanced backtest on the test block, with one sizing decision every 21 trading days (81 months), S&P price returns only and no transaction costs, gives the following results.

| Strategy | Annual return | Annual vol | Sharpe | Max drawdown | Worst month | Months below −5% |
|---|---|---|---|---|---|---|
| Volatility target only | 8.2% | 12.2% | 0.71 | −14.9% | −9.7% | 4 |
| Always 25% less risk | 6.2% | 9.1% | 0.71 | −11.3% | −7.3% | 1 |
| With model overlay (cut 68 of 81 months) | 6.1% | 9.6% | 0.67 | −13.4% | −7.3% | 2 |
| With mean reversion rule overlay (cut 55 of 81) | 6.5% | 10.0% | 0.68 | −13.4% | −7.3% | 4 |

The overlay reduced the tail, halving the number of months below −5% relative to the unhedged portfolio, but a permanent 25% reduction did as well or better on every risk measure with the same Sharpe ratio. Two episodes illustrate why. In **July 2024**, trailing volatility was 6% and the portfolio was running at 1.6x leverage. The model assigned a 0.88 probability to a rise and was correct: the August carry-trade unwind would have cost 9.7% at full size and 7.3% after the haircut. In **March 2020**, trailing volatility was already 34% and the VIX stood at 32. The model assigned a probability of 0.13, expecting mean reversion, and realized volatility reached 93%. The low leverage limited the loss to 5.7%, but the episode exposes the main weakness of a model dominated by mean reversion: it fails to anticipate the escalation of a shock already under way. With 81 monthly decisions and a few episodes driving the outcome, none of these backtest differences is statistically significant.

## Limitations & next steps

**The binary label ignores magnitude.** A month in which volatility ends 1% above its trailing level counts the same as March 2020, when it nearly tripled, and the cost function treats both misses identically. This is the main reason the classification results look stronger than the backtest. A natural extension is to model the log ratio of next-month to trailing volatility with a linear regression, using the same pipeline with a continuous target, or to weight observations by the size of the overshoot, and to scale exposure continuously instead of switching at a threshold.

**Escalating crises are the main blind spot.** The coefficients are dominated by mean reversion, so the model expects high volatility to fall, which is usually correct and badly wrong in episodes such as March 2020. Features that distinguish a fading shock from a spreading one, such as the VIX term structure (VIX against 3-month VIX, available from 2007) and credit spreads from FRED, should be tested on the same validation block.

**The 3x miss cost is an assumption, not an estimate.** The sensitivity analysis shows that the conclusions depend on it. In practice it should be derived from the fund's actual drawdown rules, as the probability of hitting a stop multiplied by the capital at risk.

**The effective sample is small.** Because labels overlap, validation contains roughly 59 independent months and test 81. AUC differences of a few thousandths and all backtest differences lie within sampling noise. Block-bootstrap confidence intervals and a walk-forward evaluation over 2000 to 2026 would quantify that uncertainty.

**A single split without retraining.** A production model would be re-estimated as new data arrive. Annual walk-forward retraining, together with monitoring of calibration drift, would test whether the threshold keeps its meaning over time.

**Simplified portfolio and imperfect data.** The backtest uses price returns without dividends, financing or trading costs, rebalances only monthly and relies on Yahoo data whose open prices proved unreliable. A total-return index from a professional vendor and a cost charged on every haircut would make the economic results more realistic.

## Conclusion

The answer to the research question is yes for prediction and not yet for the decision. For a leveraged platform the distinction matters, because what threatens it is not average performance but the months in which volatility is underestimated. Daily prices and the VIX predict, out of sample, when next month's volatility will exceed the level a volatility-targeting portfolio is sized on, with an AUC of 0.79 against 0.65 for the rule a practitioner would use. The signal comes mainly from the gap between implied and realized volatility and from mean reversion. Turning that ranking into a binary cut lowers the cost of sizing errors, but over 81 test months it did not improve on simply running less risk at all times, because the binary label ignores how large a volatility rise is and because the model misses escalating crises. The ranking is useful, the on/off switch is not yet, and the natural next step is to size continuously on a forecast of the volatility ratio. Methodologically, the three implementations reach the same optimum and identical decisions, which shows that the model is defined by its objective rather than by the software used to fit it, provided the regularization is mapped consistently and gradient descent is allowed to converge.

## References

Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), pages 174 to 196.

Ghose, R. (2025). Hedge fund leverage latest stats. *Substack*. https://rupakghose.substack.com/p/hedge-fund-leverage-latest-stats

Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), pages 61 to 65.

## Generative AI use disclosure

I used Claude (Anthropic) for the following:

1. Validating whether my ideas were sound and comparing them with academic research from decades ago.
2. Writing the code and finding the most efficient way to run the analyses. Fixing a lot of bugs and problems.
3. Building and formatting the dashboard. Guiding me through the code for making it interactice as well as informative as possible.
4. After writing the report, I ran it through Claude to check for any spelling or grammatical errors, as well as formatting issues, to ensure it was ready for submission.
