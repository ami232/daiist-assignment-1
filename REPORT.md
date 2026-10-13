# Assignment 1 Report



- **Name**: Antonio Vicenzo Argenta Medina
- **Student ID**: 20498
- **Email**: aargenta.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

I used the Kepler Objects of Interest (KOI) "cumulative" table from the NASA Exoplanet Archive (https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative). A copy of the CSV is committed in the repo at `data/koi_cumulative.csv`.

Each row is one KOI: a repeating dip in a star's brightness that Kepler flagged as a possible planet passing in front. The table has about 9,500 rows and 153 columns (roughly 12 MB). Of those, 2,748 are CONFIRMED, 4,839 are FALSE POSITIVE and 1,977 are still CANDIDATE, meaning not defined yet.

I picked it because I wanted to work on something related to space, but also because there's a real decision behind the data, that someone has to choose which signals are worth chasing to define a planet through Kepler's method. On top of that, the raw columns don't tell the whole story on their own, so feature engineering actually matters here.

## Business / real-life framing

Telescope time to confirm a planet is scarce and expensive. The idea is that the model ranks incoming signals by how likely they are to be real, and the team of astronomers would only follow up the ones above a certain probability threshold.

There are two ways to get it wrong. Following up a signal that turns out to be a false positive wastes some telescope time, which I gave a cost of 1. Missing a real planet costs 3 (being a false negative). I think a miss is clearly worse: wasted time is a bounded, one-off loss, but a planet we dismiss might never get looked at again, and confirmed planets are the whole point of the mission. With those costs, it's worth following up a signal whenever the expected cost of skipping it is higher than the expected cost of checking it, i.e. when 3p > (1 − p). That works out to p > 0.25.

**Target.** I kept only CONFIRMED (1) and FALSE POSITIVE (0) and dropped CANDIDATE entirely. A candidate isn't a "no", it's an "we don't know yet", so labelling it 0 would just teach the model wrong answers. I also made sure never to use columns that leak the answer: `koi_score`, `koi_pdisposition`, the `koi_fpflag_*` flags and `kepler_name`. These are either outputs of the vetting process itself or only exist once a planet has been confirmed.

**Split.** I split by star rather than randomly. Many stars have several KOIs, and these siblings share the same stellar values (radius, temperature, and so on), so a random split would let the model see near-copies of test rows during training. I did think about a time-based split, and to be honest it would have been a reasonable choice, since KOI numbers roughly follow discovery order. I chose not to do it and stuck with the grouped split, but I come back to this in the limitations.

**Threshold metric.** The threshold is chosen by minimising business cost, not by maximising accuracy. Accuracy treats both mistakes as equally bad, and in this problem they really aren't.

## Data preparation & feature engineering

- **Log transforms.** Several columns are heavily skewed and span orders of magnitude (depth, period, radii). I took logs of these. Without them, gradient descent didn't converge properly, which was one of the first things I ran into.
- **SNR per transit.** A measure of how strong a single dip is. A real planet should produce a clear, consistent signal each time it passes.
- **Centroid offset.** If the light centre of the image shifts during the dip, the dip is probably coming from a neighbouring star rather than the target, which is a classic false-positive pattern.
- **Flags.** `is_giant` for objects above 15 Earth radii (too big to be a planet in most cases), `is_grazing` for impact parameter above 1 (the object only clips the edge of the star), and `is_multi` for stars with more than one KOI.

**Data quality.** I capped the impact parameter at 2, because values going up to 100 are clearly failed fits, not physics. Missing values were filled with the training-set medians, and the scaler was fitted on the training set only. Fitting either of these on the full data would let information about the test set leak into training and make the results look better than they really are.

## Modeling: three implementations, one model

I used logistic regression because the target is binary and, more importantly, I need actual probabilities to apply the cost-based threshold.

I built the same model three times:

1. **scikit-learn**, using its ready-made solver.
2. **A manual PyTorch loop**, where I write every step myself: predict, compute the loss, backpropagate, update the weights.
3. **`nn.Module` + `optim.SGD`**, where PyTorch handles the update step.

To make sure they really are the same model, all three use the same split, the same features, the same loss and the same L2 penalty. Since scikit-learn expresses regularisation through C, I converted it for the PyTorch versions using lam = 1 / (C · n).

**Tuning.** I tried C in {0.01, 0.1, 1, 10, 100} and ended up with C = 100, though validation loss is basically flat from C ≈ 1 onwards, so the choice doesn't matter much past that point. For the learning rate I tried {0.01, 0.1, 1.0} and picked 1.0 for both PyTorch versions. I didn't tune anything else.

### Results

| Model | ROC-AUC | Log loss | Threshold | Precision | Recall | Cost / 100 KOIs |
|---|---|---|---|---|---|---|
| Naive baseline (follow up all) | 0.500 | 0.672 | 0.00 | 0.391 | 1.000 | 60.9 |
| scikit-learn | 0.957 | 0.263 | 0.28 | 0.801 | 0.952 | 14.9 |
| Manual PyTorch | 0.957 | 0.263 | 0.28 | 0.803 | 0.952 | 14.8 |
| nn.Module + SGD | 0.957 | 0.263 | 0.28 | 0.803 | 0.952 | 14.8 |

**Do they agree?** Yes, on everything that matters. The metrics are the same, and the coefficients match except for `log_depth`, `log_prad` and `log_srad`. That makes sense once you look at the physics: transit depth is roughly (planet radius / star radius)², so in log space these three features are almost a linear combination of each other. The loss surface is very flat along that direction, so gradient descent moves slowly there and ends up at a slightly different point than scikit-learn's solver, even though the predictions are practically identical.

The manual loop and the `nn.Module` version are identical to each other, which is expected: plain SGD is exactly the update rule I wrote by hand. scikit-learn uses a different solver, which explains the tiny differences. Interestingly, my first run without the log transforms disagreed a lot more, which is another sign of how much the scaling helped optimisation.

**Compared to the baseline**, the model is about 4× cheaper (14.8 vs 60.9 per 100 KOIs) while still catching 95% of the real planets.

**Interpreting a few coefficients.** Centroid offset and `is_giant` are strongly negative, which matches intuition: a shifting light centre points to a neighbour star, and something giant is more likely a star or a blend than a planet. `is_multi` is positive, meaning signals from stars with several KOIs are more likely to be real (with a caveat, see below).

## Limitations & next steps

- **The costs are assumed, not measured.** The 1-to-3 ratio is my own judgement. With real follow-up costs from an observatory the threshold could move. In the meantime, the slider shows how cost and the confusion matrix change with the threshold.
- **Labels partly reflect the confirmation process.** Multi-planet systems were often confirmed *because* they were multis, so `is_multi` is partly circular. A good check would be retraining without that feature and seeing how much performance drops.
- **CANDIDATEs aren't like the training data.** They tend to be newer, smaller and weaker signals, so the probabilities may not transfer well to them, and this is exactly where the model would be used in practice. A time-ordered evaluation, or recalibrating on recently vetted KOIs, would give a more honest picture.
- **It's a linear model.** It can't learn interactions between features (for example, depth only being suspicious for certain star types). Hand-crafted interaction features or a tree-based model would be the next step.
- **Overlapping depth and radius features.** As shown above, these three are nearly redundant. Dropping one of them, or relying more on regularisation, would make the coefficients more stable and easier to interpret.

## Generative AI use disclosure

I employed AI to help me construct the syntax for the models I wanted to develop. I used class learnings and previous knowledge to build the structure of the 3 types of implementation of logistic regression, and AI aided me in combining the codes arranged by me into a refactored, clean look. 

Moreover, from the insights I got from developing the project, understanding the business outcomes of classifying a planet positively or not, I bundled my thoughts into a blend of writings, and used AI to help make sense of the findings. Later writing that structured version of my thoughts into this present Report, and using an LLM to organize it and make it clean.