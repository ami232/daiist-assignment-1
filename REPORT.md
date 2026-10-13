# Assignment 1 Report

- **Name**: James Haunso
- **Student ID**: 19661
- **Email**: jhaunso.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

Since I am interested in venture capital I chose a dataset that would let me work on something similar to what an analyst at a VC would do. So I got a Kaggle dataset made for Start-up Success Prediction. It’s from around late 2013. One row is a US start-up with location, sector, founding date, funding history, milestones and final status. 923 rows and 49 columns. 

## Business / real-life framing

The scenario is that you are an analyst at a VC deciding which start-ups get a meeting and which don't. The model will give a probability that the startup will be acquired. It’s a good indicator to sort the pile of startups but the decision remains in the hands of the analyst and the partners. Only information an analyst would actually have is included otherwise the model is cheating. (many columns give away the answer). 

Target: acquired = 1, closed = 0. labels (an exact copy of the target) and closed_at (only filled in for closed companies) were removed. An acquisition is only a real win if the exit is valuable, and the data doesn’t record exit size.

Split: by founding date. The oldest 60% of startups train, the next 20% validate, the newest 20% test (founded 2008–09). Startups founded after 2009 were removed. A time-based split mirrors how the fund would actually use the model, learning from past startups to judge new ones. Startups founded after 2009 were removed because by 2013 many had not yet had time to be acquired, which would unfairly label them as less successful.

Threshold: chosen on the validation set to minimise cost = 1 × wasted meeting + 5 × missed winner. This is because missing a winner is wayyyyyy worse than taking a meeting that isn’t successful. If you miss out on a great startup because the threshold is too high the boss will be very angry.

## Data preparation & feature engineering

- Removed leaky and identifier columns: labels, closed_at, the ID columns, name, a duplicate state column.
- Removed 46 startups whose first funding date is before their founding date. At this stage ages aren’t features, so this acts as a data-quality filter only.
- State is grouped into CA / NY / MA / TX / other. Sector is one-hot encoded, with categories that have fewer than 20 training rows grouped as “other”.
- All preprocessing is fit on the training rows only.
- Final data: 744 startups (446 train / 149 validation / 149 test).

## Modeling: three implementations, one model

Logistic regression, because the target is binary. All three versions minimise the same objective (mean log loss + L2 penalty, intercept not penalised). C = 1.0 was chosen on validation log loss; lr = 1.0 for both PyTorch versions; 4,000 epochs of full-batch gradient descent.

| Model | ROC AUC | PR AUC | Log loss | Threshold | Meetings (of 149) | Cost / 100 |
|---|---|---|---|---|---|---|
| scikit-learn | 0.692 | 0.816 | 0.572 | 0.44 | 149 | 30.2 |
| Manual PyTorch | 0.692 | 0.816 | 0.572 | 0.44 | 149 | 30.2 |
| nn.Module + optim | 0.692 | 0.816 | 0.572 | 0.44 | 149 | 30.2 |
| Baseline (train rate) | 0.500 | 0.698 | 0.613 | 0.00 | 149 | 30.2 |

The three agree to 2.5e-6 in predicted probability and 2.4e-4 in coefficients. The gap shrinks with more epochs (4e-4 at 1,000, 1e-7 at 16,000), so it comes from gradient descent not being fully converged, not from a difference in the models. At 5:1 the threshold falls below every test prediction, so the model recommends meeting all startups, the same decision as the baseline.

Although meeting everyone slightly defeats the point, it is crucial to do so. If you miss the right start-up your job is gone as an analyst. Of course to solve this you could increase the threshold but in my opinion that's the wrong move. But you can also use the model to understand who needs to be met first and who has priority. 

## Limitations & next steps

The target ignores exit size. A small acqui-hire counts the same as a large exit, so “acquired” is a rough stand-in for what a fund actually cares about. Next step: add acquisition prices (e.g. from Crunchbase’s acquisitions data) and only count exits above a set value as a success.

Some outcomes are missing entirely. Startups still operating in 2013, and those that went public, are not in the data, so the model learns “acquired vs closed” rather than “good vs bad startup”. Next step: pull the full Crunchbase company list, including operating and IPO companies, and count an IPO as a success too.

Data quality. 46 records show funding before the founding date, and at least one has contradictory location data, so some records can’t be fully trusted. Next step: cross-check dates and locations against another source, such as SEC filings or company websites, before using them.

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment. It wrote the code for the training pipeline, experiments and Gradio dashboard, ran comparisons of the pipeline options, and helped draft the report by organising my answers and suggesting wording and options. I made the final choices on the dataset, framing and pipeline decisions, and reviewed every section.
