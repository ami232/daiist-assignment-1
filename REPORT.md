# Assignment 1 Report

- **Name**: Anna Serra
- **Student ID**: 18815
- **Email**: [aserrag.ieu2022@student.ie.edu](mailto:aserrag.ieu2022@student.ie.edu)
- **Group**: BBADBA 5B

## Dataset

I used the [StartUp Investments (Crunchbase)](https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase/data) dataset from Kaggle. This dataset is a 2014 snapshot of Crunchbase's data on venture-backed companies. The file has 54,294 rows and 39 columns, but 4,856 rows are completely empty. That leaves 49,438 startups, each represented by a non-empty row. If you focus on companies founded between 2000 and 2008, which is the group used in the modeling section, there are 12,214 companies.

The columns cover details like company name, market, country code, founding and funding dates, funding amounts by round (such as seed, venture, and rounds A to H, plus total funding in USD), and a status category (acquired, operating, or closed).

I chose this dataset because the main business question, which is who is likely to be acquired, matches the status column. The funding columns allow for interesting feature engineering, and the founding and funding dates make it easy to use a time-based split. The full export, about 12.5 MB, is available in the repository (under the 20 MB guide). Since the dataset has more than 20,000 rows, the model uses the 12,214 companies from 2000 to 2008 instead of all startups in the file.

## Business / real-life framing

This is a small M&A advisory firm that is staffed by bankers with limited time, so they need to be selective about which venture-backed startups they approach for the sell-side pitches. To help with this, I propose they use a model that estimates how likely each startup is to be acquired compared to others of a similar age, based on funding history. Then, the bankers will use these scores to decide the order in which they contact companies, and they review the list regularly, either weekly or whenever they have time.

My goal was to focus on the status variable, selecting cases where 'acquired' is 1 and 'operating' or 'closed' is 0. Since there is no acquisition date, I could not create a label like 'acquired within X years.' Instead, I used only the 2014 status snapshot. The raw acquisition rates drop as companies get younger. For example, 22.4% of those founded in 1999 were acquired, compared to just 0.6% for those from 2013, since younger companies have had less time to be acquired. To make the label more meaningful, I included only companies founded between 2000 and 2008, so each has at least six years of data by the snapshot. I also added company_age as a control, so the model does not treat being young as the same as never being acquired. I used data from 2005 for training, 2006 for validation, and 2007 to 2008 for testing, which involved about 4,598 startups. Estimating outcomes for newer startups by looking at what happened to older ones is how a boutique firm would actually use the model. If I had used a random split, it would have mixed different cohorts, let the model see companies founded after the ones it was supposed to rank, and hidden the real shift in acquisition rates across the split (from about 16% to about 13.6% to about 9.7%). The time-based split, however, makes this shift clear. There are some compromises I am willing to accept. The ages in the test set (ranging from 6 to 7) are outside the training age range (9 to 14), so any age coefficient is being extrapolated. The funding figures might include rounds that happened after an acquisition. The mean predicted probability on the test set ends up only slightly lower (0.086 instead of the actual value of 0.097). On its own, base-rate drift would make the predictions too high. When I set the company_age of every test company to 9 (the youngest age in the training set), the mean prediction rises from 0.086 to 0.135, which is above the actual value of 0.097. So, the positive age coefficient over-corrects when applied to the younger test cohorts. That is why the probabilities end up being low.



I estimated the costs of errors as follows:


| Error          | What happens                                                | Illustrative cost |
| -------------- | ----------------------------------------------------------- | ----------------- |
| False positive | Pitch a company that is never acquired                      | €3,000 about a day of associate research + deck prep |
| False negative | Miss a company that does get acquired (without our mandate) | €37,500 expected, so a €150,000 sell-side success fee (about 1–2% of a ~€10M deal) × a 25% win-rate assumption. Missing an acquirer only costs the fee if we would have pitched and won; not every seller hires an adviser, and we wouldn't win every pitch. |


False negatives are much more costly than false positives (about 12.5 times more) so we should focus on achieving high recall. The probability threshold that would minimize cost, if there were no other limits, is about 0.074. This is calculated as 3,000 divided by 40,500. On the test set, this would mean rejecting 43.9% of startups (2,018 out of 4,598), which is about nine times more than the boutique can handle in its top 5% (230). Because of this, capacity is the main constraint. We use the euro costs to compare different values of k, but they do not set the value of k.

In practice, the company can reach out to about 200 firms each year. Since the test set has two founding groups, or about 4,598 startups, I convert this capacity into a fixed share: the top 5% of companies based on their scores. This means about 230 names for the test set and about 200 for a typical year with 4,000 companies. The process is to pitch to the top-k companies, not to use a fixed probability cutoff. This also avoids the calibration issue: if the probabilities are slightly off, a fixed threshold could be misleading, but the top-k approach depends only on ranking, so it is not affected.

The dashboard shows the expected business cost for the ranked list at a chosen k, with the default set to the top 5%.

cost = (# FP among pitched) × €3,000 + (# FN among not pitched) × €37,500

Precision@k is another view that shows how many of the companies we pitch to are actual acquirers.

## Data preparation & feature engineering

The dataset-level findings, like the raw structure and acquisition rate by founding year, are based on all rows and focus on how the export is structured, not on what predicts acquisition. In contrast, decisions about variables such as funding, rounds, leakage, pace, market, and geography (exploration 1.7–1.12) were made using only the training cohorts, which include companies founded between 2000 and 2005. The validation and test sets were not used to make these design choices.

I cleaned the file, which uses latin-1 encoding and had some extra spaces in the headers. I removed 4,856 completely empty rows and extracted funding_total_usd from the text, accounting for Indian digit grouping and treating missing values as NaN. For founding and funding dates, I set any impossible values to missing. I also removed the market designation, dropped rows without a status or founded_year, and eliminated duplicates using the permalink as the identifier. After these steps, there are 37,563 labeled companies. When I focus on companies founded between 2000 and 2008, the dataset includes 12,214 companies for modeling.

Target, exposure, and split are defined as acquired is 1, while operating or closed is 0. Company age is calculated as 2014 minus the founding year. The training set covers 2000–2005, validation is 2006, and testing is 2007–2008. Age is used to control for exposure in training, so the model does not assume that younger companies are never acquired. In testing, it only compares 2007 and 2008 to each other, using a coefficient learned from older companies (ages 9–14). This is an extrapolation I acknowledge and accept. However, absolute scores are not directly comparable across the train and test age gap.

When it came to the funding features, if funding_total_usd was missing, then all the round columns were zero and the acquisition rate was lower. Since missing data should be interpreted as 'not disclosed' rather than as a random gap, I have retained those rows, introduced a funding_disclosed flag and set the total to zero. As funding amounts vary a great deal and are strongly right-skewed, I recorded the amount using the log1p of the total, seed, angel, venture and debt. The composition of the funding includes share_venture, share_debt_financing and share_seed from the different funding types. The lettered rounds are only partly nested (about 61% of companies that had a C round also had a B round). Rather than use eight sparse and correlated dummy variables for the rounds, I use max_round (with A assigned a value of 1 up to H having a value of 8, otherwise 0) to represent progress as a single ordered number. I also include flags for seed, angel, venture, debt and grant, the number of different round types, and funding_rounds. Early pace is the number of years between the date of founding and the first funding (negative values are set to zero, which might be due to data entry errors or because money was raised before incorporation, rather than excluding the company), and log_funding_per_round. The equity and debt after an IPO are not included in the funding features since that money only becomes apparent after an IPO, which is an exit.

With respect to sector and geography, I manually grouped the main markets into nine fixed sectors (for example, software_it and health_life_sciences). All the other markets, including those with missing values, are labelled as 'other' and this is taken to be the reference category with no separate column associated with it. Regarding geography, I have included a variable called is_usa, columns for each of the five largest non-U.S. countries (GBR, CHN, CAN, FRA, ISR) which were determined solely from the training data, a variable named is_hub for the SF Bay Area, Boston, New York City and London, and n_categories from category_list. I discarded state_code since approximately one-third of its values are missing. Even though exploratory analysis indicates that is_hub is a good univariate separator, I do not make any claims, solely on the basis of the fitted coefficients, about how funding varies by sector or geography, since these features are collinear (see modeling).

I created date features along with the final set, including funding_span_years and rounds_per_year. However, these can leak information if acquired companies stop raising funds after the deal, which is possible based on the exploration phase, though a shorter span alone does not prove this. Each feature set was tested at its optimal regularisation strength, determined on the validation data (both cases used C = 10). On validation data, adding date features increased ROC-AUC by 0.009. A paired bootstrap analysis with 1,000 redraws showed a 95% confidence interval of [−0.003, +0.021], which includes zero, so the result is not statistically significant. The code keeps date features only if the entire confidence interval is above zero; otherwise, the safer set is used. For this reason, the three final models use 37 features and exclude the date features.

## Modeling: three implementations, one model

The model uses logistic regression because this is a binary problem. I standardised the features with a StandardScaler trained only on the training data, then applied it to the validation and test sets. This way, gradient descent works with both the log-funding values (around 14) and the 0/1 indicators on the same scale.

I trained the same model in three ways using that matrix: with scikit-learn, with a custom PyTorch loop (using tensors, autograd, and manual updates), and with torch.nn.Linear, BCEWithLogitsLoss, and torch.optim.SGD. To make sure the results are comparable, I used the same 37 features and the same cohort split for all runs. The L2 penalty is set by converting sklearn's C to λ = 1/(C·n_train), leaving the bias unpenalised since PyTorch does not have its own C. All calculations use float64, and both PyTorch versions start with zero weights.

I didn’t go into hyperparameter tuning in as much depth because most of the effort went into feature engineering (13 exploration sections and 37 engineered columns), not into a full grid search. I only tuned the two hyperparameters allowed by the brief with small grids:

| Hyperparameter | Where | Criterion | Grid |
| --- | --- | --- | --- |
| Regularisation `C` | 5.2 | Best validation ROC-AUC (ranking metric for the boutique) | 0.001, 0.01, 0.1, 1, 10 |
| Learning rate (both PyTorch versions) | 5.4 | Lowest training loss after a 500-step trial | 0.01, 0.1, 0.5, 1.0 |

I chose C = 10 and learning rate (lr) = 1.0. C had little effect: the validation ROC-AUC was 0.7610 when C was 1 and 0.7618 when C was 10. I set the learning rate based on the training loss, since it only changes how quickly gradient descent finds the optimum, not which optimum it finds. With 100,000 steps, any stable learning rate would reach the same result. I did not change the number of steps to improve the score; instead, I increased it in section 5.7 until PyTorch matched sklearn. When comparing date features (section 4.2), I adjusted C separately for each feature set within the same grid. In both cases, C was set to 10, matching the final models.

The results were that 4,598 startups were founded between 2007 and 2008. The top 5% made up 230 pitches. The cost was calculated as false positives times €3,000 plus false negatives times €37,500.

| Model | ROC-AUC | Precision@5% | Acquirers pitched | Cost |
| --- | ---: | ---: | ---: | ---: |
| Baseline: base rate (random ranking) | 0.500 | 9.7% | 22 | €16.55M |
| Baseline: rank by total funding | 0.579 | 9.1% | 21 | €16.60M |
| scikit-learn | 0.718 | 27.8% | 64 | €14.86M |
| PyTorch (manual loop) | 0.718 | 27.8% | 64 | €14.86M |
| PyTorch (nn + optim) | 0.718 | 27.8% | 64 | €14.86M |

The model contacts acquirers about 2.9 times more often than random chance. If ranking is based only on funding, precision at 5% is no better than random. Most of the cost comes from the 383 acquirers who are not in the top 230. Capacity is the limiting factor, so the models just reassign which acquirers fill the fixed number of pitch slots.

After the first run, the three methods didn't agree. The two PyTorch versions matched each other, but some weights differed from scikit-learn's by up to 0.48, and the training loss was only a bit higher. This happened because the optimisation problem is difficult, not because of a coding mistake. The design matrix is badly conditioned (the condition number of X′X/n is about 7,300; for example, has_angel and log_angel have a correlation of about 0.996). As a result, full-batch gradient descent moves slowly along a narrow valley, while scikit-learn's solver goes straight to the optimum. At 20,000 steps, PyTorch had not caught up. At 100,000 steps, all three methods had the same objective, their weights were within about 3e-4, their test probabilities were within about 8e-6, and their metrics matched. The diagnosis for this is printed in section 5.7 of train.py.

When looking at the coefficients, I never analyse the signs of the individual features. Since correlated groups divide their effect in an unpredictable way. For instance, log_funding_total ≈ is about +3.54 but log_funding_per_round ≈ is about −2.25 because the two features overlap to some extent. It is safer to consider the features in groups, such as funding size, mix, and stage, the pace to the first funding, the sector dummies, and geography (this includes is_hub). The large coefficients within a collinear pair do not constitute stable explanations. All the artefacts (such as the model, the scaler, the test predictions, the feature table, the results, the coefficients, and the loss curves) are regenerated each time the main.py train command is run and are ignored by git; the dashboard loads them rather than retraining.

## Limitations & next steps

The biggest gap in the label is that we never learn when a company was acquired. We only know it had been acquired by the time of the 2014 snapshot (1.2). This explains why the acquisition rate looks strong for founders in 1999 (22.4% across all non-empty rows) but almost disappears for those in 2013 (0.6%). Younger companies simply haven't had five years yet. If I had more data, I would join an acquisitions table with deal dates, redefine the target as "acquired within five years," and exclude anyone who hasn't reached that age.

The timing issue also shows up in the data split. The rates drop from 16.0% in the training set to 13.6% in the validation set, and then to 9.7% in the test set (2.7). The `company_age` factor was fitted using ages 9 to 14, but then applied to ages 6 and 7. On the test data, the average predicted probability is a bit low (0.086 compared to 0.097) because the age coefficient over-corrects when used outside its original range. If I set all the test ages to 9, the average prediction jumps to 0.135, which is higher than the true rate (5.8). Next time, I would evaluate the model within age groups or use a time-based approach so the model doesn't have to guess about ages it hasn't seen.

For the companies we labeled, the funding figures aren't always clear at the time of pitching. The totals can include amounts raised after the acquisition (1.9). Even though the date-feature check didn't show a clear leak from the funding span, this doesn't change the amounts involved. The real explanation is that the round-level data was cut off at a hypothetical decision date, or at the acquisition date for positive entries, so all the information would have been available to the boutique when choosing which companies to approach.

In the modelling process, the design matrix contains a large number of near-duplicates. The condition number of X’X/n is approximately 7,300 and the correlation between has_angel and log_angel is about 0.996. As a result, the individual coefficients become less useful and ordinary gradient descent has to take 100,000 steps to achieve what sklearn achieved in 5.7. Logistic regression is also unable to detect interactions such as hub × sector unless I include them myself. I should therefore remove or combine the redundant funding features, after which I need to try a small tree or a gradient-boosted model, or add a few explicit interactions, and then examine the calibration and precision@5%.

There are a few limitations when it comes to the evaluation. The validation is limited in that increasing the number of hits raises precision@5% by approximately 0.011, and the improvement in ROC-AUC for the date feature remained within the noise range (ranging from −0.003 to +0.021), although it was near the boundary. The highest value of precision@5% is 0.330 when the date features are not included and 0.455 when they are. This corresponds to about eleven additional acquirers among the 88 names in the top 5%. Although this may seem important, the precision@5% figure for the 88 companies is highly noisy (a single hit causes a change of 0.011). It wasn't the main factor on which the decision was based, since the keep/drop rule was established prior to looking at the results, and the leakage risk from 1.9 is still present. The euro figures also rely on assumed values for the pitch cost, the fee, and a 25% win rate. These figures assess the selection of k, not the ranking itself. I would bootstrap the ranking metrics and, for a real boutique, use their actual costs or keep cost as a sensitivity slider on the dashboard.

In the end, the data exported is outdated and flawed. Since it is based on a 2014 snaIn the end, the exported data is outdated and flawed. It is based on a 2014 snapshot from Crunchbase, and the funding landscape has changed since then. In the dataset, about one hundred companies had funding recorded 'before founding' (this was set to zero), 22% of the original entries had no founding year (these were removed), and the markets were manually assigned to sectors. The next step should be to refresh the data with a more recent dump that has better date information and to replace the manual sector classification with one used only in the training phase, either through supervised learning or embeddings.

## Generative AI use disclosure

I used Claude via Cursor for the code, it helped me write most of `train.py`, and summarize my interpretations on each step so that anyone running it can follow the process, it also built the Gradio dashboard, with my input decisions and changes, and it reviewed my drafts of this report against the pipeline and against any grammatical error. 

