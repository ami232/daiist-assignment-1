# Assignment 1 Report

- **Name**: Jimena Sanchez 
- **Student ID**: 20104
- **Email**: jsanchezcurt.ieu2022@student.ie.edu
- **Group**: PPLEDBA 5A

## Dataset

The data set is a compilation of the monthly barometers published by the CSI in spain, which is public, anonymised microdata. They range from April 2025 to September 2026 and cover 16 studies total (CIS does not publish in August). In data/ there is a file called 'build_dataset.py', which extracts the barometers from the raw SPSS file format they publish in. 
In our data, one row equals one interviee who recalls voting for PSOE in in the 2023 elections. After the cut, we have 18.503 rows along 39 columns. 

There was quite heavy feature engineering in order to reverse methdlogical questionable decisions by the CIS, especially regarding their sampling distributions, which creates significant biases that are someone even worsened during later weighing. In my work I had to become quite familiar with these methdological decisions, which made creating a feature engineering pipeline to reverse them was a good yet realistic exercise. 

- **Train db**: April 25' to Jan 26', 9 barometers, 8.741 rows and a 23.7% churn rate
- **Validation db**: Feb 26' to April 26', 3 barometers, 3.081 rows and a 24.5% churn rate
- **Test db**: May 26' to Sep 26', 4 barometers, 4.172 rows and a 24.9% churn rate

There are 2.509 of undecided voters (rows) across al months that are allocated, never trained on.

## Business / real-life framing

Every month the CIS asks a sample of Spaniards how they’d vote with the intention to, amongst other things, estimate voter intentions and predict election outcomes. This is not different from how other polling firms operate, but the CIS makes them public, which eliminates the cost of our analysis by not having to conduct interviews ourseleves.

Regardless, political analysts use this type of data to understadn voter currents (voters moving from party to party). We focus on those who voted for the current governing party in 23J. Some continue declaring their intention to vote PSOE in 2 months, others declare a change of party and the remaining admit not having made a decision yet. With the elections closer and closer, a key questions arises for the PSOE: for each undecided 2023 PSOE voter, do we count them as going back to PSOE, or not?

We can answer that question by learning from the people who did answer. It does this by asking: Among 2023 PSOE voters who gave an intention, what separates those who stayed from those who left? The idea is to then apply that pattern to the undecided

## Data preparation & feature engineering

The CIS consistently scores each party very differently from the rest of the sector, this was my main incentive to investigate their data lineage from the micro data to the published estimate. During this process, I found a key flaw with the data that came from the sampling design. In order to extrapolate to the Spanish electorate, the sample must resemble the electorate for an extrapolation to be methodologically defensible. However, if one compares the variable 'Recuerdo de Voto', that distribution differs significantly from the official results of the 2023 election. Thus adjustment must be made in order to extract a usable subset of the data that minimizes this effect. Another big inconvenient with data was the fact that some questions reuse certain codes with different meanings across different windows. That is why I had to translate each study with its own labels and recode everything into one single table.

The adjustment is done incrementally, using raking these axis: recall of 23J vote, sex x age and education. Each axis rewieghts people who have a category in it. The change is capped at 0.2 to 5 times the CIS weight, weights are rescaled to mean 1, and the loop stops wwhen every target is met within 0.05 points (max 50 rounds). Moreover, the target is defined by the lasta vailable data at that date to simulate a running product. There are limits to my correction. For example, only after all of this are the 2023 PSOE voters kept, so their weights still differ by sex, age and education.

To understand the results fo the raking, i investigated further. All 16 barometers converge in 3-4 rounds and no wight actually hits the cap, and PSOE does to 31.7% in every edition. However, this took a toll on precision, as the design effect went up from 1.9 to 2.5, so we get about half of each borameter's interviews as our actual usable data. I wanted to add a quick note to notice two data errors and study 3528 which has no PESO variable, making it fall into the regional version of the variable

Regarding the weights and when they are used, every number that describes the population is weighted: actual and estimated
churn by month, the pollster's monthly error, the threshold chosen on it, the bootstrap intervals, and the
final allocation of the undecided voters. The models are fitted unweighted, the demographic axes of the correction are
already features and recall is the same for every row. A weighted fit is technically better on the
validation monthly error (1.50 against 1.77 pp) but worse on the weighted validation log-loss (0.2823 against 0.2792).

Table with the Churn among declared voters by month, three ways (%):

| Month | Raw | CIS weight | Corrected |
|---|--:|--:|--:|
| Apr 2025 | 19.6 | 18.1 | 17.7 |
| May 2025 | 17.1 | 18.3 | 18.5 |
| Jun 2025 | 21.1 | 20.6 | 20.9 |
| Jul 2025 | 31.4 | 31.6 | 31.9 |
| Sep 2025 | 24.4 | 20.2 | 20.8 |
| Oct 2025 | 25.2 | 26.4 | 26.9 |
| Nov 2025 | 23.2 | 21.0 | 21.5 |
| Dec 2025 | 25.9 | 23.0 | 23.1 |
| Jan 2026 | 24.7 | 25.6 | 25.6 |
| Feb 2026 | 26.0 | 26.1 | 26.5 |
| Mar 2026 | 26.7 | 28.0 | 28.2 |
| Apr 2026 | 21.2 | 21.2 | 21.6 |
| May 2026 | 20.6 | 19.1 | 19.6 |
| Jun 2026 | 26.2 | 24.2 | 25.0 |
| Jul 2026 | 26.8 | 24.5 | 25.0 |
| Sep 2026 | 26.1 | 25.3 | 25.9 |

Although the correction is justifiable, it could potentially shrink the originally over-represented recallers as a group without tell which of them did not actually vote PSOE. Moreover, there is a slight mismatch between source and target that although it does not greatly affect results must be declared. The population targets include foreign residents which are not considered in the CIS interviews, as they only interviee spanish nationals. 

I tried a couple of things here that I ended up discarding. For example, k-means with a k set to 3. It returned a weak silhouette score of 0.28, so it wasn0t worth persuing further

## Modeling: three implementations, one model

The model I chose was a logitic regression. The reaosnsing was: the decision needs a probability per voter with which I could do two things. First, I could thershold it into an allocations (is he/her back to PSOE or not?). Secondly, I could avergae into an expected share. 

All 3 models have the same objective, and all three use the same 81 features from the same fitted preprocessor, the same training months, and float64. Furthermore, the only tuning made was for C, whcih was chosen on validation log loss of C = 0.1 and lamda = 5.72e-4 on the 8,741 interviews. Here are the different results i saw

| C | 0.001 | 0.01 | **0.1** | 1 | 10 |
|---|--:|--:|--:|--:|--:|
| Validation log-loss | 0.3412 | 0.3033 | **0.3004** | 0.3021 | 0.3026 |

The learning rate was chosen directly over the trianing objective instead of simple validation. The problem inoftself is convex, so the C is what fixed where the optimum is and the lr only decides whether the 10.000 steps reaches it. I initially made the mistake os chosing C on validation, but I ended up accidentaly rewarding an early stop as a sort of hidden extra regularisation layer I wasn0t counting on. 

| Learning rate | 0.1 | 0.5 | 1 | **2** | 4 |
|---|--:|--:|--:|--:|--:|
| Objective minus scikit-learn's | 1.4e-05 | 3.3e-07 | 5.5e-09 | **1.4e-12** | 1.3e-02 |

lr = 4 does not converge. The conservative bound for gradient descent is lr < 2/L = 1.66, and lr = 2 still
converges because that bound is a worst case.

**Threshold.** 

I decided a voter is counted as leaving PSOE if p > t, where t minimises the mean absolute monthly error of
the estimated churn rate (weighted with the corrected weights) on the validation months. I found t = 0.29 to be the best for all three. The usual 0.5 would be far worse for a published number, even though it scores better on accuracy. At 0.29 the two errors almost cancel, which is what I needed. At 0.5 the model misses three times as many leavers as it wrongly flags, and the estimate falls far short.:

| Validation | False positives | False negatives | Accuracy | Monthly error |
|---|--:|--:|--:|--:|
| t = 0.29 | 221 | 206 | 86.1% | 1.77 pp |
| t = 0.50 | 76 | 318 | 87.2% | 7.63 pp |

**test months** 

with this complements:

Bootstrap 95% intervals (1,000 resamples within each month): AUC 0.903 to 0.924; monthly error 0.88 to 3.22 pp for the model and 6.59 to 10.31 pp for the closest-party rule


| Method | AUC | Log-loss | Brier | Threshold | Est. churn | Monthly error | Accuracy |
|---|--:|--:|--:|--:|--:|--:|--:|
| scikit-learn | 0.914 | 0.2987 | 0.0902 | 0.29 | 25.6% | 1.76 pp | 86.6% |
| Manual PyTorch loop | 0.914 | 0.2987 | 0.0902 | 0.29 | 25.6% | 1.76 pp | 86.6% |
| nn.Module + optim | 0.914 | 0.2987 | 0.0902 | 0.29 | 25.6% | 1.76 pp | 86.6% |
| Model, mean of p (no threshold) | 0.914 | 0.2987 | 0.0902 | | 23.2% | 1.02 pp | |
| Naive: training churn rate (22.9%) | 0.500 | 0.5621 | 0.1873 | | 22.9% | 2.58 pp | |
| Rule: closest party is PSOE | | | | | 32.2% | 8.47 pp | 76.2% |
| All undecided back to PSOE | | | | | 0.0% | 23.84 pp | 75.1% |

Actual churn in the test months: 23.8%. . The rule's error minus the model's is 4.70 to 8.44 pp, so the model beats the rule clearly. It also beats the naive baseline on the point estimate (1.76 vs 2.58 pp), but the model's interval contains 2.58, so with four test months I can't claim that difference is firm with as much confidence. .

| Test month | Interviews | Actual | Model (t = 0.29) | Mean of p | Rule | AUC |
|---|--:|--:|--:|--:|--:|--:|
| May 2026 | 1,078 | 19.6% | 20.9% | 20.3% | 28.2% | 0.918 |
| Jun 2026 | 1,023 | 25.0% | 27.3% | 24.0% | 33.3% | 0.925 |
| Jul 2026 | 1,098 | 25.0% | 27.5% | 24.1% | 32.1% | 0.889 |
| Sep 2026 | 973 | 25.9% | 26.7% | 24.4% | 35.7% | 0.922 |

Interestingly enough, validation and test disagree on one decision. On validation the threshold beats the plain average of the
probabilities (1.77 and 2.91 pp), so that is what I chose. On test it is the other way round (1.76 vs 1.02 pp): the thresholded estimate is too high in all four test months, the average slightly too low in three. I kept the threshold because it is what validation picked, and choosing again on test would be fitting to it. The gap between the two is also a **calibration** (like we saw in class) problem: on validation the model under-predicts in the middle deciles (predicted 15.7% where 25.3% left; 29.2% where 36.4% left).

**Do the three agree? YES.**

| Pair (test months) | Max diff. in p | Mean diff. in p | Max diff. in coefficients | Same allocation |
|---|--:|--:|--:|--:|
| scikit-learn vs manual PyTorch | 2.6e-06 | 1.4e-07 | 7.2e-05 | 100% |
| scikit-learn vs nn.Module | 2.6e-06 | 1.4e-07 | 7.2e-05 | 100% |
| manual PyTorch vs nn.Module | 5.7e-12 | 1.0e-12 | 1.1e-11 | 100% |


The two PyTorch versions are essetnially  the same computation written two ways, so it makes sense they agree to 1e-11. Them the gap to scikit-learn is small but not zero, so I checked where it is and it all sits in one direction:

| Parameters | PyTorch minus scikit-learn |
|---|--:|
| Intercept | +7.2 × 10⁻⁵ |
| Each one-hot block, summed (closest party, age, education, religion, work, class, region, marital status) | −4.0 × 10⁻⁵ each |
| Largest of the other 24 coefficients | 3.3 × 10⁻⁶ |

I noticed something interestinf. With one-hot encoding, each category block adds up to 1 in every row (just like the model’s intercept), creating a redundancy. This means you I could increase every category coefficient by 1 and decrease the intercept by 1 without changing any predictions. Because the data cannot distinguish between these different coefficient combinations, the model has no unique solution in that specific direction. L-BFGS finds a solution efficiently, while gradient descent moves very slowly because the objective function is almost flat in that direction. This explains why their coefficients differ slightly even though their training objectives and predictions are practically identical. I then learnt that using drop="first" solves the redundancy by removing one category.

As I said before, the models are fitted unweighted, because the demographic axes of the correction are already features. A weighted scikit-learn fit, with its own threshold chosen on validation, does better on the validation monthly error (1.50 vs 1.77 pp) but worse on weighted validation log-loss (0.2823 vs 0.2792). The evidence is mixed, so I kept the unweighted fit.

## Limitations & next steps

The first important limitation is that the undecided voters are not the same from the people who already declare their voting intention. For example, undecided voters give Sánchez a lower rating (4.82 and 6.87) and are slightly more right-wing in their ideology (3.90 and 3.27). Essentially, the model is being applied to a group that is different from the one it was trained on. With the available data, it is not possible to know whether the model correctly predicts how these people will eventually vote. To improve this, I would use a panel survey that interviews the same people over time, or the CIS post-election study, to compare their previous answers with their actual reported vote.

Another limitation is that the data does not follow the same individuals over time. Different people are interviewed each month, so it is not possible to observe how a person’s voting intention changes. In addition, the natural limitations of a survey. We assume people are declaring the truth, but implement no assurance.

A further, more to-real-life limitation is that voting intention is not the same as actual voting behaviour. The model predicts what people say they intend to do months before an election, but their preferences may change right before election day. Therefore, even a model that performs well on voting intention might not predict the final result accurately. To address this, I would test the model against actual election results and post-election surveys when these become available.

The data is also relatively short. Each barometer contains many interviews, giving us about 18.000, but distributed across the small number of 16 barometers. For example, we can only evaluate on 4 months. With more data, I would test the model on a longer period, including more months and, ideally, different electoral periods, to check whether its performance remains stable over time.

Another issue is calibration, which we discussed in class. The validation results i get in in train.ipynb show that the model underestimates the probability of voting intention in the middle deciles. This helps explain why the estimate that uses the model’s predicted probabilities performs worse than the estimate that applies a threshold. In other words, the predicted probabilities do not always reflect the observed rates accurately. I would address this by testing calibration methods and checking whether they improve the estimates on separate validation data, rather than choosing a method based only on its performance on the training data.

Finally, the models are trained without survey weights, while the allocation table uses the PESO variable to account for differences in respondents’ representation. This means that the model training and the final estimates do not treat the sample in exactly the same way, although similarly enough. With more time, I would test whether including survey weights during training improves the results and compare the weighted and unweighted models on the same test data. This would help determine whether weighting improves the model’s ability to estimate voting intention.


# AI Disclosure

I used Claude Code as resource, it was ofr:

1. Create the dashboard based on my design. I dicatted in detail each tab and questions it should answer as well as aked it to implement all of the visual parameters it has learnt from me. 
2. Feature engineering: i gave it an 'article' I wrote investigating the methodology for the cis in the last years and i had a applied a more sophisticated 'reverse' feature engineering task based on a markdown file i had developed for an unrelated project. With this context I created the train notebook.
3. a couple of times i did verificiations where i gave it access to my directory and to the instructions and asked it how alligned it was and what could make me more alligned. 

I also used copilot completion, mostly for the app notebook and side chat to fix errors and bugs i got along the way.

Finally, I used chatgpt for comment addition to make it easier to understand changes made. 