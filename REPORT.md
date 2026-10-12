# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Eva Yang Garrudo Salvador
- **Student ID**: 20207
- **Email**: egarrudo.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

Predicting Two-Year Recidivism Using Logistic Regression: A COMPAS Case Study

1. Dataset

The project is based on the COMPAS dataset which was made available by ProPublica as part of its inquiry into racial disparities in criminal risk assessment. The dataset includes details about defendants in Broward County, Florida, such as their demographic characteristics, criminal history, the charges brought against them, and their subsequent recidivism. I chose this dataset since my interest lies in algorithmic fairness and because of the COMPAS case that I examined at the University of Maryland.

The records in each row refer to a defendant's screening history, and the target variable, two_year_recid, shows whether or not the person was found to have reoffended within a two-year period. The original file had 7,214 records and 53 columns; after applying the eligibility filters, 6,172 records were left. I then removed 875 of the records which were screened on or after April 1, 2014, so that 5,297 records remained for the final analysis.

Since the source file had already undergone preprocessing, there was no need for the initial filtering to go through the whole original data-cleaning process. Specifically, I deliberately omitted the addition of a filter using is_recid, since the condition is_recid == -1 yielded no rows. The other eligibility filters then produced the expected subset of 6,172 rows.

The dataset is appropriate for this project since it includes a well-defined prediction target together with some important ethical issues. It is possible for a model to obtain satisfactory predictive performance even if it results in unequal errors among different demographic groups.

2. Business / Real-Life Framing

The proposed application is a decision-support system for the question of pretrial release. It calculates the probability that a defendant will commit a further offence within two years and thus provides information which could be taken into account together with all the other evidence when evaluating the risk during pretrial detention. It is not meant to make independent decisions regarding detention or release.

The two_year_recid variable was used as the target, it being a binary variable. Only information that was available at the time of the screening decision was included as a predictor. In order to prevent information leakage or the use of variables that were not available at the time of the decision, any outcome-related variables, existing COMPAS scores, identifiers, race, and details regarding when someone was sent to jail or released were excluded. Age and sex were included, whereas race was kept aside for the purpose of fairness auditing.

A temporal split was required since the intended application was meant to predict future outcomes. The records were split according to time into training, validation, and test sets in a 60/20/20 ratio, making sure to respect the date boundaries. The training set was used for fitting the models, the validation set for choosing the regularisation and the decision threshold, and the test set for assessing the final system.

At the beginning, a temporal split showed that the recidivism rate among the test set was unusually high at 81.7%, as compared to 36.6% in the training set. Monthly checks indicated that cases where individuals did not reoffend became very uncommon from April 2014 onwards. This led me to believe that the more recent records might have incomplete two-year follow-up, even though this explanation had not been independently verified. Because of this, I decided to exclude any records that were screened from April 1, 2014, onward. As a result, the recidivism rates became approximately 37.0%, 35.9%, and 35.9% for the training, validation, and test sets, respectively.

It should be noted that the monthly diagnostic looked at the outcome labels throughout the whole observation period, including those records which in later stages would be part of the test set. Even though the final chronological division was kept, the choice to exclude the most recent records was based on information from the entire dataset. This means that there is some degree of data selection influenced by the test set, and this point should be taken into account when viewing the evaluation as a completely independent assessment.

The threshold for decision-making was determined on the basis of a maximum false positive rate (FPR) rather than attaching monetary values to the errors of prediction. A false positive arises when the model predicts that a person will reoffend even though they do not reoffend during the observation period. In a pretrial context, such mistakes could lead to decisions that are unnecessarily restrictive and thus affect an individual's liberty. The maximum allowable FPR was set at the level of the COMPAS validation set's FPR of 26.5%. This served as a benchmark against which recall could be compared at a comparable error rate, not because the threshold is inherently fair or ethically ideal.

The main metric, which depended on a threshold, was recall at the chosen false positive rate limit, in this way showing how many actual reoffenders were identified while keeping the number of false positives under control. ROC-AUC was included as well since it assesses ranking performance at all thresholds. A majority-class classifier was used as the simple baseline.

3. Data Preparation & Feature Engineering

The emphasis in data preparation was on creating features that were interpretable, using only the information available at the screening stage. Eligibility filters were applied to exclude cases in which the arrest had taken place more than 30 days before or after screening and cases where the charge degree had been recorded as "O". Since the condition is_recid == -1 yielded no rows, no further records were eliminated on that basis. The final dataset, consisting of 6,172 rows, corresponded to the expected ProPublica-filtered subset.

The exploratory analysis and the design of features were carried out on the training data in order to avoid the test set affecting the definition of the features. Where appropriate, the continuous or those with a high number of categories were transformed into categories that were interpretable.

Age was divided into four groups: under 23, 23 to 30, 31 to 45, and 46 and over. The number of previous offences was divided into the categories of 0, 1 to 2, 3 to 5, and 6 or more. In the training set the recidivism rates showed a monotonic pattern among the groups, therefore no further merging of categories was needed.

The descriptions were grouped according to rule-based categories that took into account their meaning and how frequently they occurred, not on the basis of the recidivism rates observed. This approach promoted interpretability and helped to lower the chance of defining the categories with the target in mind. The common description 'arrest case no charge' was given its own group, and missing charge information was placed in the 'other' category.

The juvenile history was indicated by a binary has_juvenile variable rather than by a count. In the training set, the rate of recidivism was 61.8% for people with a juvenile history and 33.5% for those without it. Even though this feature was informative, the use of the binary representation fails to capture the differences in the number or kind of juvenile offences.

The variables related to time were kept as indicators and bins. Yet the exploratory analysis indicated that their predictive value was limited: the recidivism rates linked to c_days_from_compas differed by just 1.9 percentage points, and the positive category for days_b_screening_arrest included only 36 records. In the end, two of the features had coefficients close to zero, about -0.045 and +0.038, which means that they made only a small contribution to the final model.

Standardisation was not carried out since the engineered features were mainly binary indicators or categorical encodings. As a result, further scaling was not regarded as necessary for this implementation.

4. Modeling: Three Implementations, One Model

Logistic regression was chosen since the target variable is binary and the model provides estimated probabilities together with interpretable coefficients; this is advantageous in a high-stakes situation where it is important to know how the predictors relate to the model's outputs. L2 regularisation was used in order to reduce overfitting.

The same logistic-regression specification and test set were used to compare three implementations:

1. Scikit-learn: Logistic regression using the standard library implementation.
2. Manual PyTorch: Logistic regression trained using a hand-written, full-batch gradient descent loop.
3. Standard PyTorch: Logistic regression implemented with torch.nn.Module and trained using torch.optim.Adam.

The manual PyTorch implementation employed full-batch gradient descent, which meant that the gradients were computed using the whole training dataset for each update; it did not use stochastic gradient descent, since stochastic gradient descent would have updated the parameters based on individual observations or smaller batches. The standard PyTorch implementation used Adam, an adaptive gradient-based optimiser.

The value of the regularisation parameter was determined by means of validation data; C was taken as 0.1 since C = 0.01 worked much worse, whereas C = 0.1, 1 and 10 all gave essentially identical results. The learning rate for the gradient-based methods was set at 0.1.

In the case of the PyTorch implementations, the L2 penalty was calculated with the use of weight_decay = 1/(C·n), where C represents the inverse regularisation strength employed by scikit-learn and n is the number of training observations. The reason for dividing by n is that the loss is given as an average over the training observations, while the regularisation convention in scikit-learn is different. This modification is intended to ensure that the effective regularisation strength is comparable between the various implementations. However, since the exact equivalence relies on the loss reduction, the objective-function convention, and the particular optimiser implementation, matching only the formula does not guarantee that the coefficients will be identical.

Test-Set Results


| Model | Test ROC-AUC | Test recall | Test FPR |
|---|---|---|---|
| Majority-class baseline | 0.500 | 0.000 | 0.000 |
| scikit-learn | 0.729 | 0.580 | 0.229 |
| Manual PyTorch | 0.729 | 0.580 | 0.229 |
| Standard PyTorch | 0.729 | 0.580 | 0.229 |
| COMPAS reference (decile ≥ 5) | 0.690 | 0.530 | 0.237 |


The three implementations matched each other when their test metrics were rounded to three decimal places, with ROC-AUC values of 0.729 as compared to 0.690 for the COMPAS reference. This shows better ranking performance in this evaluation, even though it does not prove that the model would improve actual pretrial decisions.

The discrepancy in the initial coefficient between the scikit-learn and the PyTorch implementations was examined by reducing the convergence tolerance. The remaining absolute differences were about 8.6 × 10⁻⁶ and 7 × 10⁻⁵, both of which were below 10⁻⁴. This showed that the two implementations had converged on very similar solutions and that the previous discrepancy had been due to the convergence settings rather than to a fundamental difference in the model.

When the threshold was selected, the proposed model obtained a test recall of 0.580 with a false positive rate of 0.229, as compared to COMPAS's reference recall of 0.530 and a false positive rate of 0.237. It thus managed to identify a higher proportion of actual reoffenders while having a slightly lower overall false positive rate in this instance. Yet the bootstrap confidence intervals contained zero for the difference in performance, and the uncertainty resulting from the choice of threshold on the validation data was not completely taken into account. For this reason, the results can only be regarded as suggestive and not as conclusive evidence of an improvement over COMPAS.

A majority-class baseline serves as a simple point of reference when evaluating whether the model performs better than simply always predicting the most common class.

Fairness Audit

Race was not included during the training of the model but was kept for evaluation after the fact. In this way, I was able to check whether eliminating race as a predictor would stop disparities in errors among different racial groups.

The audit of the test set found that it did not. For African-American defendants the model's false positive rate was 31.6%, for Caucasian defendants 18.7%, and for Hispanic defendants 10.3%. The figure for the Hispanic group was based on just 68 people who did not reoffend and so should be treated with caution. In some racial groups the samples were too small to allow reliable estimates: the test set included only four Asian individuals and no Native American individuals, and these groups were therefore regarded as having sample sizes that were too small for meaningful comparisons.

The audit also showed some differences. While the model had a false positive rate of 27.7% for men and 10.3% for women, COMPAS had rates of about 23.4% and 24.3% respectively. The model included sex through the is_male variable, which had a positive coefficient of +0.302. With all the other predictor variables held constant, this positive coefficient shows that being male was linked to higher predicted odds of reoffending in the model. However, the coefficient does not prove a causal connection and does not account for the whole difference in the false positive rates for the various groups.

It is clear that omitting race from the features does not ensure that the error rates are equal; other factors, such as sex, criminal history, and the nature of the charge, can lead to disparities among different groups. It is an ethical and policy issue whether such differences should be acceptable in a pretrial situation, and this question cannot be answered simply by referring to predictive performance.

5. Limitations & Next Steps


Initially, the distribution of the outcomes varied with time and the training and test sets were not the same. For instance, the percentage of cases which involved violent offences rose from about 26% in the training set to 37% in the testing set. This could have an impact on performance since cases involving violent offences may differ in both their predictors and outcome rates. To look into this I would compare the feature distributions in the different temporal splits, report the performance for each type of offence, and assess the model on further consecutive test periods.

Secondly, there is a gap in the dataset for June and July 2013. This missing period could have an impact on the representativeness of the chronological split and might make it more difficult to tell whether any temporal changes indicate real shifts in the population or are simply due to gaps in data collection. If I had more time, I would look at the original records and the accompanying documentation to find out why those months are missing and to evaluate whether the gap causes systematic bias.

Thirdly, the exclusion that led to the recent record was based on a diagnostic which looked at the labels throughout the entire dataset, including the final test period. Even though the reason for the exclusion was a reasonable worry regarding incomplete two-year follow-up, the choice to make this decision fed information from the test period into the data-selection process. A more robust evaluation would determine follow-up eligibility from the dates when observations were recorded before splitting the data and would ensure that a completely independent test set remained for the final assessment.

Fourthly, the source file had already undergone preprocessing. This meant that it was not possible to examine the effects of the original data-cleaning decisions or to reconstruct the entire process from the raw records. The next version should include a record of the provenance of each field and, where feasible, reproduce the preparation procedure starting from the raw source data.

Fifth, the model was assessed in terms of ROC-AUC, recall, and false positive rates, but calibration was not measured. AUC shows how well the model ranks individuals, not whether the predicted probability matches the actual frequency of the outcome. As the model generates probabilities, I would subsequently evaluate its calibration using reliability plots and a measure such as the Brier score, both as a whole and within sufficiently large demographic groups.

Sixth, the coefficients of two of the engineered features were about -0.045 and +0.038. Since these make only a minor contribution, they increase the complexity of the model without offering much obvious predictive value; therefore, I intend to compare the present model with a simpler one in which these features have been removed, and use the validation data to see whether there is a meaningful change in predictive performance.

Seven, the fairness audit showed significant variations in the rate of false positives according to race and sex. The coefficient for is_male was positive and racial disparities still remained even when race was removed from the analysis. Moreover, the small sizes of some groups meant that the reliability of certain estimates was limited. Further research should involve the inclusion of uncertainty intervals for each group, sensitivity analyses based on different thresholds, and comparisons of various fairness criteria. In making these comparisons, it should be clearly recognised that fairness criteria can be in conflict and that no one metric can guarantee that a system is ethically acceptable.

In the end, the dataset includes defendants from a single county and a specific historical period, so its patterns cannot be expected to apply to other judicial areas or to current populations. It would be necessary to carry out external validation on more recent and independent data before the dataset could be considered for use in practice. More generally, the model predicts recorded recidivism rather than an independently observable measure of future offending, and the fact that it performs well does not prove that using these predictions would lead to better judicial decisions or that it would be ethically acceptable. If it were to be put into practice, it would require legal scrutiny, human supervision, clear communication of the uncertainties involved, and a careful examination of whether algorithmic risk prediction should at all influence decisions before a trial.

6. Generative AI Use Disclosure

I kept responsibility for the final decisions, the interpretation, and the report, using Claude and Grammarly as auxiliary tools during the project.

At every stage of the project Claude put forward alternative methods and outlined the advantages and disadvantages of each one. I looked over these suggestions and then decided on the final options regarding the dataset, the prediction scenario, feature engineering, the temporal split, the threshold-selection criterion, and the evaluation method. Claude also prepared the code for both train.ipynb and app.py, gave the Git commands required to manage the project, explained the relevant concepts, and assisted in spotting and fixing the coding errors. I used the explanations that Claude provided to understand how the code was implemented and the reasons for the various modelling choices.

I was in charge of making the project decisions, selecting among the alternatives proposed, examining and carrying out the implementation, interpreting the results, and assessing the implications of the findings for the pretrial scenario and the fairness analysis. I prepared the arguments and conclusions set out in this report and was responsible for the final version of it. I also checked the material that was assisted by AI and continue to be accountable for the report's accuracy and adherence to the course’s AI policy.
*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*


