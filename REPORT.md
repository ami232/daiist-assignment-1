# Assignment 1 Report

- **Name**: Carlota Navarro Calle
- **Student ID**: 19992
- **Email**: cnavarro.ieu2022@student.ie.edu
- **Group**: BBADBA 5A

## Dataset

This dataset supports predicting student dropout and academic success in higher education.

**Source**: I found it on Kaggle:
https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
Originally collected for a project funded by Portugal's SATDAP program: https://zenodo.org/records/5777340#.Y7FJotJBwUE

**Unit of observation**: one row represents one student enrolled in an undergraduate degree at a Portuguese higher education institution. Features span five groups:
- enrollment-time demographics: marital status, nationality, gender, age, international/displaced status, educational special needs
- academic path at application: application mode, application order, course, daytime/evening attendance, previous qualification
- socio-economic/family background: parents' qualifications and occupations, scholarship status, debtor status, tuition payment status
- per-semester academic performance: units credited, enrolled, evaluated, and approved, plus grades, for both the 1st and 2nd semester
- macroeconomic context at enrollment: unemployment rate, inflation rate, GDP.

**Size**: 4,424 rows, 35 columns (34 features + target). No missing values.

**Target**: Originally a three-class categorical field (Dropout / Enrolled / Graduate). I have redefined this (see Business Framing below).

**Why this dataset**: I'm building a business centered on learning how to learn so understanding why certain students succeed or struggle in higher education is directly relevant to what I want to do professionally. Any insight this dataset surfaces about which factors predict dropout vs. completing higher education is something I can potentially apply to helping students beyond this assignment.

## Business / real-life framing

The hypothetical business problem is to help a university identify students who are at risk of dropping out. At the end of the first semester, the university's student support team could use the model to identify students who may benefit from additional academic or financial support. The model is intended as an early warning tool to help staff decide where to focus limited support resources.

The original `Target` column has three classes (Dropout / Enrolled / Graduate). However, I defined the target as a binary classification problem. Students whose eventual outcome is Dropout are assigned a target of 1, while students whose eventual outcome is Graduate or Enrolled are assigned a target of 0. I did this because I want my model to focus on finding students that are at risk of leaving, not on if they will graduate this specific year.  

Because the prediction is made at the end of the first semester, the features used by the model should only contain information that would be available at that point. In particular, information from the second semester should not be used as a predictor, because it would not yet be available when the intervention decision is made. Removing the second semester features ensures there is no data leakage.

A time-based train/test split would be appropriate if the dataset contained reliable dates that allowed the model to be trained on earlier students and tested on later students. The macroeconomic columns (Unemployment rate, Inflation rate, GDP) show that such structure does exist as each has only 9-10 unique values across 4,424 rows, suggesting students were enrolled across a small number of distinct periods rather than continuously. However, the dataset does not provide an explicit date or cohort label, so these periods cannot be reliably ordered or separated into a time-based split. Therefore, a stratified random train/test split is used instead, so that the proportion of dropout and non-dropout students is kept similar in both sets. This is a limitation of the dataset as a real deployment would ideally use a time-based split to validate that the model generalizes to future cohorts, not just unseen students from the same ones.

Given the business framing, recall on the Dropout class is the metric that should drive the decision threshold, since a false negative (missing an actual at-risk student) is far more costly than a false positive (an unnecessary advisor check-in). The Gradio dashboard
lets a user move the threshold interactively and see how the confusion matrix and an estimated business cost change in response. The costs have been calculated assuming an average annual tuition of €5,000 and 2 remaining years lost per dropout so a missed at-risk student (false negative) costs the institution roughly €10,000 in lost tuition. Assuming an advisor's time costs €30/hour and a check-in takes 1 hour, a false alarm (false positive) costs roughly €30. This gives a cost ratio of approximately 333:1, reflecting that missing an actual dropout is significantly more costly than an unnecessary check-in. Therefore, minimizing total cost strongly favors a low threshold that catches nearly all dropouts. However, as discussed in the limitations section below, pushing the threshold to the optimum produces an impractical number of flagged students.

## Data preparation & feature engineering

**Leakage prevention**: All 2nd-semester curricular columns were dropped, since the model predicts after only the 1st semester (see Business Framing).

**Rare category grouping**: Several categorical columns had many categories with very few students each (e.g: Nationality had 21 unique codes, several with only a handful of students). Encoding these separately would have produced mostly empty dummy columns while significantly increasing the feature count. Therefore, categories with fewer than 30 students were grouped into a shared "Other" bucket before one-hot encoding, reducing the feature count from 234 to 120 columns.

**Engineered features**, each justified by a specific finding from EDA:
- sem1_pass_rate (units approved / units evaluated): show how well a student performed among units they actually sat exams for. Non-dropouts passed 71% of evaluated units on average vs. 33% for dropouts.
- financial_risk_score (count of: has debt, tuition overdue, no scholarship): dropout rate rose almost perfectly with this score with 9.3% at 0, 29.3% at 1, 69.2% at 2, 88.8% at 3 suggesting financial pressure is a major driver of dropout.
- has_zero_grade (binary flag for a 1st-semester grade of exactly 0): the grade distribution showed a large spike at 0 specifically among dropouts (40.1% of dropouts vs. 4.9% of non-dropouts).

**Scaling**: Numeric features were on very different scales (e.g: Age: 17-70, Inflation rate: small decimals). Since logistic regression and the PyTorch implementations optimize via gradient steps, large scale features can dominate training. StandardScaler was fit only on the training set and applied unchanged to the test set, to avoid data leakage.

**Missing values from engineering**: sem1_pass_rate produces undefined values for students with 0 evaluations (division by zero). These were filled with 0, since 0 evaluations reasonably implies a 0% pass rate.

**A feature considered and dropped**: an "evaluations minus enrolled" gap was explored but dropped. On average, both groups showed more evaluations than enrolled units possibly due to exam retakes, though the data dictionary doesn't confirm this and the gap was similar between dropouts and non-dropouts, which made the feature's meaning too ambiguous to include.


## Modeling: three implementations, one model

Model: logistic regression, since the target is binary (Dropout vs. Not-Dropout).

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive baseline (majority class) | 0.679 | 0.000 | 0.000 | 0.000 |
| scikit-learn | 0.841 | 0.799 | 0.673 | 0.730 |
| Manual PyTorch loop | 0.840 | 0.803 | 0.662 | 0.726 |
| Standard PyTorch (nn.Module + optim) | 0.838 | 0.808 | 0.651 | 0.721 |

All three trained methods outperform the naive baseline, which by definition catches zero actual dropouts (0% recall) confirming the models are learning and not just exploiting class imbalance.

**Do the three agree?** Yes, all three models converge to essentially the same result, within about 1 percentage point of each other on every metric. Initially this was not the case as I trained the manual pytorch for 1000 epochs, which underperformed scikit-learn by 2-4 points across all metrics. Increasing to 5000 epochs closed nearly the entire gap, confirming this was simply a matter of training time.

## Limitations & next steps

**No reliable time-based validation**: the dataset shows evidence of multiple enrollment cohorts (via the narrow range of macroeconomic values) but provides no explicit date/cohort label, so a random split was used instead of a time-based one. With access to real enrollment dates, I would validate the model on a future cohort to confirm it generalizes beyond the specific years represented here.

**Cost-based threshold is impractical at the optimum**: because a missed dropout is weighted 333x more heavily than a false alarm, minimizing total cost pushes the optimal threshold toward flagging nearly every student as at-risk which isn't operationally useful since a student support team has limited capacity. A more realistic approach would cap the number of students flagged (e.g: the highest-risk 20%) or build a capacity constraint directly into the cost function, rather than purely minimizing cost.

**Model is linear only**: logistic regression assumes a linear relationship between features and log-odds of dropout. However, some relationships in the data may be more complex than a linear model can capture. Therefore, a tree-based model (e.g: random forest or gradient boosting) could potentially improve recall further, at the cost of losing the coefficient interpretability that this project currently benefits from.

**Only one semester of data was used**: predicting after 1st semester was a deliberate choice to keep the problem realistic for early intervention, but it does mean the model has less information than a model using the full first year. A real deployment might offer both an early (1st-semester) and a later (full-year) model.

## Generative AI use disclosure

I used Claude throughout this assignment for: explaining unfamiliar concepts (e.g., the difference between application mode and application order, what each curricular-units column represents, what pandas/sklearn/PyTorch errors meant and how to fix them) and help with code for saving models, and building the Gradio dashboard structure. All business framing decisions (prediction timing, target definition, split strategy, threshold metric, cost ratio) and feature engineering choices were my own, based on EDA I ran and interpreted myself. I did not use AI to generate the analysis, findings, or conclusions reported above.

