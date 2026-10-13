# Assignment 1 Report

- **Name**: Sarah-Ghislaine Coste
- **Student ID**: 19863
- **Email**: sg.coste@gmail.com
- **Group**: BBADBA 5B

## Dataset

The dataset I chose is called "Predict Students’ Dropout and Academic Success". I got it from UC Irvine (UCI Machine Learning Repository). The data gives information about higher-education students in portugal, particularly whether they Graduated or Dropped out.

One row represents one student and in total there are 4424 rows. There are 37 columns, I listed the names below if interested. There are 36 features plus the target: personal and family background (age, nationality, parents’ education and jobs), application details (course, admission grade), financial status (scholarship, debtor, fees), academic results for the 1st and 2nd semesters, and three national economic indicators. The target column has 3 categories: Enrolled, Graduate, Dropout. 

Column Names:
Marital status, Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Previous qualification (grade), Nacionality, Mother’s qualification, Father’s qualification, Mother’s occupation, Father’s occupation, Admission grade, Displaced, Educational special needs, Debtor, Tuition fees up to date, Gender, Scholarship holder, Age at enrollment, International, Curricular units 1st sem (credited),, Curricular units 1st sem (enrolled), Curricular units 1st sem (evaluations), Curricular units 1st sem (approved), Curricular units 1st sem (grade), Curricular units 1st sem (without evaluations), Curricular units 2nd sem (credited), Curricular units 2nd sem (enrolled), Curricular units 2nd sem (evaluations), Curricular units 2nd sem (approved), Curricular units 2nd sem (grade), Curricular units 2nd sem (without evaluations), Unemployment rate, Inflation rate, GDP, Target.

I picked this dataset because I think although a lot of unis have resources for students who are struggling, they don't actually actively outreach to students that need it. Its usually just a general resource for whoever wants to use it (could be counseling, tutoring, etc...). At the end of the day that ends up costing unis money. Let's take IE for example, by the end of our first year, 5 students either dropped out one of their degrees (BBA or DBA) or dropped out of IE overall. And that's just what I remember from our class. In BBA, if I'm not mistaken, plenty of students left as well at the end of the first year for some reason or another. It impacted me one day when in first year a friend of mine told me that he was leaving IE because when he went to ask for help, the administration told him to his face "maybe IE isnt the place for you, there's nothing we can do to help you". I'm not sure of the exact number of people who drop out of IE per year, or even if it is significant. But it did shock me how little universities do to maintain their current student base. 

## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.*

Hypothetically, this model would be destined for the student-support office, or whatever equivalent name it has, so that they could run it at the end of the first semester, and determine which students they need to contact proactively to avoid them dropping out at the end of the year. The goal would be for universities to act PROACTIVERLY instead of REACTIVELY.

We would predict whether the student is a Dropout (=1) or a Graduate (=0). We would also need to remove any row with the "Enrolled" students because their final outcome is still unknown. We would also not need the 2nd semester grades, as we want the recommendations to come from the 1st semester

For the split, I was thinking of doing a random stratified split to make sure both training and testing get the same share of dropouts. Its not going to be time-based because we don't have a date column.

For the metric that should drive the threshold, missing a droupout matters way more than giving support to a student who doesn't need it, because it costs more than an unneeded meeting. Since IE’s data isn’t public, I used data from a Portuguese public university, so the scenario and costs below are based on that school. I assumed a missed dropout costs the university €1,500 (about two years of lost tuition) and contacting a student costs €300 (a short support plan with an advisor or tutor).

## Data preparation & feature engineering

I only kept students who had a known outcome (Graduates and Dropouts), and removed the Enrolled students (their outcome isn't known yet and so we cant really do anythign with them). So, Dropout = 1, Graduate = 0, and the 794 “Enrolled” students were removed, leaving 3,630 students (which represented 39% dropouts).

My idea for this model was for it to run after the 1st semester (the goal is to see which students are struggling the first semester, to identify and help them during the second on and avoid dropouts). This means that data from the 2nd semester is leakage. I removed the 6 columns with information about the second semester.

There were a few columns I was hesitant to remove. The first ones were the Fees columns: Debtors and Payments on Time. Students with unpaid fees dropped out 94% of the time, and debtors 76%. The dataset doesn’t say when these were recorded. If it was recorded after the student dropped out (and obviously by then was not paying fees), it could describe the dropout instead of predicting it. Since I couldn't check I left them out.

Another column I was hesitant to remove was Gender. On average, men dropped out more than women (56% vs 30%). I ended up removing it for 2 main reasons: 1. I didn't want a student to not receive help, or be less likely to receive help, because of gender, and 2. I believe that this difference between men and women is most likely due to other factos (for example on average men take harder courses then women (like engineering, maths, etc))

The last column I decided to cut out was the Nacionality column (Nationality). 98% of student were portuguese, which made the other 18 nationalities, who only represent 2% of students, have very few students for each one. We already have an "International" column, with Yes/No, so I kept that instead.

I also made some new features, particularly with the 1st semester data. pass_rate_1st_sem is the share of enrolled courses passed (3 of 6 says more than 3). passed_nothing_1st_sem flags students who passed nothing: 88% of them dropped out, vs 29% of the rest. A grade of 0 for these students means “no grade”, not “a bad grade”, and this flag lets the model tell the difference. no_units_1st_sem flags the 152 students who enrolled in no courses (51% dropped out). has_credits flags students with credits from previous studies; most have 0, so the yes/no matters more than the count.

For the split, I decided on a random 60/20/20 train/validation/test split, stratified so each part keeps the 39% dropout share (random_state = 42 so the split is the same every run). A time-based split wasn’t possible because there’s no year column.

As for preprocessing, category codes (course, parents’ jobs, etc.) are one-hot encoded, with codes that have fewer than 30 training students grouped together. Numeric columns are scaled to mean 0 and spread 1, and yes/no columns are kept as they are. All of these rules are learned from the training set only and then applied to validation and test, so nothing from those sets leaks into training. This gives 98 model inputs.

## Modeling: three implementations, one model

The target is binary (dropout or not), so I used logistic regression. I trained it three ways on the same split and the same 98 features.

1. scikit-learn: LogisticRegression. I tried C = 0.01, 0.1, 1 and 10, and picked C = 1 (lowest validation log loss, 0.328; C = 10 was not that much better).

2. Manual PyTorch: raw weight tensors, autograd and a manual update step (the same way we did in in Session 5), with the same L2 penalty as C = 1. I tried learning rates 0.01, 0.1, 0.5 and 1.0 for 2,000 epochs and picked 1.0 (validation log loss 0.328). With 0.01 the model hadn’t finished learning (0.377).

3. Standard PyTorch: nn.Linear with a sigmoid, nn.BCELoss and the SGD optimizer, with the same L2 penalty. I used plain SGD so that the only difference from the manual model is the tooling, not the algorithm. A learning rate of 1.0 again worked best.

I chose C and the learning rates on the validation set. The test set was used only once, for the final results below. The cost uses €300 per student contacted and €1,500 per missed dropout, with a 20% threshold.

| Model | ROC-AUC | Log loss | Recall | Precision | Contacted | Missed | Cost (€) |
|---|---|---|---|---|---|---|---|
| Naive baseline (everyone gets 39%) | 0.500 | 0.669 | 1.000 | 0.391 | 726 | 0 | 217,800 |
| scikit-learn | 0.929 | 0.326 | 0.933 | 0.659 | 402 | 19 | 149,100 |
| Manual PyTorch | 0.929 | 0.326 | 0.937 | 0.660 | 403 | 18 | 147,900 |
| Standard PyTorch | 0.929 | 0.326 | 0.937 | 0.660 | 403 | 18 | 147,900 |

All 3 models agree. They have the same ROC-AUC and log loss, and their predictions only differ for one student who sits right at the 20% threshold. All three cut the cost by about €70,000 (32%) compared with contacting everyone, while catching about 93% of dropouts.

The weights don’t match exactly. The predictions match, but individual weights differed by up to 0.275. The biggest gaps were in features that overlap with others: passed_nothing_1st_sem and no_units_1st_sem (which overlap with pass rate, approved courses and grade), and rare one-hot category columns. Because these features overlap, the model can share the “risk points” between them in many ways and get almost the same predictions, so gradient descent settles those weights slowly. To check this, I trained the manual model for 20,000 epochs instead of 2,000, and the largest gap shrank from 0.275 to 0.061. So the models aren’t different, just gradient descent was still settling.

I also did a threshold check. On the test set, the cheapest threshold is 0.17 (€147,000), close to the 0.20 that comes from the costs (300 ÷ 1,500). This suggests the predicted probabilities are roughly accurate. The dashboard lets you explore other thresholds and costs, but I didn’t tune the threshold on test data.

## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*

There are several limitations that I noticed while working with this data and with the models. The first one is that the data comes from 1 institution in portugal (the Instituto Politécnico de Portalegre). This is a publc university, with lower tuition fees, and that changes in comparison to private universities such as IE University, Kings College, etc... If we were to use it for another university, we would need to retrain the model on their data before we could use it.

Cost assumptions are estimates, I tried keeping them as real as possible by checking tuition fees for public universities in Spain and Portugal, but this is still a limitation. €1,500 and €300 are my best guesses, the formula assumes support always works, and it ignores how many students the support office can actually handle. Real figures from the university, a success rate for the support plan, or simply contacting the top highest-risk students would make the threshold more realistic. The dashboard already lets you test other costs.

I was also not able to do a time-based split. There’s no enrollment year column, so I split randomly and can’t test whether the model works on newer students. The data covers about a decade, and conditions changed over that time. With the year available, I’d train on older students and test on the newest ones. Having no time or date indicators also affected the Fee columns. I left them out because I couldn’t confirm when they were recorded. If the registry confirms they’re known by the end of the 1st semester, adding them back would improve the model. If we could get the enrollment year, I could train train on older students, test on the most recent ones, and recheck the model’s accuracy every year. And as for the fees if I could ask the university’s registry when the Debtor and Payments on time fields are updated, and see ff they’re known by the end of the 1st semester, I could then add them back and use them in the model.

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*

I used AI as a tutor/assistant in the project. It explained some concepts I didn't know/ didn't understand. It also helped me with coding, correcting my mistakes when my code wouldn't run, and explaining WHY I was making these mistakes. It particularly helped me with the creation of the gradio dashboard, which I had never done before. All decisions, opinions and business-problem aspects of the project I did myself. AI also helped me format my report (like the results table in Modeling section).