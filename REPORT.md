# Assignment 1 Report

- **Name**: Juan C Chavarria
- **Student ID**: 19313
- **Email**: jchavarria.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

- **Source:** UCI Machine Learning Repository, Bike Sharing Dataset (Fanaee-T & Gama, 2013), licence CC BY 4.0. File used: `hour.csv`.
- **One row:** one hour of the Capital Bikeshare system in Washington DC, from 2011-01-01 to 2012-12-31.
- **Size:** 17,379 rows, 17 columns, no missing values. Target: `cnt`, the total number of rentals in that hour (mean 189, median 142, max 977).

I picked the bike dataset because I found it personally interesting, as I use the BiciMAD bikes every day, and I believe it fits the project.

## Business / real-life framing

This model would be used by the operations team of the company that runs the bike stations in the city. They would use it to decide how many bikes to have available in the next hours. Each row in the data is the total number of rentals in the whole system in one hour, so the model tells us how much demand to expect, not where to put the bikes.

Predicting too low is worse than predicting too high. If we predict too low, there may be no bikes at the stations and we lose customers. If we predict too high, we only have some idle bikes, which costs much less.

Because the model is meant to predict future hours, the data is split by time and not randomly. Training is January 2011 to June 2012, validation is July to September 2012 (used to choose the learning rate), and test is October to December 2012, used only once at the end. A random split would be cheating because the hour next to a test hour could be in the training set, and neighboring hours are almost the same, so the test would be too easy.

The main metric is MAE, the average number of bikes the model misses by in one hour, which is easy to explain. I also report RMSE, which punishes big misses more, like in rush hour. MAE and RMSE treat predicting too low and too high the same, so I also report the average signed error (actual minus predicted, positive means too low) and the share of hours where the model predicts too low, to check the error that matters most.

## Data preparation & feature engineering

Split (chronological): train 2011-01-01 to 2012-06-30 (13,003 rows), validation 2012-07-01 to 2012-09-30 (2,208 rows), test 2012-10-01 to 2012-12-31 (2,168 rows). `temp` and `hum` were scaled using the mean and standard deviation of the training rows only.

| Column | Decision | Evidence |
|---|---|---|
| casual, registered | dropped | they add up exactly to `cnt` |
| instant | dropped | it is only a row counter |
| atemp | dropped | its correlation with temp is 0.99 |
| hr + workingday | kept, as yes/no columns for 48 hour and day-type groups (one left out as the reference) | at 8am the average is 477 rentals on working days and 106 on non-working days |
| yr | kept | average rentals were 144 in 2011 and 235 in 2012 |
| temp, hum | kept, scaled | correlation with `cnt` is 0.40 and -0.32 |
| season, weathersit | kept as yes/no columns | weathersit 4 has only 3 rows, so it was merged into 3 |
| windspeed | not used | its correlation with `cnt` is only 0.09 |
| mnth, holiday, weekday | not used | see below |

I combined hour and workday because working days and weekends do not have the same daily shape: at 8am the average is 477 rentals on working days and 106 on non-working days, while at 12pm it is 201 and 366. I turned the hour into yes/no columns because demand spikes at certain hours instead of rising or falling in a straight line, so a single 0-23 number could not capture it.

I chose year because demand changed greatly from 2011 to 2012, season because it captures yearly patterns that temperature alone misses, and humidity because it may affect riding beyond what temperature explains.

I did not use holiday because every holiday already has a value of 0 in the workingday column, which the model already uses. I did not use weekday because workingday already separates working days from weekends and holidays. I did not use mnth because it overlaps with season and temperature, so I left it out for simplicity.

Note: the dataset's own readme lists the season codes wrongly. In the data, season 1 covers January to March, so 1 = winter, 2 = spring, 3 = summer, 4 = fall.

## Modeling: three implementations, one model

I used linear regression because the target, `cnt`, is a number (how many bikes are rented in an hour), not a yes or no answer.

| Method | Val MAE | Test MAE | Test RMSE | Test signed error | Test % too low |
|---|---|---|---|---|---|
| Baseline (training average) | 197.8 | 156.6 | 208.1 | 51.5 | 50.9 |
| scikit-learn | 88.7 | 72.3 | 98.3 | -4.0 | 39.9 |
| Manual PyTorch (lr 0.5, 5000 epochs) | 88.9 | 72.5 | 98.4 | -4.0 | 40.0 |
| nn.Linear + Adam (lr 1.0, 2000 epochs) | 89.0 | 72.6 | 98.5 | -4.0 | 39.9 |

Learning-rate search (validation MAE): manual 0.01 gave 99.4, 0.1 gave 89.4, 0.5 gave 88.9 and 1.0 gave nan; Adam 0.1 gave 102.0, 0.5 gave 89.6 and 1.0 gave 89.0.

The three methods agree, within 0.3 bikes of MAE. The small gap is because scikit-learn calculates the exact best answer directly, while the PyTorch models walk down toward the lowest point of the loss in steps and stop close to it.

At lr=1.0 the steps were so large that the model overshot the lowest point and got worse with every step, until the values blew up into nan. At lr=0.01 the steps were so small that the model was still on its way down after 5,000 steps, so it ended with a higher MAE (99.4).

I did not use regularization; I only tuned the learning rate.

## Limitations & next steps

- **Limitation:** The model predicts too few bikes at rush hours (roughly +135 at 8am and +105 at 5pm on working days in the test period) and too many at night, and the overall average hides this. **Fix:** Add a safety buffer only at rush hours, not one flat buffer for every hour.
- **Limitation:** The test set is only one season (October to December 2012), so the results may not hold in other seasons. **Fix:** Test on several different periods, not just one season.
- **Limitation:** The average error leaned too low on validation (+18) but slightly too high on test (-4), so I cannot say the model leans in one direction. **Fix:** Check the signed error on several other periods before trusting a direction.

## Generative AI use disclosure

I used Claude (Anthropic) throughout this assignment. It wrote the code for loading the data, the time split, the features, the scaling, the metrics, the manual PyTorch loop, the nn.Module version, saving the models and the Gradio dashboard, and it explained how each part works. It also suggested the split dates, the signed-error metric, the yes/no encoding of hour and workday, merging weather category 4 into 3, and dropping atemp, and it pointed out that casual and registered leak the target. I chose the dataset from its suggestions, decided the final feature list (including keeping year, season and humidity), gave the business answers (who uses the model and why predicting too low is worse), and ran and checked everything myself. Claude turned my business answers into the framing paragraph, which I reviewed. For the data preparation, modeling and limitations sections, I gave my own explanations and choices in the chat, and Claude corrected the wording and formatted them; for the limitations, I picked the fixes from options Claude listed.