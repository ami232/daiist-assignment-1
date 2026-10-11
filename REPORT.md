# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Cloe Chapotot
- **Student ID**: 18549
- **Email**: cchapotot.ieu2022@student.ie.edu
- **Group**: BBADBA 5A 

## Dataset



The dataset I used is a bike sharing dataset (kind of like Bicimad, but in the US) that contains hourly count of rental bikes (in 2011-2012) in Capital bikeshare system. It has weather and seasonal data as well as other details. It's available in the UC Irvine Machine Learning Repository. Each row represent an hour (0-23). it has 17,379 rows with 17 columns. I picked this one because I think it creates an interesting business case and mostly because bike sharing is becoming super popular right now so I thought it could be fun. Also because the data has the temporall component and certain variables, it will be good to analyze for the models I'm building in the assignment.

## Business / real-life framing


The scenario I want to focus on is a biksharing operator deciding hour by hour how many bikes to stock at stations and how many to rebalance between them. Themodel would predict total rentals (cnt) for a given hour from weather information known in advance.

The target is to predict the toal hourly rentals (cnt) because the stocking decisions depends on total demand.

For split, I'm using a time plit (chronological) because for instance, the mean hourly demand grew from 144 in 2011 to 235 in 2012 which reflects the service being more popular. A random split would leak the future into the training data and overstate accuracy. A chronological split allows me to train with the past and predict the future.

The metric I decided to focus on is RMSE which is in the sam unit as the target (bikes per hour) and penalizes larger errors. This measure is more simple to translate into business logic. Also, I believe that large misses are most important here because if you fail to predict demand by a lot for a specific time, it means that you are losing a lot of potential customers. In addition, I believe that under-predicting is costlier, is the fact that users of the service will be very disatisfied if they see the bike stations empty. 

## Data preparation & feature engineering



First, there were a few columns I dropped. Casual and registered added up to "cnt" so I removed those two. "instant' was a row counter, so I removed it. "dteday" is a data string but I already had yr, mnth, and hr, so I removed the string column. "atemp" was very similar to "temp" so I removed it. "season" repeants the month comumn, and "weekday" isn't as relevant as having "workindgday" and "holiday".

The future hummidity had a few rows that had it at 0. instead of deleting it, I substituted with teh median.

The weathersit column I one-hot encded and combined the last category (which is for extreme rain) into the 3rd to make it more simple since it isn't a frequent catzgory. The category 1 was the baseline. I also had to one-hot encode month so that each month has its own effect.

I also one-hot encoded hours but in two different columns, one for working days and one of weekends.

In terms of scaling, I used standardscaler for temp, hum, and windspeed to use in the pytorch model with gradient descent.

## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*

I used a linear regression model (using Ridge) because the target is continuous. It was also important to use ridge to penalize coefficients of featured that are correlated like temp and month. since the linaer model cannot bend, the hourxworking-day features carre the shape of the day.

Moving on, I chose alpha on the validation set. As alpha was growing, the Val RMSE was growing, so the smallest alpha was the best.

To go deeper into each model, scikit-learn calculates the best weights directly with a formula in one step. Manual PyTorch finds the weights by trial and improvement, starting with the weights at 0, and in each of 30,000 rounds I predict and measure the error, use outograd, and then update every weight. The nn.Module version does the same, but torch.optim.SDG replaces my hand-written update step. I kept split,the features, the penalty strength, the zero starting weights, and full-batch training the same across all three so that the only difference would come from the method and not the setup. 

| Model | Train RMSE | Val RMSE | Test RMSE | Test bias |
|---|---|---|---|---|
| Naive baseline (train mean) | 160.78 | 260.65 | 208.06 | −51.48 |
| scikit-learn Ridge | 65.09 | 114.02 | 101.67 | +6.33 |
| Manual PyTorch | 65.09 | 114.02 | 101.67 | +6.32 |
| `nn.Module` + `optim` | 65.09 | 114.02 | 101.67 | +6.32 |

in my first run, the models didn't agree and the predictions differed by up to 46 bikes/hour. the sklearn train MSE was lower than the putorch which meant that pytorch hadn't trached its optimum. So I made it again with 30,000 epochs and learning rates up to 0.7. After that, the train MSE matched and the predictions differend by 0.011 bikes/hourthen the manuan and nn.module are identical becauser they run the same algorithmn.

On the test set, the baseline's RMSE is 208.1 and the models' is 101.7, a reduction of about 51%. However, since the test set averages 219 rentals per hour, a typical error of 102 means that the model is still far off from being optimal but it's better than guessing.

The overall test bias is +6.3 bikes/hour, which suggests the model is unbiased. However, broken down by hour, the oerros go in oposite directions in different hours, so they cancel in the average. For example, at hour 8 on working days, the model predicts 133 fewer bikes than actual, and at hour 17 103. At night on working days, hours 0 to 5 are over-predicted by roughly 48 to 61 bikes, while actual demand is only about 5 to 42. These errors cancel in the average. 

## Limitations & next steps

- Rush hours are under-predicted. On working days the model misses by −133 bikes/hour at hour 8 and −103 at hour 17, which is the costly direction for the operator. A linear model adds effects together, so it can't make the peak bigger in good weather. Next step: add hour × weather interactions or try gradient boosting.


- Night hours are over-predicted. At hours 0-5 on working days, the model predicts about 50-60 bikes too many (actual mean 5-42), and about 0.5% of predictions were negative before I clipped them. Next step: model the log of demand or use a Poisson regression, which can't go negative.


- Weather inputs are observed, not forecast. The model uses the actual temperature and weather of the hour it predicts, but a real stocking decision would only have a forecast, so real errors would be larger. Next step: train and test with forecast data.

## Generative AI use disclosure

I used Claude to build code, especially the feature engineering, applying the models, and for the GRadio dashboard. Also, I used it to explain concepts that I wasnt familiar with and explain results that I got from the models. 
