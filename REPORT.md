# Assignment 1 Report

- **Name**: Juan Diaz-Bernardo
- **Student ID**: 19785
- **Email**: jdiazbernard.ieu2022@student.ie.edu
- **Group**: BBADBA A

## Dataset

I worked with the **Bike Sharing Dataset** (UCI Machine Learning Repository), specifically the hourly file `hour.csv`: 17,379 rows covering two full years, 2011 and 2012, of a bike-sharing system in Washington D.C. Each row represents one specific hour of one specific day, along with the weather at that time and how many bikes were rented. The target is `cnt`, the total number of rentals in that hour.

The available columns range from the obvious (hour, month, season) to normalized weather variables (temperature, feels-like temperature, humidity, wind speed), plus calendar indicators (holiday, working day, weekday). I picked this dataset because, unlike flatter ones, the relationship between time and demand here isn't linear or obvious: there are two daily usage peaks, not one, and the same weather affects demand differently depending on whether people actually need to commute. That leaves real room for feature engineering, rather than just plugging in the raw columns.

## Business / real-life framing

For a bike-sharing company, the real problem isn't "how many bikes exist in total" — it's something far more operational: every morning, someone has to decide how many bikes to redistribute between stations before the day starts. Get it wrong one way and they lose customers because bikes aren't available when needed; get it wrong the other way and they're paying for trucks and staff to move bikes nobody was going to use.

This model targets exactly that: predicting how many rentals (`cnt`) will happen in a given hour, using only the information operations would actually have ahead of time — hour of day, month, holiday/working-day status, and the weather forecast. I deliberately left out `casual` and `registered`: both columns are literally the components that sum to `cnt`. Using them would have let the model "cheat" instead of learning any real demand pattern.

Framing the problem this way isn't just narrative — it drives concrete pipeline decisions:

**The split couldn't be random.** This is time-series data (2011-2012), and in a real deployment you'd never have access to future data to train a model you then use in the present. That's why training uses the oldest months and testing uses the most recent ones — it simulates the real situation of predicting something that hasn't happened yet.

**The metric that actually matters is error in real units, not an abstract score.** An R² of 0.65 doesn't mean much to an operations team; a MAE of 95 bikes does, because it translates directly into how much buffer to leave at each station. That's why MAE is reported as the metric that should drive an actual decision, with R² as a secondary check.

**The cost of being wrong isn't symmetric.** Underestimating demand during Monday rush hour costs more — in frustrated customers and reputation — than having a few extra bikes sitting at a quiet station on a Sunday afternoon. This is reflected later in how I read the model's larger errors, not just its average.

## Data preparation & feature engineering

The dataset came in fairly clean: I checked `df.isnull().sum()` before touching anything, and there wasn't a single null value in any column, so no imputation decisions were needed. I did convert `dteday` from text to an actual datetime type, which was necessary to sort chronologically and build the time-based split correctly.

From there I built five new features, each targeting a pattern the raw columns didn't capture well on their own:

- **`is_weekend`**: weekends follow a much more even usage pattern throughout the day than working days; separating this out helps the model distinguish between the two demand regimes.
- **`is_rush_hour`**: combines `workingday` and `hr` to directly flag commuting windows (7-9h and 17-19h on working days). This turned out to be the strongest single signal in the whole dataset.
- **`hr_sin` / `hr_cos`**: hour of day is a cyclical variable — 23h and 0h are right next to each other in reality, but numerically look like opposites. This trigonometric encoding fixes that, and it matters especially for a linear model, which would otherwise treat hour as a straight-line magnitude from 0 to 23.
- **`bad_weather_workday`**: crosses `weathersit` and `workingday`, since the same bad weather hits differently depending on whether people still have to commute or can just stay home.

As mentioned above, `casual` and `registered` were excluded from the final feature set for the same reason: they're literally the components that sum to the target.

## Modeling: three implementations, one model

I trained a **linear regression** (continuous target: `cnt`) three ways, on exactly the same time-based split and the same 16 scaled features (`StandardScaler` fit only on train, then applied to validation and test).

| Model              | MAE (test) | RMSE (test) | R² (test) |
|---------------------|-----------:|------------:|----------:|
| scikit-learn         | 95.48      | 126.70      | 0.651     |
| Manual PyTorch       | 95.52      | 126.73      | 0.651     |
| Standard PyTorch     | 95.52      | 126.73      | 0.651     |
| Baseline (mean)      | 168.46     | 227.18      | -0.123    |

All three methods converge to nearly indistinguishable results, which has a simple explanation: all three are optimizing the same loss function (mean squared error) on the same linear regression and the same split. Scikit-learn solves it analytically in closed form; both PyTorch versions reach the same optimum through iterative gradient descent, and with 500 epochs and a learning rate of 0.1 they're already converged by roughly epoch 100 — the loss drops from 48,903 to 6,557 in the first 100 epochs and then stays essentially flat for the rest of training.

One result worth flagging: the baseline scores a **negative R²** on test (-0.123). That means predicting "always the training mean" is worse than just predicting the test set's own mean — average demand during the training period (January 2011 to May 2012) doesn't represent the test period well (September to December 2012), which drops off due to the seasonal shift. Rather than being a problem, this actually confirms the time-based split was the right call: a random split would have mixed both periods together and hidden this seasonal effect, giving a false impression that "predicting the mean" was a reasonable strategy.

**Tuning.** Following the instruction to tune only regularization and learning rate, I tried Ridge with alpha values of 0, 1, 10, and 50 on validation: MAE barely moved (110.27 → 110.35), suggesting the model wasn't meaningfully overfitting with 16 features over more than 12,000 training rows. For the manual loop's learning rate I tried 0.01, 0.1, and 0.5: 0.01 converges somewhat slower (val MAE 110.33); 0.1 lands at a near-optimal point (val MAE 110.28, the value used in the final model); and 0.5 causes the training to **diverge entirely** — the weights blow up to NaN because the update step is too large and overshoots the minimum instead of approaching it. It's a textbook example of why a poorly chosen learning rate breaks training, and it confirms that 0.1 was a reasonable choice.

## Limitations & next steps

The clearest limitation shows up right in the dashboard: the model **systematically underestimates demand during the highest-traffic hours**. Once `cnt` goes above roughly 600-800 rentals per hour, predictions from all three methods fall noticeably below the "perfect prediction" line in the prediction-vs-actual plot. That makes sense once you look at the target's histogram: the vast majority of hours have low demand, so the model has relatively few examples to learn from at the extremes — and a linear model has no real way to capture that non-linear behavior even if the data were better balanced.

Given more time, I'd try two concrete things:

1. **A non-linear model** (random forest or gradient boosting), which could capture the interaction between rush hour, weather, and season directly, without me having to hand-build every interaction as an explicit feature.
2. **Log-transforming the target** (`log(cnt + 1)`) before training, undoing the transform at prediction time. This is a standard move when the target is this skewed, and it should help specifically in the high tail of the distribution, which is exactly where the current model struggles most.

A separate, more structural limitation: the dataset only covers 2011-2012, in a single city. There's no way to know whether these patterns (rush hour, the weather effect) hold up in a different city with different climate or commuting culture, so any real deployment would need, at minimum, validation against the company's own data before trusting these predictions.

## Generative AI use disclosure

I used Claude (Anthropic) as support for planning the pipeline, understanding concepts, and debugging errors throughout the project.


