# Assignment 1 Report
- **María Martínez González**
- **18752**
- **mmg.ieu2022@student.ie.edu**
- **BBADBA 5B**

## Dataset
I use a public Spotify tabular dataset from Kaggle with songs from 1921 to 2020. Each row is one song. For each song we know audio features, which are numbers spotify calculates from the sound like: danceable, how energetic and how loud the song is. Also other characteristics like artist ,title, release date and a popularity score that goes from 0 to 100. As i mentioned, it is a tabular dataset so one row per song and one colum per feature.

Why did I remove and not use all the years? Old songs have a very low popularity score. The average for songs from 1921 to 1950 is 1.9, while for 2000 to 2020 it is about 55. That does not mean old songs sound worse. The score is based on recent plays, so old songs just get fewer. If I kept them, the model would mainly learn "is this song old?" instead of anything about the music. So I kept only 2000 to 2020.

How did I clean the dataset and make it smaller? The original file is 28 MB, which is too big for this project, so I did this once, in a separate script make_sample.py:
- I removed duplicates (the same artist with the same title).
- I kept at most 850 songs per year, chosen at random. This way no year counts more than another.
- The result is 17,850 songs (2.8 MB). This is the file the project uses. The original 28 MB file is not in the repository.
- One more cleaning step that i will explain later is that I found that songs from June 2020 have faulty popularity scores and removed them (131 songs). The model is trained on the remaining 17,719 songs. The details are in the next section.

## Business / real-life framing

**The decision.** Imagine a music label that receives many new songs but only has budget to promote some of them, for example with playlist promotion. The model gives each new song a score, so the label can rank the songs and promote the ones at the top. So the model is not there to say "this song will have exactly 63 popularity". It is there to help choose.

**What the model predicts.** Not the raw popularity, but how popular a song is compared with the other songs from the same year: target = popularity - average popularity of that year. If the number is positive, the song did better than the typical song of its year. If it is negative, it did worse. Why did I do it like this? ecause the average popularity goes up every year: 45.8 in 2000 and 69.4 in 2019. Part of this is just how the score is measured (recent songs get more recent plays), not that songs are getting better. A label that compares songs released at the same time does not care about that general rise. It only cares about which songs beat the others. Without this change, the model would mostly learn "is the song recent?" again.

**How this decided the pipeline:**
- **I split the data by time, not randomly.** In real life, the label never knows the future. A random split would let the model learn from songs released after the ones it is tested on, which is cheating. So I used:
  - **Train: 2000 to 2015** (13,600 songs). The model learns here.
  - **Validation: 2016 to 2017** (1,700 songs). I only use it to choose settings.
  - **Test: 2018 to May 2020** (2,419 songs). I look at it only at the end, to see
    how the model does on songs it has never seen.
- **I do not use the year as an input.** For a brand-new song, the year gives no information about how it sounds.
- **Everything the model learns from data, it learns only from the training years.** This includes the scaling of the numbers and the artist information. If not, information from the future would leak into the model.
- **I chose the metrics because of the decision.** The label needs a good ranking, so besides the usual errors (MAE, RMSE and R²) I use a business number, the **hit rate**: if we promote the top k% of songs, how many of them really didbetter than the typical song of their year? The dashboard has a slider to change k.

## Data preparation & feature engineering
**I found a data problem** in June 2020: I was checking how spread out the popularity is in each year. Every year had a spread between 5.8 and 8.8, except 2020, which had 18.3. That looked wrong, so I looked month by month. from Jan to May spread was between (6.4 and 7.6) which was normal but in June spread was 29.8. Also, 5.6% of the 2020 songs had a popularity of 5 or lower, while in 2018 and 2019 there were none. **Why do I think it happened?** I believe June songs were so new when the data was collected that they had not collected plays yet, so their score was still very low. So I removed the 131 June 2020 songs **before** calculating each year's average. This matters because the target is "popularity minus the average of the year". If I had kept June, it would pull the 2020 average down, and every other 2020 song would get an artificial bonus of about +5 in its target, just because of the error in june 2020 songs. **Solving the promblem was key** After removing the songs , the 2020 spread went from 18.3 to 7.0, like the other years. Also, the relation between the artist feature (explainedin the next part) and the target on the test set went from 0.034 to 0.178. So the broken data was hiding a real signal. The final dataset has **17,719 songs**.

### The features (30 columns in total)
- **Audio features.** Acousticness, danceability, energy, loudness, tempo and valence are the main audio features, there were already numerical so I used them as they were.
- **Song length.** Most songs last about 3 or 4 minutes, but a few last over an hour. Those few songs distort the numbers. The logarithm is a way of shrinking huge numbers a lot and small numbers only a little, so the very long songs get pulled back closer to the rest. "Skew" just measures how lopsided the data is. It went from 18 (very lopsided) to almost 0 (balanced).
- **Speechiness, instrumentalness and liveness.** Most songs are near zero and a few are high. I used the square root, which reduces the problem to make it work but does not fully fix it
- **Explicit and mode.** They are already yes/no (0 or 1), so I did nothing.
- **Musical Key.** Stored the key as a number from 0 to 11, but the number is only a label . If I gave the model the raw numbers, it would treat key 11 as “bigger” than key 2 and try to find a straight-line pattern (higher key number means more or less popular), which makes no sense. Instead of one column with a number, I made one column per key, each answering a yes/no question (0 or 1).
- **Artists and title.** I added the number of artists (maximum 5) and whether the title has “feat” or “remix”, because collaborations and remixes might behave differently. “5 or more is just ‘many’”. It is the same idea as the logarithm for song length: don’t let a few extreme cases dominate.
- **Release month.** as months are a loop December comes January again. numbers 1 to 12, the model would think December and January are far apart. So i used sine and cosine to write the position as a circle.
- **Songs with no month.** For about 7% of songs, we only know the year. Months were written as a position in the circle so songs with no month I put 0 and 0, which is the center of the circle. It points to no month at all, so it says “no information about the month” and doesn’t favor any season. I chose to remove the year as well because most of these songs were old so it would tell the model “this is an old song”.
- **Different Scales.** Putting everything on the same scale. Tempo is around 120, while danceability is between 0 and 1. tempo would seem more important just because its numbers are bigger. For each feature I subtracted its average and divided by its spread (standard deviation). The result is that every feature has an average of 0 and a typical size of 1. A tempo of 120 and a danceability of 0.6 both become numbers like 0.3 or -1.2, meaning “a bit above average” or “a bit below”.

### The artist score (the feature that matters most)
The best audio feature, danceability, has a correlation of only 0.07 with the target. That is almost zero, so knowing how danceable a song is tells me very little about whether it beat its year. Since every audio feature is that weak, a model using only audio cannot predict much. That is why:
- the audio-only model had a validation R² of about -0.01, slightly worse than
  just guessing the average,
- the artist score became the most important feature, since it was the one clue
  with a real relation (0.31 in training, 0.21 in validation and 0.18 in test),
- the final R² is low (0.049).
- The same artists appear many times (8,542 different artist names for 17,850 songs). So artist's songs are a useful clue: if an artist's songs usually beat their year, a new song by them probably will too.

**Methodology** For each song, I calculate the average target of that artist's other songs. If the artist has few songs, I pull the average towards the overall average, so one lucky song does not make an artist look great. If a song has several artists, I average their scores. I also added how many songs the artist has, so the model knows how much to trust the score.

**Two protections against cheating:**
- **A song never sees its own answer.** For training songs, I split them in 5 groups. A song's artist score is calculated using only the other 4 groups. If not, the score would contain the answer the model is trying to guess.
- **Only the past is used.** For validation and test songs, the artist score uses only the training years. If the artist is not in the training years, the song gets a neutral value (the overall average). About 57% of validation songs and 45% of test songs have an artist the model already knows.

**Did it help?** Yes. On the validation set, the model without the artist score was slightly worse than just guessing the average (R² of about -0.01). With it, it was slightly better (about +0.015). The artist score is also by far the biggest weight in the model.

## Modeling: three implementations, one model

**Which model and why?** I used linear regression with a small penalty (Ridge). It is a simple formula: each feature is multiplied by a weight and the results are added up. I chose it because the assignment asks for a simple model, and because it is easy to explain, since you can read the weights. The penalty shrinks weights that are mostly noise, which helps here because the signal is weak.

**Three ways to train the same model, on the same split:**
- **scikit-learn (Ridge):** it solves the problem directly with a formula.
- **Manual PyTorch loop:** I write every step myself: predict, measure the error, find which direction improves it (the gradient), move the weights a little, and repeat 3,000 times.
- **Standard PyTorch (nn.Module + optimizer):** the usual way, where PyTorch handles the steps.

**Settings I tuned (using only the validation set):**
- **Penalty strength:** I tried 9 values, from 0.1 to 100,000. The best was 10,000. That is a strong penalty, which tells me most features carry very little signal.
- **Learning rate** (the size of each step in PyTorch): I tried 4 values and all gave exactly the same result, so the choice does not matter here. The features are on the same scale, so the problem is easy.

**Results on the test set (2018 to May 2020).** The naive baseline, which predicts the training average for every song, has an average error (MAE) of 5.12 and an RMSE of 6.45. All three methods give an MAE of 4.90, an RMSE of 6.29 and an R² of 0.049. So they beat the baseline a little, and the three give the same numbers.

**Do the three agree?** Yes. The weights are the same up to 0.0000013, which is just rounding. To make this true I had to get two details right:
- scikit-learn adds up the errors over all songs, but PyTorch uses the average. So the penalty must be divided by the number of songs in my manual loop, and doubled in the optimizer's weight decay (because of the derivative of a square).
- The intercept (the model's starting level) is not penalized in any of the three.

**One thing that looks like a difference but is not:** the final loss printed is 53.4 for the manual loop and 51.9 for standard PyTorch. The manual loss includes the penalty and the standard one does not. The weights are identical, so the models are the same.

**What is the model good for?** If the label promotes the top 20% of test songs according to the model, 57% of them really beat their year, against 43% for a random pick. At 5% it is 59%, and at 50% it is 53%. So the model is a modest help for ranking, not a predictor of exact popularity. The random rate is 43% and not 50% because a few very popular songs pull the average up.

## Limitations & next steps

- **Very low R² (0.049).** Only about 5% of the differences between songs of the same year are explained. Audio features say little about success, which also depends on marketing, fame and luck. Next step: add information known before release, like artist followers, label or marketing budget.
- **The artist signal gets weaker over time.** The relation falls from 0.31 in training to 0.21 in validation and 0.18 in test, and about 55% of test songs have an artist the model has never seen. Next step: retrain regularly so new artists enter.
- **The popularity score is a snapshot** taken when the data was collected, not at release. The yearly comparison helps but does not fully fix it. Next step: use the popularity measured a fixed time after release, like 30 days.
- **The June 2020 cause is a guess.** The numbers prove June is on a different scale, but I did not verify why. Next step: check when the data was collected.
- **Some features are still lopsided** and a linear model cannot capture non-linear effects. I stayed with linear models because the assignment asks for it and they are easy to explain.
- **Settings were chosen on only two validation years.** Next step: use several time-based validation windows.
- **The hit rate ignores money.** A real decision needs the cost of promoting a song and the value of a hit.

## How to run

Run uv run python main.py train to train the three models and save them in models/. Run uv run python main.py app to open the dashboard, which only loads the saved models and trains nothing. The checks I used to find the June 2020 problem are in eda/, run them from the repo root with uv run python -m eda.check_2020.

## Generative AI use disclosure

I used Claude on this assignment to:
- discuss and check my decisions, for example the dataset, how to clean data and how to handle June 2020.
- help write the three training loops and the Gradio dashboard.
- help me rewrite this report in simpler words.
  
I ran all the code myself in a Codespace, checked the outputs, and the numbers in this report come from my own runs. I went through the process step by step so that I can explain every decision without the code in front of me.
