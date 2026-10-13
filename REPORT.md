# Assignment 1 Report

- **Name**: Deniz Meten
- **Student ID**: 18758
- **Email**: dmeten.ieu2022@student.ie.edu
- **Group**: [BBADBA 5A]

## Dataset

I used a Dota 2 public-match dataset collected through the OpenDota API.

One row represents one completed match and contains:

* `match_id`
* `radiant_win`
* `radiant_team`
* `dire_team`

The project also uses:

* `dota_hero_id_map.csv` to convert hero IDs to readable hero names.
* `complete_opendota_attributes.csv` to add each hero's primary attribute and OpenDota role tags.

The training pipeline keeps the 20,000 most recent valid matches from the committed match file. A match is treated as valid when each team contains exactly five heroes, all ten picks are unique, and every hero ID exists in the metadata.

I chose Dota 2 dataset because I play it from time to time. A lot of people on the game make a strong emphasis on the draft, however from my experience draft does not necessarily mean you outright lost, or won a game. Draft however does somewhat dictate how easy or hard a game is going to be. By creating a model to predict an outcome of a match I want to see for myself how does it actually affect your win probability in a game.

## Business / real-life framing

The problem is:

**Can a model estimate Radiant's probability of winning a Dota 2 match using only information available when the draft is finished?**

A realistic use would be a draft-analysis tool for players, coaches, or analysts. The model deliberately does not use information created after the game begins, such as kills, gold, items, objectives, or match duration.

The target is:

* `1` = Radiant wins
* `0` = Dire wins

The selected matches are ordered by `match_id`, with older matches used for training and newer matches used for validation and testing. This is intended to mimic a real use case where historical matches are used to estimate the outcome of later matches.

For threshold analysis in the dashboard, the cost is:

`business cost = false positives + false negatives`

so every incorrect draft call costs one unit.

A time ordered-split was used because it is the closest reflection to the most recent balance patch in the game. By mixing older games it will not provide accurate data, since throughout patches heroes and items get either their numbers decreased or increased. There are also at times changes such as terrain changes, so by filtering the most recent matches we can avoid balance and map changes as much as possible.

The treatment of false positives and false negatives is the same because the outcome for each match is binary its either a loss or a win, so if the model predicts the win or loss wrongly, then it has the same exact weight.

## Data preparation & feature engineering

The raw team columns are comma-separated hero IDs, so the pipeline first parses each team and removes malformed matches.

The features used by the final model are:

1. **Signed hero-presence features**
  
  * `+1` if a hero is on Radiant
  * `-1` if a hero is on Dire
  * `0` if the hero is not in the match
2. **Role-composition differences**
  
  * For each OpenDota role, the feature is:
  * `Radiant role count - Dire role count`
3. **Primary-attribute composition differences**
  
  * Strength, Agility, Intelligence, and Universal counts are compared between the two teams.

The feature matrix is standardized using `StandardScaler`. The scaler is fitted only on the training data and then applied to validation, test, and dashboard inputs.

Hero picks matter because each and every hero has their own strengths and weaknesses. As an example, Anti-Mage is a strong character only in the later stages of the game, hence picking up strong early to mid game characters that provide a lot of CC (crowd control), such as Lion will be beneficial since it will allow the Anti-Mage character to get to their power spike. Hence why despite individual strengths, a good team composition is important.

A big feature that is failed to be captured by the dataset is the usefulness of a hero with the amount of gold they have. This is important because if a whole team has heroes that all require a lot of gold to be useful, an example of such a team composition would be Terrorblade, Anti-Mage, Juggernaut, Phantom Assassin and Lifestealer, then it will be difficult for that team to win since they would be starving each other of resources on the map. They also do not cover each other for their kit weaknesses since they have similar roles.

## Modeling: three implementations, one model

Because the target is binary, the project uses **logistic regression**.

The same feature matrix and the same train/validation/test split are used for all three required implementations:

1. scikit-learn `LogisticRegression`
2. manual PyTorch logistic regression using raw tensors, sigmoid + hand-written binary cross-entropy, autograd, and a manual gradient update
3. standard PyTorch using `nn.Sequential(nn.Linear(...), nn.Sigmoid())`, `torch.optim.Adam`, and `nn.BCELoss`

A naive baseline uses the Radiant win rate from the training set as a constant probability.

| Model | Accuracy | Log loss | ROC-AUC |
| --- | --- | --- | --- |
| Naive baseline | 0.5693 | 0.6857 | 0.5000 |
| scikit-learn | 0.5870 | 0.6749 | 0.5835 |
| Manual PyTorch | 0.5870 | 0.6749 | 0.5835 |
| PyTorch nn.Module | 0.5870 | 0.6749 | 0.5835 |

The three logistic regression implementations produced almost identical results, which is what I expected because all three models are solving the same classification problem using the same features and the same train, validation, and test split. Scikit-learn and the manual PyTorch version both achieved an accuracy of 58.70%, while the standard PyTorch version was effectively the same. Their ROC-AUC scores were also almost identical at around 0.5835. The small numerical differences are mainly caused by differences in optimization methods, initialization, and how each library updates the model parameters. Overall, this comparison shows that the three implementations are learning essentially the same relationship from the data, which also gives me confidence that the manual PyTorch implementation was built correctly.

## Limitations & next steps

The main limitation of this model is that it only looks at the draft, while the actual outcome of a Dota 2 match depends on much more than the ten heroes selected. One major issue is player skill. There is no reliable way in this dataset to quantify how good each individual player is, so two identical drafts can have very different outcomes depending on who is playing them.

Another limitation is that the model does not understand how gold and experience scale differently across heroes throughout a match. For example, extra gold on Anti-Mage can have a much larger impact in the long run than the same amount of gold on Lion. The current model only sees that the heroes were picked and does not know how efficiently they convert resources into power at different stages of the game.

The dataset also does not contain any information about hero mastery. A player who has hundreds of games on a hero will usually perform very differently from someone playing the hero for the first few times, but the model treats both cases exactly the same.

Role familiarity is another missing factor. A player may be forced into a role that they are not comfortable playing. For example, a position 1 carry player could be forced to play position 4. Even if the hero itself is familiar to them, their lack of experience in the role can still reduce their performance. This is especially important for heroes such as Naga Siren, which can be played in more than one role, because the model only sees that Naga Siren was picked and not the exact role or how experienced the player is in that role.

Finally, the model does not fully capture patch balance. This is not only about direct hero buffs and nerfs, but also about item changes. A hero can have strong base numbers but still become weaker if the items that normally support its playstyle are nerfed. For example, Earthshaker may look strong based on his own statistics, but if important items such as Blink Dagger or Shiva's Guard become significantly weaker, his actual effectiveness can drop even if the hero itself was not directly nerfed. Because Dota 2 changes frequently, a model trained on one patch can become less accurate as the meta changes.

With more time and better data, I would improve the model by including player MMR or ranking, hero mastery, actual player roles, patch information, and more detailed information about how different heroes scale with gold and experience. I would also retrain the model regularly so that it stays relevant as hero and item balance changes.

## Generative AI use disclosure

I used ChatGPT to help with coding boilerplate, debugging, code organization, and formatting. I reviewed the implementation and made sure I could explain the business framing, feature choices, model setup, and limitations myself.
