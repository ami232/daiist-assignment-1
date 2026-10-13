"""
Feature engineering shared by train.py and app.py.

Both files import this, so the app builds the features in exactly the same
way as training did.
"""

import numpy as np
import pandas as pd

# The six main skills every outfield player has in FIFA.
SKILLS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

# Columns we use straight from the dataset.
BASIC_FEATURES = [
    "age", "potential", "height_cm",
    "skill_moves", "weak_foot", "international_reputation",
] + SKILLS

# Columns we create ourselves in add_features().
ENGINEERED_FEATURES = [
    "age_from_25_squared", "best_skill", "is_left_footed",
    "is_midfielder", "is_attacker",
]

ALL_FEATURES = BASIC_FEATURES + ENGINEERED_FEATURES

# Rounded average age of the players, used to centre the squared age term.
AVERAGE_AGE = 25

DEFENDER_POSITIONS = ["CB", "LB", "RB", "LWB", "RWB"]
MIDFIELDER_POSITIONS = ["CDM", "CM", "CAM", "LM", "RM"]
ATTACKER_POSITIONS = ["ST", "CF", "LW", "RW"]


def position_group(player_positions):
    """Turn 'RW, CF, ST' into 'attacker' using the first position listed."""
    main_position = player_positions.split(",")[0].strip()
    if main_position in DEFENDER_POSITIONS:
        return "defender"
    if main_position in MIDFIELDER_POSITIONS:
        return "midfielder"
    return "attacker"


def clean_data(df):
    """Remove the rows the model is not meant for."""
    # Goalkeepers have no pace/shooting/passing... values (they are empty).
    df = df[df["player_positions"] != "GK"]
    # Players with value 0 are free agents: there is no price to learn from,
    # and log(0) does not exist.
    df = df[df["value_eur"] > 0]
    return df.reset_index(drop=True)


def add_features(df):
    """Add the engineered columns. Each one only uses the player's own row."""
    df = df.copy()

    # Value goes up with age and then down. A straight line cannot bend,
    # so we give the model a squared age term to draw that curve.
    # We square the distance to 25 (about the average age) instead of the
    # age itself: plain age squared is almost a copy of age (correlation
    # 0.99) and that made the manual gradient descent loop very slow.
    df["age_from_25_squared"] = (df["age"] - AVERAGE_AGE) ** 2

    # A striker should not be punished for bad defending. The best of the
    # six skills says how good the player is at what he does best.
    df["best_skill"] = df[SKILLS].max(axis=1)

    df["is_left_footed"] = (df["preferred_foot"] == "Left").astype(int)

    # Position as 0/1 columns. Defender is the reference group (both 0),
    # so we only need two columns for three groups.
    group = df["player_positions"].apply(position_group)
    df["position_group"] = group
    df["is_midfielder"] = (group == "midfielder").astype(int)
    df["is_attacker"] = (group == "attacker").astype(int)

    # Target: the log of the value. Values go from a few thousand euros to
    # 100 million, and the log makes that range something a line can fit.
    if "value_eur" in df.columns:
        df["log_value"] = np.log(df["value_eur"])

    return df
