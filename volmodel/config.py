"""Every number the pipeline depends on, in one place.

Each constant here is a decision. REPORT.md explains why each one has the value it has.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"

# Time
TRADING_DAYS = 252          # trading days per year, used to annualize volatility
HORIZON = 21                # the label looks one month (21 trading days) ahead
LONG_WINDOW = 252           # one year, for the long-run volatility and drawdown features

# Chronological split. A row belongs to a block only if its whole 21-day label
# window ends inside that block, which purges 21 rows at each boundary.
TRAIN_END = "2014-12-31"
VAL_START, VAL_END = "2015-01-01", "2019-12-31"
TEST_START = "2020-01-01"

# Preprocessing
CLIP_QUANTILES = (0.01, 0.99)   # winsorize each feature at its training percentiles
VOL_FLOOR = 0.005               # 0.5% annualized, keeps log(vol) finite on flat days

# Hyperparameters (the only ones we tune, per the brief)
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]     # inverse L2 strength, chosen on validation log loss
LR_GRID = [0.01, 0.1, 0.5, 1.0]                   # PyTorch learning rate, chosen on convergence
EPOCHS = 3000                                      # full-batch gradient descent steps
SEED = 0

# Business set-up: a vol-target fund that cuts gross exposure when the model fires
TARGET_VOL = 0.10           # the fund targets 10% annualized volatility
MAX_LEVERAGE = 2.0          # cap on vol-target leverage
HAIRCUT = 0.25              # cut gross exposure by 25% for the next month when the model says "vol will rise"
DEFAULT_MISS_MULTIPLE = 3.0 # a missed vol rise costs 3x a false alarm (PM risk appetite)
