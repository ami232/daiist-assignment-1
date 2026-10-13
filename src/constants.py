from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

# delivery days are CET days, one stored file per day
MARKET_TZ = "Europe/Madrid"

PRICE_ES = "price_es"
PRICE_ES_DIFF_1D = "price_es_diff_1d"
