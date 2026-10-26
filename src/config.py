"""Shared settings for the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_injuries.csv"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"

SEED = 2023
TEST_SIZE = 0.3

OUTCOME = "fatal"
CATEGORICAL = ["gender", "age_group", "nationality", "industry",
               "occupation", "body_part", "injury_type"]
NUMERIC = ["claim_year"]

# Final XGBoost hyperparameters reported in Lee, Lee & Cho (2023), Heliyon 9:e20138.
XGB_PARAMS = dict(
    learning_rate=0.3,      # eta
    gamma=1,
    max_depth=6,
    n_estimators=500,
    colsample_bytree=1,
    eval_metric=["error", "auc"],
)
CV_FOLDS = 10
