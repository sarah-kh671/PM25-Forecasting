from pathlib import Path

N_SPLITS = 5
TEST_START = "2025-01-01"  # final future test period: agree on this date with the team

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed" / "cleaned_pm25.csv"
RESULTS = ROOT / "results"
FOLDS = ROOT / "folds"
FIGURES = ROOT / "figures"