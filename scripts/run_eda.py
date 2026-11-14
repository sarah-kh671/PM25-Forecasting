import sys

sys.path.insert(0, ".")

from src.temporal_exploration import run_temporal_eda

if __name__ == "__main__":
    run_temporal_eda(input_path="data/processed/cleaned_pm25.csv",
                     output_path="results/Exploratory_temporal.csv")