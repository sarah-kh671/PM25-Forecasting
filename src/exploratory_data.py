# ============================================================
# PM2.5 FORECASTING - GENERAL DATA EXPLORATION
# Person 2
#
# Run this script from the main project folder:
# python src/exploratory_data.py
#
# Required input:
# data/cleaned_pm25.csv
#
# Outputs are saved in:
# results/
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_FOLDER = os.getcwd()
DATA_FILE = os.path.join(PROJECT_FOLDER, "data", "cleaned_pm25.csv")
RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "results")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "The file data/cleaned_pm25.csv was not found.\n"
        "Please run this script from the main PM25_Forecasting project folder."
    )

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(RESULTS_FOLDER, "histograms"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_FOLDER, "boxplots"), exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("GENERAL DATA EXPLORATION")
print("=" * 60)
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

if "pm25" not in df.columns:
    raise ValueError("The target column 'pm25' is missing from the dataset.")

# Convert known date columns when they exist.
for column in ["date", "datetime_utc"]:
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

# Numeric columns used for statistical exploration.
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude identifier-like numeric columns from statistical plots/correlations.
excluded_columns = ["id"]
analysis_columns = [c for c in numeric_columns if c not in excluded_columns]

# ------------------------------------------------------------
# 3. BASIC DATASET INFORMATION
# ------------------------------------------------------------

overview = pd.DataFrame({
    "item": [
        "number_of_rows",
        "number_of_columns",
        "number_of_numeric_columns",
        "number_of_exact_duplicate_rows"
    ],
    "value": [
        len(df),
        len(df.columns),
        len(numeric_columns),
        int(df.duplicated().sum())
    ]
})
overview = overview.astype(object)

if "date" in df.columns:
    overview.loc[len(overview)] = [
        "first_date",
        str(df["date"].min().date()) if df["date"].notna().any() else ""
    ]
    overview.loc[len(overview)] = [
        "last_date",
        str(df["date"].max().date()) if df["date"].notna().any() else ""
    ]

overview.to_csv(
    os.path.join(RESULTS_FOLDER, "dataset_overview.csv"),
    index=False
)

# ------------------------------------------------------------
# 4. MISSING VALUES
# ------------------------------------------------------------

missing = pd.DataFrame({
    "variable": df.columns,
    "n_missing": [int(df[c].isna().sum()) for c in df.columns],
    "pct_missing": [
        round(float(df[c].isna().mean() * 100), 4) for c in df.columns
    ]
})

missing.to_csv(
    os.path.join(RESULTS_FOLDER, "missing_values.csv"),
    index=False
)

plt.figure(figsize=(12, 6))
plt.bar(missing["variable"], missing["pct_missing"])
plt.xticks(rotation=70, ha="right")
plt.ylabel("Missing values (%)")
plt.title("Percentage of Missing Values by Variable")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_FOLDER, "missing_values.png"),
    dpi=200
)
plt.close()

# ------------------------------------------------------------
# 5. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

descriptive = df[analysis_columns].describe().T
descriptive["skewness"] = df[analysis_columns].skew()
descriptive["kurtosis"] = df[analysis_columns].kurtosis()
descriptive = descriptive.rename(columns={
    "count": "n_valid",
    "mean": "mean",
    "std": "std",
    "min": "min",
    "25%": "q1",
    "50%": "median",
    "75%": "q3",
    "max": "max"
})

descriptive.to_csv(
    os.path.join(RESULTS_FOLDER, "descriptive_statistics.csv")
)

# ------------------------------------------------------------
# 6. IQR OUTLIER COUNTS
# IQR flags are potential outliers, not automatic errors.
# ------------------------------------------------------------

outlier_rows = []

for column in analysis_columns:
    values = df[column].dropna()

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = int(((values < lower) | (values > upper)).sum())

    outlier_rows.append({
        "variable": column,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_out_iqr": count,
        "pct_out_iqr": round(count / len(values) * 100, 4)
    })

outliers = pd.DataFrame(outlier_rows)
outliers.to_csv(
    os.path.join(RESULTS_FOLDER, "iqr_outlier_summary.csv"),
    index=False
)

# ------------------------------------------------------------
# 7. HISTOGRAMS AND BOXPLOTS
# ------------------------------------------------------------

for column in analysis_columns:
    values = df[column].dropna()

    if len(values) == 0:
        continue

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=40, edgecolor="black")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title("Distribution of " + column)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_FOLDER, "histograms", column + "_histogram.png"),
        dpi=200
    )
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.boxplot(values)
    plt.ylabel(column)
    plt.title("Boxplot of " + column)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_FOLDER, "boxplots", column + "_boxplot.png"),
        dpi=200
    )
    plt.close()

# ------------------------------------------------------------
# 8. CORRELATIONS
# Pairwise correlations use available numeric observations.
# Correlation does not prove causation.
# ------------------------------------------------------------

correlation_columns = [
    c for c in analysis_columns if c != "pm25"
]

pearson = df[analysis_columns].corr(method="pearson")
spearman = df[analysis_columns].corr(method="spearman")

pearson.to_csv(
    os.path.join(RESULTS_FOLDER, "pearson_correlations.csv")
)
spearman.to_csv(
    os.path.join(RESULTS_FOLDER, "spearman_correlations.csv")
)

plt.figure(figsize=(10, 8))
plt.imshow(pearson, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Pearson correlation")
plt.xticks(range(len(pearson.columns)), pearson.columns, rotation=90)
plt.yticks(range(len(pearson.index)), pearson.index)
plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_FOLDER, "pearson_correlation_heatmap.png"),
    dpi=200
)
plt.close()

plt.figure(figsize=(10, 8))
plt.imshow(spearman, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Spearman correlation")
plt.xticks(range(len(spearman.columns)), spearman.columns, rotation=90)
plt.yticks(range(len(spearman.index)), spearman.index)
plt.title("Spearman Correlation Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_FOLDER, "spearman_correlation_heatmap.png"),
    dpi=200
)
plt.close()

if "pm25" in pearson.columns:
    target_corr = pd.DataFrame({
        "predictor": [c for c in analysis_columns if c != "pm25"],
        "pearson_corr_with_pm25": [
            pearson.loc[c, "pm25"] for c in analysis_columns if c != "pm25"
        ],
        "spearman_corr_with_pm25": [
            spearman.loc[c, "pm25"] for c in analysis_columns if c != "pm25"
        ]
    })
    target_corr["abs_pearson"] = target_corr["pearson_corr_with_pm25"].abs()
    target_corr = target_corr.sort_values("abs_pearson", ascending=False)
    target_corr.to_csv(
        os.path.join(RESULTS_FOLDER, "target_correlations.csv"),
        index=False
    )

# Predictor pairs with strong absolute Pearson correlation.
strong_pairs = []
for i in range(len(correlation_columns)):
    for j in range(i + 1, len(correlation_columns)):
        a = correlation_columns[i]
        b = correlation_columns[j]
        value = pearson.loc[a, b]
        if pd.notna(value) and abs(value) > 0.9:
            strong_pairs.append({
                "variable_1": a,
                "variable_2": b,
                "pearson_correlation": value
            })

pd.DataFrame(
    strong_pairs,
    columns=["variable_1", "variable_2", "pearson_correlation"]
).to_csv(
    os.path.join(RESULTS_FOLDER, "strong_predictor_correlations.csv"),
    index=False
)

# ------------------------------------------------------------
# 9. PM2.5 THRESHOLDS
# ------------------------------------------------------------

pm25 = df["pm25"].dropna()
thresholds = pd.DataFrame({
    "threshold": [25, 50],
    "comparison": [">", ">"],
    "n_observations": [
        int((pm25 > 25).sum()),
        int((pm25 > 50).sum())
    ],
    "percentage": [
        round(float((pm25 > 25).mean() * 100), 4),
        round(float((pm25 > 50).mean() * 100), 4)
    ]
})

thresholds.to_csv(
    os.path.join(RESULTS_FOLDER, "pm25_thresholds.csv"),
    index=False
)

# ------------------------------------------------------------
# 10. PROFESSOR'S EXPLORATORY_DATA.CSV STRUCTURE
# Only values calculated in this script are filled.
# Not-calculated analyses remain blank; they are not invented.
# ------------------------------------------------------------

professor_columns = [
    "model_id", "model_group", "variable", "family", "type", "unit",
    "n_total", "n_missing", "pct_missing", "n_unique",
    "n_duplicates_key", "min", "q1", "median", "q3", "max",
    "mean", "std", "skewness", "kurtosis", "iqr",
    "n_out_iqr", "pct_out_iqr", "n_out_z", "pct_out_z",
    "n_out_if", "pct_out_if", "n_out_lof", "pct_out_lof",
    "n_sentinel_low", "n_sentinel_high", "n_removed_total",
    "max_abs_corr", "corr_partner", "vif", "modalities",
    "target_assoc", "target_pval", "acf_lag1", "acf_lag24",
    "pacf_lag1", "n_lags_recommended", "adf_p",
    "is_leakage_suspect", "notes"
]

schema_rows = []

for column in analysis_columns:
    values = df[column].dropna()
    row = {name: np.nan for name in professor_columns}

    row["model_id"] = "general_eda"
    row["model_group"] = "G8"
    row["variable"] = column
    row["family"] = "target" if column == "pm25" else "predictor"
    row["type"] = "numeric"
    row["n_total"] = len(df)
    row["n_missing"] = int(df[column].isna().sum())
    row["pct_missing"] = round(float(df[column].isna().mean() * 100), 4)
    row["n_unique"] = int(df[column].nunique())
    row["min"] = values.min() if len(values) else np.nan
    row["q1"] = values.quantile(0.25) if len(values) else np.nan
    row["median"] = values.median() if len(values) else np.nan
    row["q3"] = values.quantile(0.75) if len(values) else np.nan
    row["max"] = values.max() if len(values) else np.nan
    row["mean"] = values.mean() if len(values) else np.nan
    row["std"] = values.std() if len(values) else np.nan
    row["skewness"] = values.skew() if len(values) else np.nan
    row["kurtosis"] = values.kurtosis() if len(values) else np.nan

    q1 = values.quantile(0.25) if len(values) else np.nan
    q3 = values.quantile(0.75) if len(values) else np.nan
    iqr = q3 - q1 if len(values) else np.nan
    row["iqr"] = iqr

    if len(values) and pd.notna(iqr):
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_iqr = int(((values < lower) | (values > upper)).sum())
        row["n_out_iqr"] = n_iqr
        row["pct_out_iqr"] = round(n_iqr / len(values) * 100, 4)

    if column in pearson.columns:
        other = pearson[column].drop(labels=[column]).dropna()
        if len(other):
            partner = other.abs().idxmax()
            row["max_abs_corr"] = abs(float(other.loc[partner]))
            row["corr_partner"] = partner

    row["notes"] = (
        "IQR values are potential outliers, not automatic errors. "
        "Isolation Forest, LOF, VIF, significance tests, time-series "
        "statistics and leakage review were not calculated by this script."
    )

    schema_rows.append(row)

exploratory_data = pd.DataFrame(schema_rows, columns=professor_columns)
exploratory_data.to_csv(
    os.path.join(RESULTS_FOLDER, "Exploratory_Data.csv"),
    index=False
)

# ------------------------------------------------------------
# 11. NOTES ABOUT ANALYSES NOT PERFORMED
# ------------------------------------------------------------

with open(
    os.path.join(RESULTS_FOLDER, "analysis_limitations.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(
        "GENERAL DATA EXPLORATION - ANALYSIS LIMITATIONS\n\n"
        "This script calculates descriptive statistics, missing values, "
        "exact duplicate rows, IQR potential outliers, histograms, boxplots, "
        "Pearson and Spearman correlations, and PM2.5 threshold counts.\n\n"
        "The following analyses are not calculated here: Isolation Forest, "
        "Local Outlier Factor (LOF), VIF, statistical p-values, ACF, PACF, "
        "ADF stationarity tests, sentinel-value validation, and full data "
        "lineage/leakage verification. These should remain unreported until "
        "they are completed and verified.\n\n"
        "IQR flags identify unusual values; they do not prove that values "
        "are incorrect and do not justify automatic deletion.\n"
    )

print("\nExploration completed successfully.")
print("Results saved in:", RESULTS_FOLDER)
print("Main output: results/Exploratory_Data.csv")
print("=" * 60)
