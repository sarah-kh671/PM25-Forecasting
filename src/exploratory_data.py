
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/cleaned_pm25.csv")

os.makedirs("results/histograms", exist_ok=True)
os.makedirs("results/boxplots", exist_ok=True)
os.makedirs("results/correlations", exist_ok=True)

print("=" * 60)
print("PM2.5 GENERAL DATA EXPLORATION")
print("=" * 60)

# ==========================================
# 2. DATASET OVERVIEW
# ==========================================

print("\n1. First five rows:")
print(df.head())

print("\n2. Dataset shape:")
print(df.shape)

print("\n3. Dataset information:")
df.info()

print("\n4. Column names:")
print(df.columns.tolist())

# ==========================================
# 3. MISSING VALUES
# ==========================================

print("\n5. Missing values:")

missing = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": df.isnull().mean() * 100
})

print(missing)

print("\nTotal missing cells:", df.isnull().sum().sum())

# Save missing values visualization
missing_to_plot = missing[missing["Missing_Count"] > 0]

if not missing_to_plot.empty:
    plt.figure(figsize=(9, 5))
    missing_to_plot["Missing_Count"].plot(kind="bar")
    plt.title("Missing Values by Variable")
    plt.xlabel("Variables")
    plt.ylabel("Number of Missing Values")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/missing_values.png", dpi=300)
    plt.close()

# ==========================================
# 4. DESCRIPTIVE STATISTICS
# ==========================================

numeric_df = df.select_dtypes(include=np.number)

statistics = numeric_df.describe().T

statistics["Skewness"] = numeric_df.skew()
statistics["Kurtosis"] = numeric_df.kurtosis()
statistics["Missing_Count"] = numeric_df.isnull().sum()
statistics["Missing_Percentage"] = numeric_df.isnull().mean() * 100

print("\n6. Descriptive statistics:")
print(statistics)

# Export numerical exploration results
statistics.to_csv("results/Exploratory_Data.csv")

print("\nExploratory_Data.csv saved successfully!")

# ==========================================
# 5. SKEWNESS AND KURTOSIS
# ==========================================

print("\n7. Skewness:")
print(numeric_df.skew())

print("\n8. Kurtosis:")
print(numeric_df.kurtosis())

# ==========================================
# 6. SENSOR COUNTS
# ==========================================

print("\n9. Observations by sensor:")
sensor_counts = df["sensor_name"].value_counts()
print(sensor_counts)

sensor_counts.to_csv("results/sensor_counts.csv")

# ==========================================
# 7. SELECT VARIABLES FOR VISUALIZATION
# ==========================================

variables = [
    "pm25",
    "AOD",
    "NO2",
    "CO",
    "O3",
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v"
]

# ==========================================
# 8. HISTOGRAMS
# ==========================================

print("\n10. Creating histograms...")

for column in variables:

    plt.figure(figsize=(8, 5))

    plt.hist(df[column].dropna(), bins=30, edgecolor="black")

    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"results/histograms/{column}_histogram.png",
        dpi=300
    )

    plt.close()

print("Histograms saved successfully!")

# ==========================================
# 9. BOXPLOTS
# ==========================================

print("\n11. Creating boxplots...")

for column in variables:

    plt.figure(figsize=(8, 5))

    plt.boxplot(df[column].dropna())

    plt.title(f"Boxplot of {column}")
    plt.ylabel(column)

    plt.tight_layout()

    plt.savefig(
        f"results/boxplots/{column}_boxplot.png",
        dpi=300
    )

    plt.close()

print("Boxplots saved successfully!")

# ==========================================
# 10. PEARSON CORRELATION
# ==========================================

correlation_variables = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD",
    "pm25"
]

correlation_df = df[correlation_variables]

pearson_corr = correlation_df.corr(method="pearson")

print("\n12. Pearson correlation matrix:")
print(pearson_corr)

pearson_corr.to_csv("results/correlations/pearson_correlation.csv")

# ==========================================
# 11. SPEARMAN CORRELATION
# ==========================================

spearman_corr = correlation_df.corr(method="spearman")

print("\n13. Spearman correlation matrix:")
print(spearman_corr)

spearman_corr.to_csv("results/correlations/spearman_correlation.csv")

# ==========================================
# 12. CORRELATION HEATMAPS
# ==========================================

def create_heatmap(correlation, title, filename):

    fig, ax = plt.subplots(figsize=(11, 8))

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation.columns)

    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):

            ax.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    ax.set_title(title)

    fig.colorbar(image, ax=ax, label="Correlation")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


print("\n14. Creating correlation heatmaps...")

create_heatmap(
    pearson_corr,
    "Pearson Correlation Heatmap",
    "results/correlations/pearson_heatmap.png"
)

create_heatmap(
    spearman_corr,
    "Spearman Correlation Heatmap",
    "results/correlations/spearman_heatmap.png"
)

print("Correlation heatmaps saved successfully!")

# ==========================================
# 13. HIGHLY CORRELATED PREDICTORS
# ==========================================

print("\n15. Highly correlated predictors (|r| > 0.9):")

predictor_corr = correlation_df.drop(columns="pm25").corr()

high_correlations = []

for i in range(len(predictor_corr.columns)):
    for j in range(i + 1, len(predictor_corr.columns)):

        value = predictor_corr.iloc[i, j]

        if abs(value) > 0.9:

            high_correlations.append({
                "Variable_1": predictor_corr.columns[i],
                "Variable_2": predictor_corr.columns[j],
                "Correlation": value
            })

if high_correlations:
    high_corr_df = pd.DataFrame(high_correlations)
    print(high_corr_df)
    high_corr_df.to_csv(
        "results/correlations/high_correlations.csv",
        index=False
    )
else:
    print("No predictor pairs found with |r| > 0.9.")

# ==========================================
# 14. END
# ==========================================

print("\n" + "=" * 60)
print("GENERAL DATA EXPLORATION COMPLETED!")
print("Results saved in the results folder.")
print("=" * 60)


print("\nPotential outliers using the IQR method:")
print(f"{'Variable':<22}{'Outliers':<12}{'Percentage'}")
print("-" * 48)

for col in variables:
    x = df[col].dropna()

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = ((x < lower) | (x > upper)).sum()
    percentage = count / len(x) * 100

    print(f"{col:<22}{count:<12}{percentage:.2f}%")
