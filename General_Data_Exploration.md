# General Data Exploration

## 1. Introduction

This section presents the general exploratory data analysis (EDA) of the cleaned PM2.5 dataset. The main objective is to understand the dataset, examine the distributions of numerical variables, identify missing values and potential outliers, and investigate the relationships between PM2.5 and the other variables.

The analysis was performed using Python, mainly with the Pandas, NumPy, and Matplotlib libraries.

## 2. Dataset Overview

The cleaned dataset contains **40,983 observations and 15 columns**. It includes air pollution measurements, meteorological variables, geographical information, dates, and sensor identifiers.

The main numerical variables examined include:

- PM2.5
- Aerosol Optical Depth (AOD)
- Nitrogen Dioxide (NO2)
- Carbon Monoxide (CO)
- Ozone (O3)
- Temperature
- Atmospheric pressure
- Wind components (U and V)

The dataset also contains geographical coordinates and sensor information.

## 3. Missing Values Analysis

The missing-value analysis revealed that most variables contain no missing values. However, three atmospheric variables have missing observations:

| Variable | Missing values | Percentage |
|---|---:|---:|
| NO2 | 2,407 | 5.87% |
| CO | 2,407 | 5.87% |
| O3 | 2,448 | 5.97% |

The total number of missing cells is **7,262**.

PM2.5 and the other remaining variables have no missing values.

These missing observations should be considered during future preprocessing and model development.

## 4. Descriptive Statistical Analysis

Descriptive statistics were calculated to understand the central tendency, variability, and distribution of the numerical variables.

For PM2.5, the main results are:

- Mean: 9.46
- Median: 7.85
- Standard deviation: 9.73
- Minimum: 0.0067
- Maximum: 484.98

The mean is higher than the median, suggesting a positively skewed distribution.

The AOD variable also presents a large difference between its mean (0.149) and maximum (5.475), indicating the presence of some unusually high observations.

The complete descriptive statistics are available in `Exploratory_Data.csv`.

## 5. Distribution Analysis

Histograms were generated for the nine main numerical variables to examine their distributions.

The PM2.5 histogram shows that most observations are concentrated at relatively low values, while a small number of observations have much higher concentrations.

The skewness analysis confirms this observation:

| Variable | Skewness |
|---|---:|
| PM2.5 | 17.83 |
| AOD | 7.61 |
| NO2 | 2.12 |
| CO | 1.40 |
| O3 | 1.15 |
| Temperature | -0.10 |

PM2.5 has an extremely positive skewness, while AOD also shows strong positive skewness. NO2, CO, and O3 are moderately positively skewed.

The kurtosis values also indicate heavy-tailed distributions for PM2.5 (651.67) and AOD (246.85).

These results suggest that the dataset contains a small number of extreme observations compared with the majority of measurements.

## 6. Outlier Analysis

Boxplots were generated for the numerical variables. Potential outliers were also identified using the Interquartile Range (IQR) method, with values outside the interval defined by Q1 − 1.5 × IQR and Q3 + 1.5 × IQR considered potential outliers.

The results are summarized below:

| Variable | Potential outliers | Percentage |
|---|---:|---:|
| PM2.5 | 1,613 | 3.94% |
| AOD | 1,114 | 2.72% |
| NO2 | 2,061 | 5.34% |
| CO | 572 | 1.48% |
| O3 | 1,219 | 3.16% |
| Temperature | 0 | 0.00% |
| Pressure | 3,179 | 7.76% |
| Wind U | 2,133 | 5.20% |
| Wind V | 2,234 | 5.45% |

Atmospheric pressure has the highest percentage of potential outliers, while temperature has none according to this method.

It is important to note that statistical outliers are not necessarily errors. Therefore, these observations were not automatically removed and should be investigated before any further treatment.

## 7. Correlation Analysis

Pearson and Spearman correlation matrices were calculated to examine the relationships between PM2.5 and the other numerical variables.

The strongest correlations with PM2.5 are:

| Variable | Pearson | Spearman |
|---|---:|---:|
| NO2 | 0.159 | 0.247 |
| Pressure | 0.113 | 0.167 |
| AOD | 0.094 | 0.180 |
| O3 | -0.087 | -0.135 |
| Wind U | -0.079 | -0.164 |

The results show that NO2 has the strongest positive association with PM2.5 in both correlation methods, although the relationship remains weak.

AOD and atmospheric pressure also show weak positive associations, while O3 and wind U show weak negative associations.

Pearson measures linear relationships, whereas Spearman measures monotonic relationships based on ranks. Their differences suggest that some relationships may not be purely linear.

Additionally, no pairs of predictors were found to have an absolute Pearson correlation greater than 0.9. Therefore, no highly correlated predictor pairs were identified using this threshold.

Correlation does not imply causation, and weak individual correlations do not necessarily mean that a variable will be useless for future prediction models.

## 8. PM2.5 Threshold Analysis

The distribution of PM2.5 was further examined using two concentration thresholds:

| Threshold | Number of observations | Percentage |
|---|---:|---:|
| Above 25 µg/m³ | 1,223 | 2.98% |
| Above 50 µg/m³ | 117 | 0.29% |

Only a small proportion of observations exceed these thresholds. This indicates that high PM2.5 concentrations are relatively rare in the dataset.

This imbalance should be considered during future model evaluation because prediction errors on rare high-concentration observations may be important.

## 9. Conclusion

The general exploratory analysis provided an overview of the cleaned PM2.5 dataset and its main characteristics.

The analysis revealed that PM2.5 has an extremely right-skewed distribution, with most observations concentrated at low values and a small number of very high measurements. Missing values were mainly found in NO2, CO, and O3. The IQR method also identified potential outliers in several variables, which should be investigated rather than automatically removed.

The correlation analysis showed that NO2 has the strongest positive association with PM2.5, although all individual correlations with the target are relatively weak. No highly correlated predictor pairs were identified using the absolute Pearson correlation threshold of 0.9.

Overall, these findings provide useful information for the next stages of the project, particularly data preprocessing, feature analysis, and PM2.5 forecasting.
