import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

MIN_RECORDS = 100  # stations below this are flagged in notes


def lag_corr(s, k):
    """Correlation between day t and day t-k, using only pairs where both days exist."""
    a, b = s, s.shift(k)
    m = a.notna() & b.notna()
    if m.sum() < 10:
        return np.nan
    return a[m].corr(b[m])


def longest_run(s):
    """Longest stretch of consecutive days without gaps."""
    block = s.isna().cumsum()
    runs = s.dropna().groupby(block[s.notna()])
    return max((r for _, r in runs), key=len, default=pd.Series(dtype=float))


def run_temporal_eda(input_path="data/processed/cleaned_pm25.csv", output_path="results/Exploratory_temporal.csv"):
    print("Loading cleaned data...")
    df = pd.read_csv(input_path)

    # Daily timeline: one value per station per day
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc']).dt.normalize()

    # Extract month and year for seasonal analysis
    df['month'] = df['datetime_utc'].dt.month
    df['year'] = df['datetime_utc'].dt.year

    # Compute and save monthly pollution trends (seasonality)
    print("Computing monthly pollution trends...")
    monthly_avg = df.groupby(['id', 'month'])['pm25'].mean().reset_index()
    monthly_avg.to_csv("results/monthly_pollution_trends.csv", index=False)

    # NO2 / CO / O3 columns (matched by name, any capitalisation)
    gas_cols = [c for c in df.columns if c.lower() in ("no2", "co", "o3")]

    temporal_results = []
    print("Computing advanced temporal features per station...")

    # Analyze each monitoring station independently
    for station_id, group in df.groupby('id'):
        group = group.sort_values('datetime_utc')

        # Check for duplicate timestamps within the same station, then keep one per day
        n_duplicate_timestamps = group['datetime_utc'].duplicated().sum()
        group = group.drop_duplicates(subset='datetime_utc')

        y = group.set_index('datetime_utc')['pm25']
        n_records = len(y)

        # Regular daily timeline (missing days become NaN) so that "lag 1" = 1 day
        daily = y.reindex(pd.date_range(y.index.min(), y.index.max(), freq='D'))
        pct_missing_intervals = 100 * (1 - n_records / len(daily))

        # Time intervals regularity check (between observed records)
        time_diffs = y.index.to_series().diff().dt.total_seconds().dropna()
        median_interval = time_diffs.median() if len(time_diffs) else np.nan
        std_interval = time_diffs.std() if len(time_diffs) > 1 else np.nan

        # Autocorrelation on the daily timeline (only pairs where both days exist)
        acf_lag1 = lag_corr(daily, 1)
        acf_lag7 = lag_corr(daily, 7)
        acf_lag24 = lag_corr(daily, 24)

        n_days_above_25 = int((y > 25).sum())
        n_days_above_50 = int((y > 50).sum())
        pct_days_above_25 = (n_days_above_25 / n_records) * 100 if n_records > 0 else 0
        pct_days_above_50 = (n_days_above_50 / n_records) * 100 if n_records > 0 else 0

        # Series for the tests: longest gap-free stretch, else observed values in order
        run = longest_run(daily).values
        gap_free = len(run) >= 30
        if not gap_free:
            run = y.values
        enough = len(run) >= 30

        ljung_box_p = adf_p = kpss_p = np.nan
        if enough:
            try:
                ljung_box_p = acorr_ljungbox(run, lags=[7])['lb_pvalue'].iloc[0]
                adf_p = adfuller(run)[1]  # < 0.05 -> stationary
                kpss_p = kpss(run, nlags='auto')[1]  # > 0.05 -> stationary
            except Exception:
                pass

        # Linear trend on observed values
        trend_slope = trend_pval = np.nan
        if n_records >= 10:
            days = (y.index - y.index.min()).days.values
            tr = stats.linregress(days, y.values)
            trend_slope, trend_pval = tr.slope, tr.pvalue

        # Seasonal amplitude: max - min of monthly means
        monthly = y.groupby(y.index.to_period('M')).mean()
        seasonal_amplitude = monthly.max() - monthly.min() if len(monthly) > 1 else np.nan

        # Target drift: first half vs second half of the record (KS test)
        drift_stat = drift_pval = np.nan
        if n_records >= 20:
            ks = stats.ks_2samp(y.values[:n_records // 2], y.values[n_records // 2:])
            drift_stat, drift_pval = ks.statistic, ks.pvalue

        row = {
            'node_id': station_id,
            'n_records': n_records,
            'n_duplicate_timestamps': n_duplicate_timestamps,
            't_min': y.index.min().date(),
            't_max': y.index.max().date(),
            'sampling_interval_median_sec': median_interval,
            'sampling_interval_std_sec': std_interval,
            'pct_missing_intervals': pct_missing_intervals,
            'acf_lag1': acf_lag1,
            'acf_lag7': acf_lag7,
            'acf_lag24': acf_lag24,
            'pacf_lag1': acf_lag1,  # PACF at lag 1 equals ACF at lag 1
            'ljung_box_p': ljung_box_p,
            'adf_p': adf_p,
            'kpss_p': kpss_p,
            'trend_slope_per_day': trend_slope,
            'trend_pval': trend_pval,
            'seasonal_amplitude': seasonal_amplitude,
            'target_drift_stat': drift_stat,
            'target_drift_pval': drift_pval,
            'n_days_above_25': n_days_above_25,
            'n_days_above_50': n_days_above_50,
            'pct_days_above_25': pct_days_above_25,
            'pct_days_above_50': pct_days_above_50,
        }

        # NO2 / CO / O3: % missing by month (JSON string)
        for c in gas_cols:
            by_month = group.groupby(group['datetime_utc'].dt.to_period('M'))[c].apply(
                lambda x: round(100 * x.isna().mean(), 1))
            row[f'{c.lower()}_pct_missing_by_month'] = json.dumps({str(k): v for k, v in by_month.items()})

        # Recommended CV: autocorrelated station -> temporal, otherwise spatial
        row['cv_strategy_recommended'] = 'temporal' if (not np.isnan(ljung_box_p) and ljung_box_p < 0.05) else 'spatial'
        row['gap_between_train_test_sec'] = 86400.0

        notes = []
        if n_records < MIN_RECORDS:
            notes.append(f'fewer than {MIN_RECORDS} records: exclude')
        if not enough:
            notes.append('fewer than 30 observations: ADF/KPSS/Ljung-Box not computed')
        elif not gap_free:
            notes.append('no 30-day gap-free stretch: tests run on observed values, gaps ignored')
        row['notes'] = '; '.join(notes)

        temporal_results.append(row)

    df_temporal_summary = pd.DataFrame(temporal_results)

    # Save summary to disk
    df_temporal_summary.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Temporal EDA summary saved successfully to {output_path}!")
    print(df_temporal_summary.head())


if __name__ == "__main__":
    run_temporal_eda()