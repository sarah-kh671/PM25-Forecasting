import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def run_temporal_eda(input_path="data/cleaned_pm25.csv", output_path="results/Exploratory_temporal.csv"):
    print("Loading cleaned data...")
    df = pd.read_csv(input_path)
    
    # Convert timestamp column to datetime objects
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

    # Extract month and year for seasonal analysis
    df['month'] = df['datetime_utc'].dt.month
    df['year'] = df['datetime_utc'].dt.year 

    # Compute and save monthly pollution trends (seasonality)
    print("Computing monthly pollution trends...")
    monthly_avg = df.groupby(['id', 'month'])['pm25'].mean().reset_index()
    monthly_avg.to_csv("results/monthly_pollution_trends.csv", index=False)
    
    temporal_results = []
    print("Computing advanced temporal features per station...")
    
    # Analyze each monitoring station independently
    for station_id, group in df.groupby('id'):
        group = group.sort_values('datetime_utc')
        
        # Check for duplicate timestamps within the same station
        n_duplicate_timestamps = group['datetime_utc'].duplicated().sum()
        
        # Time intervals regularity check
        time_diffs = group['datetime_utc'].diff().dt.total_seconds()
        median_interval = time_diffs.median()
        std_interval = time_diffs.std()
        
        # Autocorrelation for PM2.5 at specific lags (1, 7, 24 days/steps)
        acf_lag1 = group['pm25'].autocorr(lag=1)
        acf_lag7 = group['pm25'].autocorr(lag=7)
        acf_lag24 = group['pm25'].autocorr(lag=24)
        
        n_records = len(group)
        n_days_above_25 = (group['pm25'] > 25).sum()
        n_days_above_50 = (group['pm25'] > 50).sum()
        
        pct_above_25 = (n_days_above_25 / n_records) * 100 if n_records > 0 else 0
        pct_above_50 = (n_days_above_50 / n_records) * 100 if n_records > 0 else 0
        
        # Track data gaps where time gap exceeds normal sampling periods (> 1.5 days)
        time_diffs_days = time_diffs / 86400.0
        n_gaps = (time_diffs_days > 1.5).sum()

        # Stationarity check using Augmented Dickey-Fuller (ADF) test (if enough samples)
        adf_pvalue = np.nan
        if n_records > 30:
            try:
                # Drop NaN values for ADF test
                pm25_series = group['pm25'].dropna()
                result = adfuller(pm25_series)
                adf_pvalue = result[1] # p-value indicates stationarity if < 0.05
            except Exception:
                pass

        temporal_results.append({
            'station_id': station_id,
            'n_records': n_records,
            'n_duplicate_timestamps': n_duplicate_timestamps,
            't_min': group['datetime_utc'].min(),
            't_max': group['datetime_utc'].max(),
            'sampling_interval_median_sec': median_interval,
            'sampling_interval_std_sec': std_interval,
            'acf_lag1': acf_lag1,
            'acf_lag7': acf_lag7,
            'acf_lag24': acf_lag24,
            'n_days_above_25': n_days_above_25,
            'n_days_above_50': n_days_above_50,
            'pct_above_25': pct_above_25,
            'pct_above_50': pct_above_50,
            'n_gaps': n_gaps,
            'adf_pvalue': adf_pvalue
        })

    df_temporal_summary = pd.DataFrame(temporal_results)

    # Save summary to disk
    df_temporal_summary.to_csv(output_path, index=False)
    print(f"Temporal EDA summary saved successfully to {output_path}!")
    print(df_temporal_summary.head())

if __name__ == "__main__":
    run_temporal_eda()