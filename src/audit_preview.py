import pandas as pd
import os

# Read original dataset
df = pd.read_csv("combined_gee_pm25.csv")

# Count invalid rows
missing = df["pm25"].isna().sum()
zero_or_negative = (df["pm25"] <= 0).sum()
too_high = (df["pm25"] >= 500).sum()

# Clean PM2.5
clean_df = df[
    (df["pm25"].notna()) &
    (df["pm25"] > 0) &
    (df["pm25"] < 500)
].copy()

# Create results folder
os.makedirs("Person1_Results", exist_ok=True)

# Save cleaned dataset
clean_df.to_csv(
    "Person1_Results/cleaned_pm25.csv",
    index=False
)

# Save cleaning report
report = pd.DataFrame({
    "Item": [
        "Original rows",
        "Missing PM2.5",
        "PM2.5 <= 0",
        "PM2.5 >= 500",
        "Removed rows",
        "Remaining rows"
    ],
    "Count": [
        len(df),
        missing,
        zero_or_negative,
        too_high,
        len(df) - len(clean_df),
        len(clean_df)
    ]
})

report.to_csv(
    "Person1_Results/cleaning_report.csv",
    index=False
)

print(report)
print("\nCleaning completed!")
import pandas as pd

df = pd.read_csv("Person1_Results/cleaned_pm25.csv")

print("\n--- STATION AUDIT ---")
print("Unique station IDs:", df["id"].nunique())
print("Unique sensor names:", df["sensor_name"].nunique())

print("\nRepeated date + station ID:", 
      df.duplicated(subset=["date", "id"]).sum())

print("Repeated date + sensor name:",
      df.duplicated(subset=["date", "sensor_name"]).sum())

print("\n--- DATE AUDIT ---")
print("Unique dates:", df["date"].nunique())
print("First date:", df["date"].min())
print("Last date:", df["date"].max())

print("\nRows per station:")
print(df.groupby("id").size().describe())
import pandas as pd

df = pd.read_csv("Person1_Results/cleaned_pm25.csv")

duplicates = df[
    df.duplicated(
        subset=["date", "sensor_name"],
        keep=False
    )
].sort_values(["sensor_name", "date"])

print(duplicates[
    ["date", "id", "sensor_name", "latitude", "longitude", "pm25"]
].to_string(index=False))
import pandas as pd

df = pd.read_csv("Person1_Results/cleaned_pm25.csv")

print("\n--- GEOGRAPHICAL AUDIT ---")

print("Latitude minimum:", df["latitude"].min())
print("Latitude maximum:", df["latitude"].max())

print("Longitude minimum:", df["longitude"].min())
print("Longitude maximum:", df["longitude"].max())

print("Missing coordinates:",
      df[["latitude", "longitude"]].isna().sum().sum())

print("Unique coordinate pairs:",
      df[["latitude", "longitude"]].drop_duplicates().shape[0])

print("Unique station IDs:", df["id"].nunique())

print("\n--- POSSIBLE LEAKAGE CHECK ---")

print("Columns:", df.columns.tolist())

print("\nCorrelation with PM2.5:")
print(df.select_dtypes(include="number").corr()["pm25"].sort_values(
    ascending=False
))