import pandas as pd

# Load feature dataset
df = pd.read_csv("data/processed/model_data.csv")

# Convert date
df["MatchDate"] = pd.to_datetime(df["MatchDate"])

# Chronological split
train = df[df["MatchDate"] < "2022-01-01"].copy()

validation = df[
    (df["MatchDate"] >= "2022-01-01") &
    (df["MatchDate"] < "2024-01-01")
].copy()

test = df[df["MatchDate"] >= "2024-01-01"].copy()

# Save datasets
train.to_csv("data/processed/train.csv", index=False)
validation.to_csv("data/processed/validation.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

# Print results
print("=== DATA SPLIT ===")

print("\nTraining:")
print(f"Rows: {len(train):,}")
print(f"Date range: {train['MatchDate'].min().date()} → {train['MatchDate'].max().date()}")

print("\nValidation:")
print(f"Rows: {len(validation):,}")
print(f"Date range: {validation['MatchDate'].min().date()} → {validation['MatchDate'].max().date()}")

print("\nTest:")
print(f"Rows: {len(test):,}")
print(f"Date range: {test['MatchDate'].min().date()} → {test['MatchDate'].max().date()}")

print("\n=== RESULT DISTRIBUTION ===")

print("\nTraining:")
print(train["FTResult"].value_counts(normalize=True).round(3))

print("\nValidation:")
print(validation["FTResult"].value_counts(normalize=True).round(3))

print("\nTest:")
print(test["FTResult"].value_counts(normalize=True).round(3))