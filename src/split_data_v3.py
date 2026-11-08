import pandas as pd

df = pd.read_csv("data/processed/model_data_v3.csv")
df["MatchDate"] = pd.to_datetime(df["MatchDate"])

# Chronological split
train = df[df["MatchDate"] < "2022-01-01"].copy()

validation = df[
    (df["MatchDate"] >= "2022-01-01") &
    (df["MatchDate"] < "2024-01-01")
].copy()

test = df[df["MatchDate"] >= "2024-01-01"].copy()

# Save
train.to_csv("data/processed/train_v3.csv", index=False)
validation.to_csv("data/processed/validation_v3.csv", index=False)
test.to_csv("data/processed/test_v3.csv", index=False)

print("=== V3 DATA SPLIT ===")

print(f"\nTrain: {len(train):,}")
print(f"Date range: {train['MatchDate'].min().date()} → {train['MatchDate'].max().date()}")

print(f"\nValidation: {len(validation):,}")
print(f"Date range: {validation['MatchDate'].min().date()} → {validation['MatchDate'].max().date()}")

print(f"\nTest: {len(test):,}")
print(f"Date range: {test['MatchDate'].min().date()} → {test['MatchDate'].max().date()}")

print("\n=== RESULT DISTRIBUTIONS ===")

print("\nTrain:")
print(train["FTResult"].value_counts(normalize=True).round(3))

print("\nValidation:")
print(validation["FTResult"].value_counts(normalize=True).round(3))

print("\nTest:")
print(test["FTResult"].value_counts(normalize=True).round(3))