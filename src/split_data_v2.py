import pandas as pd


# Load V2 feature dataset
df = pd.read_csv("data/processed/model_data_v2.csv")

df["MatchDate"] = pd.to_datetime(df["MatchDate"])


# Chronological split
train = df[df["MatchDate"] < "2022-01-01"].copy()

validation = df[
    (df["MatchDate"] >= "2022-01-01") &
    (df["MatchDate"] < "2024-01-01")
].copy()

test = df[
    df["MatchDate"] >= "2024-01-01"
].copy()


# Save
train.to_csv("data/processed/train_v2.csv", index=False)
validation.to_csv("data/processed/validation_v2.csv", index=False)
test.to_csv("data/processed/test_v2.csv", index=False)


# Print results
print("=== V2 DATA SPLIT ===")

print("\nTraining:")
print(f"Rows: {len(train):,}")
print(
    f"Date range: "
    f"{train['MatchDate'].min().date()} → "
    f"{train['MatchDate'].max().date()}"
)

print("\nValidation:")
print(f"Rows: {len(validation):,}")
print(
    f"Date range: "
    f"{validation['MatchDate'].min().date()} → "
    f"{validation['MatchDate'].max().date()}"
)

print("\nTest:")
print(f"Rows: {len(test):,}")
print(
    f"Date range: "
    f"{test['MatchDate'].min().date()} → "
    f"{test['MatchDate'].max().date()}"
)


print("\n=== RESULT DISTRIBUTION ===")

print("\nTraining:")
print(train["FTResult"].value_counts(normalize=True).round(3))

print("\nValidation:")
print(validation["FTResult"].value_counts(normalize=True).round(3))

print("\nTest:")
print(test["FTResult"].value_counts(normalize=True).round(3))